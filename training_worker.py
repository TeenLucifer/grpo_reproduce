#!/usr/bin/env python3
"""
训练进程脚本：负责接收采样数据并进行模型训练
通过ZeroMQ接收采样进程的数据，支持DeepSpeed分布式训练
"""

import os
import time
import torch
import zmq
import yaml
import json
import pickle
import threading
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.distributed as dist
from data_types import Episode
from utils import group_advantages, grpo_loss, train_accuracy, get_batch_log_probs
import deepspeed
import numpy as np

class TrainingWorker:
    def __init__(self, config: dict):
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }

        self.gpu_id = config["gpu"]["training_gpu"]
        self.pretrained_model_path = config["model"]["pretrained_model_path"]
        self.ref_model_path = config["model"]["ref_model_path"]
        self.dtype = dtype_map.get(config["model"]["dtype"], torch.bfloat16)
        self.data_path = config["data"]["data_path"]
        self.max_gen_len = config["data"]["max_gen_len"]
        self.batch_size = config["training"]["batch_size"]
        self.num_answers_per_question = config["training"]["num_answers_per_question"]
        self.num_questions_per_batch = self.batch_size // self.num_answers_per_question
        self.zmq_data_port = config["communication"]["data_port"]
        self.zmq_sync_port = config["communication"]["sync_port"]
        self.use_deepspeed = config["deepspeed"]["enabled"]
        self.ds_config_path = config["deepspeed"]["config_path"]

        # DeepSpeed多进程相关属性
        self.rank = 0
        self.world_size = 1
        self.is_main_process = True

        self.setup_model()
        self.setup_zmq()
        self.stop_event = threading.Event()

    def setup_model(self):
        """初始化模型和优化器"""
        # 初始化新策略模型
        self.new_policy_model = AutoModelForCausalLM.from_pretrained(
            self.pretrained_model_path, 
            dtype=self.dtype, 
            _attn_implementation="sdpa"
        )
        self.new_policy_model.train()
        self.new_policy_model.requires_grad_(True)

        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_path, padding_side='left')

        # DeepSpeed配置和初始化
        if self.use_deepspeed:
            print("正在使用DeepSpeed进行优化训练...")

            dist.init_process_group(backend='gloo')  # autodl的vgpu没法用nccl通信, 需要设置为gloo
                        
            # 获取当前进程的rank和world_size
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.is_main_process = (self.rank == 0)
            
            print(f"DeepSpeed进程初始化 - Rank: {self.rank}, World Size: {self.world_size}, Is Main Process: {self.is_main_process}")
            
            # 加载DeepSpeed配置
            try:
                with open(self.ds_config_path, 'r') as f:
                    ds_config = json.load(f)
                print(f"成功加载DeepSpeed配置: {self.ds_config_path}")
            except FileNotFoundError:
                print(f"警告: DeepSpeed配置文件 {self.ds_config_path}")
                raise
            except json.JSONDecodeError as e:
                print(f"错误: DeepSpeed配置文件格式错误: {e}")
                raise

            # 加载模型参数
            model_parameters = list(self.new_policy_model.parameters())

            # 初始化DeepSpeed引擎
            self.model_engine, self.optimizer, _, self.scheduler = deepspeed.initialize(
                model=self.new_policy_model,
                model_parameters=model_parameters,
                config=ds_config
            )

            self.device=self.model_engine.device
            print(f"DeepSpeed初始化完成，世界大小: {self.model_engine.world_size}")
        else:
            # 原有的优化器和调度器
            self.optimizer = AdamW(self.new_policy_model.parameters(), lr=1e-5, weight_decay=0.01)
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=1000, eta_min=1e-6)
            self.model_engine = None
            self.device = torch.device(f'cuda:{self.gpu_id}' if torch.cuda.is_available() else 'cpu')

        # 梯度裁剪参数
        self.max_grad_norm = 1.0
        print("模型和优化器初始化完成")

    def setup_zmq(self):
        """初始化ZeroMQ通信"""
        self.context = zmq.Context()
        
        # 只有主进程(rank=0)连接ZMQ，避免多进程竞争
        if self.is_main_process:
            # 数据接收socket（PULL模式）
            self.data_receiver = self.context.socket(zmq.PULL)
            self.data_receiver.connect(f"tcp://localhost:{self.zmq_data_port}")
            print(f"ZeroMQ数据接收端口连接: {self.zmq_data_port}")

            # 模型参数发送socket（PUSH模式）
            self.sync_sender = self.context.socket(zmq.PUSH)
            self.sync_sender.connect(f"tcp://localhost:{self.zmq_sync_port}")
            print(f"ZeroMQ同步发送端口连接: {self.zmq_sync_port}")
        else:
            # 非主进程不需要ZMQ连接
            self.data_receiver = None
            self.sync_sender = None
            print(f"Rank {self.rank} 进程跳过ZMQ连接（由rank 0处理）")

        # 数据接收socket（PULL模式）
        self.data_receiver = self.context.socket(zmq.PULL)
        self.data_receiver.connect(f"tcp://localhost:{self.zmq_data_port}")
        print(f"ZeroMQ数据接收端口连接: {self.zmq_data_port}")

        # 模型参数发送socket（PUSH模式）
        self.sync_sender = self.context.socket(zmq.PUSH)
        self.sync_sender.connect(f"tcp://localhost:{self.zmq_sync_port}")
        print(f"ZeroMQ同步发送端口连接: {self.zmq_sync_port}")

    def deserialize_episodes(self, serialized_data):
        """反序列化episodes数据"""
        episodes = []
        for data in serialized_data:
            episode = Episode(
                prefix=data['prefix'],
                prefix_tokens=data['prefix_tokens'],
                prefix_token_ids=data['prefix_token_ids'],
                generated_token_ids=data['generated_token_ids'],
                whole_token_ids=data['whole_token_ids'],
                is_finished=data['is_finished'],
                text=data['text'],
                reward=data['reward'],
                reward_info=data['reward_info'],
                old_policy_log_probs=data['old_policy_log_probs'],
                ref_policy_log_probs=data['ref_policy_log_probs']
            )
            episodes.append(episode)
        return episodes

    def split_episodes_by_rank(self, episodes):
        """将episodes按rank进行分割，保持组完整性"""
        if self.world_size <= 1:
            return episodes
            
        total_questions = len(episodes) // self.num_answers_per_question
        questions_per_rank = total_questions // self.world_size
        
        # 确保能整除，保持组完整性
        if total_questions % self.world_size != 0:
            # 如果不能整除，向下取整，多余的组由rank=0处理
            questions_per_rank = total_questions // self.world_size
            if self.rank == 0:
                start_question = 0
                end_question = questions_per_rank + (total_questions % self.world_size)
            else:
                start_question = (total_questions % self.world_size) + (self.rank - 1) * questions_per_rank
                end_question = start_question + questions_per_rank
        else:
            start_question = self.rank * questions_per_rank
            end_question = start_question + questions_per_rank
        
        # 转换为episode索引
        start_idx = start_question * self.num_answers_per_question
        end_idx = end_question * self.num_answers_per_question
        
        split_episodes = episodes[start_idx:end_idx]
        
        print(f"Rank {self.rank}: 处理问题 {start_question}-{end_question-1}, "
              f"Episodes {start_idx}-{end_idx-1}, 共 {len(split_episodes)} 个")
        
        return split_episodes

    def broadcast_episodes(self, episodes):
        """将episodes数据从rank=0广播到所有其他rank"""
        if self.use_deepspeed and self.world_size > 1:
            # 先进行数据分割，再广播分割后的数据
            if self.is_main_process:
                # rank=0分割数据并决定每个rank处理哪些组
                all_split_data = []
                for rank in range(self.world_size):
                    rank_episodes = self.split_episodes_by_rank(episodes)
                    all_split_data.append(pickle.dumps(rank_episodes))
                
                # 广播数据大小信息
                data_sizes = [len(data) for data in all_split_data]
                sizes_tensor = torch.tensor(data_sizes, dtype=torch.long, device=self.device)
            else:
                sizes_tensor = torch.zeros(self.world_size, dtype=torch.long, device=self.device)
            
            # 广播所有rank的数据大小
            dist.broadcast(sizes_tensor, src=0)
            data_size = sizes_tensor[self.rank].item()
            
            # 准备数据tensor
            if self.is_main_process:
                data_tensor = torch.zeros(data_size, dtype=torch.uint8, device=self.device)
                # 获取当前rank的数据
                rank_data = all_split_data[self.rank]
                np_data = np.frombuffer(rank_data, dtype=np.uint8)
                data_tensor[:len(np_data)] = torch.from_numpy(np_data.copy()).to(self.device)
            else:
                data_tensor = torch.zeros(data_size, dtype=torch.uint8, device=self.device)
            
            # 广播实际数据（每个rank接收自己的部分）
            dist.broadcast(data_tensor, src=0)
            
            # 反序列化数据
            serialized_data = data_tensor.cpu().numpy().tobytes()
            episodes = pickle.loads(serialized_data)

            return episodes
        else:
            # 单进程模式，直接返回
            return episodes

    def train_step(self, episodes):
        """执行一个训练步骤"""
        prefix_len = len(episodes[0].whole_token_ids) - len(episodes[0].generated_token_ids)

        # 计算新策略的概率分布
        batch_token_ids = torch.tensor([episode.whole_token_ids for episode in episodes], dtype=torch.long, device=self.device)
        attention_mask = (batch_token_ids != self.tokenizer.pad_token_id).long()

        if self.use_deepspeed and self.model_engine is not None:
            new_policy_log_probs = get_batch_log_probs(
                model=self.model_engine,
                batch_token_ids=batch_token_ids,
                attention_mask=attention_mask,
                enable_grad=True  # 新策略训练，需要梯度
            )
        else:
            new_policy_log_probs = get_batch_log_probs(
                model=self.new_policy_model,
                batch_token_ids=batch_token_ids,
                attention_mask=attention_mask,
                enable_grad=True  # 新策略训练，需要梯度
            )

        # 计算优势函数
        rewards = torch.tensor([episode.reward for episode in episodes], dtype=self.dtype, device=self.device)
        advantages = group_advantages(rewards=rewards, num_answers_per_question=self.num_answers_per_question).to(self.device)

        # 计算grpo算法的loss
        ref_policy_log_probs = torch.tensor(np.array([episode.ref_policy_log_probs for episode in episodes]), dtype=self.dtype, device=self.device)
        old_policy_log_probs = torch.tensor(np.array([episode.old_policy_log_probs for episode in episodes]), dtype=self.dtype, device=self.device)

        loss = grpo_loss(
            ref_policy_log_probs=ref_policy_log_probs,
            old_policy_log_probs=old_policy_log_probs,
            new_policy_log_probs=new_policy_log_probs,
            attention_mask=attention_mask,
            advantages=advantages,
            prefix_len=prefix_len
        )

        # 反向传播和优化步骤
        if self.use_deepspeed and self.model_engine is not None:
            # DeepSpeed优化步骤
            self.model_engine.backward(loss)
            self.model_engine.step()
        else:
            # 原有优化步骤
            self.optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.new_policy_model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()

        return loss

    def sync_model_parameters(self, train_step):
        """同步模型参数到采样进程"""
        try:
            if self.use_deepspeed and self.model_engine is not None:
                # DeepSpeed获取模型状态字典
                state_dict = self.model_engine.module.state_dict()
            else:
                # 原有模型状态字典
                state_dict = self.new_policy_model.state_dict()

            # 序列化并发送
            state_dict_data = pickle.dumps(state_dict)
            self.sync_sender.send(state_dict_data)
            print(f"第 {train_step} 步训练后发送模型参数同步请求")

        except zmq.Again:
            print("同步队列已满，跳过此次同步")
        except Exception as e:
            print(f"模型参数同步错误: {e}")

    def save_checkpoint(self, train_step, loss):
        """保存模型检查点"""
        if train_step % 100 == 0:
            if self.use_deepspeed and self.model_engine is not None:
                # DeepSpeed检查点保存
                checkpoint_path = f"./checkpoints/deepspeed_model_step_{train_step}"
                os.makedirs("./checkpoints", exist_ok=True)
                self.model_engine.save_checkpoint(checkpoint_path, tag=f"step_{train_step}")
                print(f"DeepSpeed模型检查点已保存: {checkpoint_path}")
            else:
                # 原有检查点保存
                checkpoint_path = f"./checkpoints/model_step_{train_step}.pt"
                os.makedirs("./checkpoints", exist_ok=True)
                torch.save({
                    'step': train_step,
                    'model_state_dict': self.new_policy_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'loss': loss.item(),
                }, checkpoint_path)
                print(f"模型检查点已保存: {checkpoint_path}")

    def run(self):
        """主运行循环"""
        print(f"Rank {self.rank} 开始训练循环...")
        train_step = 0
        sync_interval = 25  # 每25个训练步骤同步一次

        try:
            while not self.stop_event.is_set():
                try:
                    episodes = None
                    
                    # 数据接收逻辑：只有rank=0从ZMQ接收，其他rank等待广播
                    if self.is_main_process:
                        # 主进程从ZMQ接收数据
                        if self.data_receiver and self.data_receiver.poll(100):  # 100ms超时
                            data = self.data_receiver.recv()
                            serialized_episodes = pickle.loads(data)
                            episodes = self.deserialize_episodes(serialized_episodes)
                            print(f"Rank 0 接收到数据，批次大小: {len(episodes)}")
                        else:
                            # 没有数据，继续循环
                            time.sleep(0.01)
                            continue
                    
                    # 广播数据到所有rank
                    if self.use_deepspeed and self.world_size > 1:
                        episodes = self.broadcast_episodes(episodes)
                    
                    # 如果所有rank都没有数据，继续循环
                    if episodes is None or len(episodes) == 0:
                        time.sleep(0.01)
                        continue
                    
                    print(f"Rank {self.rank} 开始训练步骤，数据批次大小: {len(episodes)}")

                    # 执行训练步骤
                    loss = self.train_step(episodes)
                    train_step += 1
                    
                    # 定期同步模型参数（只有主进程需要同步到采样进程）
                    if self.is_main_process and train_step % sync_interval == 0:
                        self.sync_model_parameters(train_step)

                    # 定期打印训练信息（只有主进程打印，避免重复输出）
                    if self.is_main_process and train_step % 10 == 0:
                        format_accuracy, answer_accuracy = train_accuracy(episodes=episodes)
                        if self.use_deepspeed and self.model_engine is not None:
                            current_lr = self.model_engine.get_lr()[0]
                        else:
                            current_lr = self.scheduler.get_last_lr()[0]
                        print(f"训练步骤 {train_step}, Loss: {loss.item():.4f}, LR: {current_lr:.2e}, F_Acc: {format_accuracy}, A_acc: {answer_accuracy}")

                    # 定期保存检查点（只有主进程保存）
                    if self.is_main_process:
                        self.save_checkpoint(train_step, loss)

                except zmq.Again:
                    # 没有数据，继续循环
                    continue
                except Exception as e:
                    print(f"Rank {self.rank} 训练步骤错误: {e}")
                    continue

                # 短暂休眠避免CPU占用过高
                time.sleep(0.01)

        except KeyboardInterrupt:
            print(f"Rank {self.rank} 训练进程收到中断信号")
        except Exception as e:
            print(f"Rank {self.rank} 训练进程错误: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """清理资源"""
        print(f"Rank {self.rank} 清理训练进程资源...")
        self.stop_event.set()

        # 只有主进程有ZMQ连接需要清理
        if self.is_main_process:
            if hasattr(self, 'data_receiver') and self.data_receiver:
                self.data_receiver.close()
            if hasattr(self, 'sync_sender') and self.sync_sender:
                self.sync_sender.close()
        
        if hasattr(self, 'context'):
            self.context.term()
        
        print(f"Rank {self.rank} 训练进程已清理完成")


def main():
    """主函数"""
    config_path = "./config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 创建训练进程实例
    worker = TrainingWorker(config=config)
    
    print("=== GRPO训练进程 ===")
    print(f"数据端口: {config["communication"]["data_port"]}")
    print(f"同步端口: {config["communication"]["sync_port"]}")
    print(f"DeepSpeed: {'启用' if config["deepspeed"]["enabled"] else '禁用'}")
    print(f"当前进程: Rank {worker.rank}/{worker.world_size-1}, 主进程: {worker.is_main_process}")
    
    print("初始化成功")
    worker.run()


if __name__ == "__main__":
    main()