#!/usr/bin/env python3
"""
训练进程脚本：负责接收采样数据并进行模型训练
通过ZeroMQ接收采样进程的数据，支持DeepSpeed分布式训练
"""

import time
import torch
import zmq
import yaml
import json
import pickle
import threading
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
import torch.distributed as dist
from data_types import Gsm8kTasksDataset, Episode
from utils import group_advantages, grpo_loss, gspo_loss, train_accuracy, get_batch_log_probs, sample_trajectory, reward_function
import deepspeed
import numpy as np
from peft import LoraConfig, get_peft_model

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
        self.train_batch_size = config["data"]["train_batch_size"]
        self.sample_batch_size = config["data"]["sample_batch_size"]
        self.test_size = config["data"]["test_size"]
        self.test_batch_size = config["data"]["test_batch_size"]
        self.data_path = config["data"]["data_path"]
        self.num_answers_per_question = config["data"]["num_answers_per_question"]
        self.num_questions_per_batch = self.train_batch_size // self.num_answers_per_question
        self.use_gspo = config["training"]["use_gspo"]
        self.eval_interval = config["training"]["eval_interval"]
        self.sync_interval = config["training"]["sync_interval"]
        self.zmq_data_port = config["communication"]["data_port"]
        self.ds_config_path = config["deepspeed"]["config_path"]
        self.ckpt_dir = Path(config["checkpoint"]["ckpt_dir"])
        self.ckpt_file = config["checkpoint"]["ckpt_file"]
        self.use_lora = config["lora"]["enabled"]
        self.lora_rank = config["lora"]["rank"]
        self.lora_alpha = config["lora"]["alpha"]
        self.lora_lr = float(config["lora"]["learning_rate"])
        self.lora_dropoutp = config["lora"]["dropout"]
        self.lora_adapter_dir = Path(config["lora"]["adapter_dir"])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.lora_adapter_dir.mkdir(parents=True, exist_ok=True)

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
        # 启用梯度检查点
        self.new_policy_model.gradient_checkpointing_enable()
        print("梯度检查点已启用")

        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_path, padding_side='left')
        # 启用LoRA配置和初始化
        if self.use_lora:
            lora_config = LoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                target_modules="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj".split(","),
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.new_policy_model = get_peft_model(self.new_policy_model, lora_config)

        # DeepSpeed配置和初始化
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

        if self.use_lora:
            ds_config['optimizer']['params']['lr'] = self.lora_lr

        # 初始化DeepSpeed引擎
        self.model_engine, self.optimizer, _, self.scheduler = deepspeed.initialize(
            model=self.new_policy_model,
            model_parameters=model_parameters,
            config=ds_config
        )

        self.device=self.model_engine.device
        print(f"DeepSpeed初始化完成，世界大小: {self.model_engine.world_size}")

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
        else:
            # 非主进程不需要ZMQ连接
            self.data_receiver = None
            print(f"Rank {self.rank} 进程跳过ZMQ连接（由rank 0处理）")

        # 数据接收socket（PULL模式）
        self.data_receiver = self.context.socket(zmq.PULL)
        self.data_receiver.connect(f"tcp://localhost:{self.zmq_data_port}")
        print(f"ZeroMQ数据接收端口连接: {self.zmq_data_port}")

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

    def broadcast_episodes(self, episodes):
        """将episodes数据从rank=0广播到所有其他rank"""
        if self.world_size > 1:
            if self.is_main_process:
                serialized_data = pickle.dumps(episodes)
                np_data = np.frombuffer(serialized_data, dtype=np.uint8)
                size_tensor = torch.from_numpy(np.array([len(np_data)], dtype=np.int64)).to(self.device)
                episodes_tensor = torch.from_numpy(np_data.copy()).to(self.device)
            else:
                size_tensor = torch.zeros(1, dtype=torch.int64, device=self.device)
                episodes_tensor = None

            # 广播数据长度
            dist.broadcast(size_tensor, src=0)
            data_size = size_tensor.item()

            if not self.is_main_process:
                episodes_tensor = torch.zeros(data_size, dtype=torch.uint8, device=self.device)

            # 广播实际数据（每个rank接收自己的部分）
            dist.broadcast(episodes_tensor, src=0)

            # 反序列化数据
            serialized_data = episodes_tensor.cpu().numpy().tobytes()
            episodes = pickle.loads(serialized_data)

            # 不同的rank分割episodes
            sample_batch_size = len(episodes)
            sample_questions_per_batch = sample_batch_size // self.num_answers_per_question
            num_questions_per_rank = sample_questions_per_batch // self.world_size
            num_data_per_rank = num_questions_per_rank * self.num_answers_per_question

            episodes_per_rank = episodes[self.rank*num_data_per_rank : self.rank*num_data_per_rank+num_data_per_rank]

            return episodes_per_rank
        else:
            # 单进程模式，直接返回
            return episodes

    def train_step(self, episodes):
        """执行一个训练步骤"""
        prefix_len = len(episodes[0].whole_token_ids) - len(episodes[0].generated_token_ids)

        # 计算新策略的概率分布
        batch_token_ids = torch.tensor([episode.whole_token_ids for episode in episodes], dtype=torch.long, device=self.device)
        attention_mask = (batch_token_ids != self.tokenizer.pad_token_id).long()

        new_policy_log_probs = get_batch_log_probs(
            model=self.model_engine,
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

        if self.use_gspo:
            loss = gspo_loss(
                ref_policy_log_probs=ref_policy_log_probs,
                old_policy_log_probs=old_policy_log_probs,
                new_policy_log_probs=new_policy_log_probs,
                attention_mask=attention_mask,
                advantages=advantages,
                prefix_len=prefix_len
            )
        else:
            loss = grpo_loss(
                ref_policy_log_probs=ref_policy_log_probs,
                old_policy_log_probs=old_policy_log_probs,
                new_policy_log_probs=new_policy_log_probs,
                attention_mask=attention_mask,
                advantages=advantages,
                prefix_len=prefix_len
            )

        # 反向传播和优化步骤
        self.model_engine.backward(loss)
        self.model_engine.step()

        return loss

    def evaluate(self):
        with torch.no_grad():
            self.model_engine.module.eval() # 模型调整为评估模式
            # 加载评估数据集
            test_dataset = Gsm8kTasksDataset(
                data_path=self.data_path,
                tokenizer=self.tokenizer,
                split="test",
                test_size=self.test_size
            )
            generator = torch.Generator(device="cpu")
            test_dataloader = DataLoader(
                test_dataset,
                shuffle=True,
                collate_fn=Gsm8kTasksDataset.collate_fn,
                generator=generator,
                batch_size=self.test_batch_size,
            )

            # 最新模型参数进行采样评估
            success_num = 0
            format_success_num = 0
            answer_success_num = 0
            entropy_sum = 0.0
            for batch in test_dataloader:
                episodes = sample_trajectory(
                    model=self.model_engine.module,
                    batch=batch,
                    tokenizer=self.tokenizer,
                    max_gen_len=self.max_gen_len,
                    num_answer_per_question=1,
                    reward_function=reward_function,
                    device=self.device,
                    dtype=self.dtype
                )
                # 评估准确率
                for episode in episodes:
                    if np.abs(episode.reward_info["format_reward"] - 1.25) < 1e-3:
                        format_success_num = format_success_num + 1
                    if np.abs(episode.reward_info["answer_reward"] - 1.0) < 1e-3:
                        answer_success_num = answer_success_num + 1
                    if np.abs(episode.reward - 2.25) < 1e-3:
                        success_num = success_num + 1

                # 评估entropy
                batch_token_ids = torch.tensor([episode.whole_token_ids for episode in episodes], dtype=torch.long, device=self.device)
                attention_mask = (batch_token_ids != self.tokenizer.pad_token_id).long()
                with torch.no_grad():
                    batch_logits = self.model_engine(input_ids=batch_token_ids, attention_mask=attention_mask).logits
                batch_logits = batch_logits[:, :-1, :]       # 去掉最后一个logits, 因为最后一个用于预测下一个token
                batch_probs = torch.softmax(batch_logits, dim=-1)
                batch_log_probs = torch.log(batch_probs + 1e-12)
                batch_token_entropy = -torch.sum(batch_probs * batch_log_probs, dim=-1) # batch_size * seq_len
                batch_entropy = batch_token_entropy.mean(dim=-1)
                entropy_sum += batch_entropy.sum().item()


            success_rate = success_num / self.test_size
            format_success_rate = format_success_num / self.test_size
            answer_success_rate = answer_success_num / self.test_size
            entropy = entropy_sum / self.test_size
            self.model_engine.module.train() # 模型调整回训练模式
        return success_rate, format_success_rate, answer_success_rate, entropy

    def run(self):
        """主运行循环"""
        print(f"Rank {self.rank} 开始训练循环...")
        train_step = 0

        # 评估原始模型的准确率
        if self.is_main_process and train_step % self.eval_interval == 0:
            eval_start_time = time.time()
            accuracy, format_accuracy, answer_accuracy, entropy = self.evaluate()
            print(f"第 {train_step} 步训练后评估模型性能, 格式准确率: {format_accuracy}, 回答准确率: {answer_accuracy}, 平均熵: {entropy}, 评估时间: {time.time() - eval_start_time:.2f}")
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
                            sample_batch_size = len(episodes)
                            assert sample_batch_size % self.num_answers_per_question == 0 # 检查sample_batch_size是否能被num_answers_per_question整除
                            sample_questions_per_batch = sample_batch_size // self.num_answers_per_question
                            assert sample_questions_per_batch % self.world_size == 0 # 检查sample_questions_per_batch是否能被world_size整除
                        else:
                            # 没有数据，继续循环
                            time.sleep(0.01)
                            continue

                    # 广播数据到所有rank
                    if self.world_size > 1:
                        episodes = self.broadcast_episodes(episodes)

                    # 如果所有rank都没有数据，继续循环
                    if episodes is None or len(episodes) == 0:
                        time.sleep(0.01)
                        continue

                    rewards = [episode.reward for episode in episodes]
                    print(f"Rank {self.rank} 开始训练步骤{train_step}，数据批次大小: {len(episodes)}, 奖励为: {rewards}")

                    loss = self.train_step(episodes)
                    train_step += 1

                    # 定期同步模型参数（只有主进程需要同步到采样进程）
                    if self.is_main_process and train_step % self.sync_interval == 0:
                        if self.use_lora:
                            self.model_engine.save_pretrained(self.lora_adapter_dir)
                            print(f"第 {train_step} 步训练后保存LoRA模型参数至 {self.lora_adapter_dir}")
                        else:
                            output_file = self.ckpt_dir / self.ckpt_file
                            torch.save(self.model_engine.module.state_dict(), output_file)
                            print(f"第 {train_step} 步训练后保存全量模型参数至 {output_file}")

                    # 定期评估模型性能(只在主进程进行评估)
                    if self.is_main_process and train_step % self.eval_interval == 0:
                        eval_start_time = time.time()
                        accuracy, format_accuracy, answer_accuracy, entropy = self.evaluate()
                        print(f"第 {train_step} 步训练后评估模型性能, 格式准确率: {format_accuracy}, 回答准确率: {answer_accuracy}, 平均熵: {entropy}, 评估时间: {time.time() - eval_start_time:.2f}")

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
    print(f"DeepSpeed: 启用")
    print(f"当前进程: Rank {worker.rank}/{worker.world_size-1}, 主进程: {worker.is_main_process}")

    print("初始化成功")
    worker.run()


if __name__ == "__main__":
    main()