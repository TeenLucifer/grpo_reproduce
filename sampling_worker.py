#!/usr/bin/env python3
"""
采样进程脚本：合并old_policy和ref_policy
负责采样数据并计算概率分布，通过ZeroMQ发送给训练进程
"""

import os
import time
import torch
import zmq
import yaml
import pickle
import threading
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from data_types import Gsm8kTasksDataset
from utils import sample_trajectory, reward_function, get_batch_log_probs, update_old_policy
from peft import PeftModel

class SamplingWorker:
    def __init__(self, config: dict):
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }

        self.gpu_id = config["gpu"]["sampling_gpu"]
        self.pretrained_model_path = config["model"]["pretrained_model_path"]
        self.ref_model_path = config["model"]["ref_model_path"]
        self.dtype = dtype_map.get(config["model"]["dtype"], torch.bfloat16)
        self.data_path = config["data"]["data_path"]
        self.max_gen_len = config["data"]["max_gen_len"]
        self.test_size = config["data"]["test_size"]
        self.train_batch_size = config["data"]["train_batch_size"]
        self.sample_batch_size = config["data"]["sample_batch_size"]
        self.num_answers_per_question = config["data"]["num_answers_per_question"]
        self.num_questions_per_batch = self.sample_batch_size // self.num_answers_per_question
        self.zmq_data_port = config["communication"]["data_port"]
        self.ckpt_dir = Path(config["checkpoint"]["ckpt_dir"])
        self.ckpt_file = config["checkpoint"]["ckpt_file"]
        self.use_lora = config["lora"]["enabled"]
        self.lora_adapter_dir = Path(config["lora"]["adapter_dir"])

        self.device = torch.device(f'cuda:{self.gpu_id}' if torch.cuda.is_available() else 'cpu')
        self.setup_models()
        self.setup_data_loader()
        self.setup_zmq()
        self.stop_event = threading.Event()

    def setup_gpu_device(self):
        """设置GPU设备"""
        if torch.cuda.is_available():
            torch.cuda.set_device(self.gpu_id)
            print(f"采样进程 {os.getpid()} 使用 GPU {self.gpu_id}")
        else:
            print("警告：CUDA不可用，使用CPU")

    def setup_models(self):
        """初始化模型"""
        self.setup_gpu_device()
        print(f"采样进程启动，PID: {os.getpid()}, GPU: {self.gpu_id}")

        # 初始化旧策略模型
        self.old_policy_model = AutoModelForCausalLM.from_pretrained(
            self.pretrained_model_path, 
            dtype=self.dtype, 
            _attn_implementation="sdpa"
        ).to(self.device)
        self.old_policy_model.eval()
        self.old_policy_model.requires_grad_(False)

        # 初始化参考策略模型
        self.ref_policy_model = AutoModelForCausalLM.from_pretrained(
            self.ref_model_path, 
            dtype=self.dtype, 
            _attn_implementation="sdpa"
        ).to(self.device)
        self.ref_policy_model.eval()
        self.ref_policy_model.requires_grad_(False)

        self.tokenizer = AutoTokenizer.from_pretrained(self.ref_model_path, padding_side='left')
        print("模型初始化完成")

    def setup_data_loader(self):
        """初始化数据加载器"""
        train_dataset = Gsm8kTasksDataset(
            data_path=self.data_path,
            tokenizer=self.tokenizer,
            split="train",
            test_size=self.test_size
        )
        generator = torch.Generator(device="cpu")
        self.train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=Gsm8kTasksDataset.collate_fn,
            generator=generator,
            batch_size=self.num_questions_per_batch
        )
        self.train_data_iter = iter(self.train_dataloader)
        print("数据加载器初始化完成")

    def setup_zmq(self):
        """初始化ZeroMQ通信"""
        self.context = zmq.Context()

        # 数据发送socket（PUSH模式）
        self.data_sender = self.context.socket(zmq.PUSH)
        self.data_sender.bind(f"tcp://*:{self.zmq_data_port}")
        print(f"ZeroMQ数据发送端口绑定: {self.zmq_data_port}")

    def sample_batch(self):
        """采样一批数据"""
        try:
            batch = next(self.train_data_iter)
        except StopIteration:
            # dataloader到头了重新开始
            self.train_data_iter = iter(self.train_dataloader)
            batch = next(self.train_data_iter)

        # 旧策略采样数据
        episodes = sample_trajectory(
            model=self.old_policy_model,
            batch=batch,
            tokenizer=self.tokenizer,
            max_gen_len=self.max_gen_len,
            num_answer_per_question=self.num_answers_per_question,
            reward_function=reward_function,
            device=self.device,
            dtype=self.dtype
        )

        # 计算旧策略概率分布
        batch_token_ids = torch.tensor([episode.whole_token_ids for episode in episodes], dtype=torch.long, device=self.device)
        attention_mask = (batch_token_ids != self.tokenizer.pad_token_id).long()

        # 旧策略log概率
        old_policy_log_probs = get_batch_log_probs(
            model=self.old_policy_model,
            batch_token_ids=batch_token_ids,
            attention_mask=attention_mask,
            enable_grad=False
        )

        # 参考策略log概率
        ref_policy_log_probs = get_batch_log_probs(
            model=self.ref_policy_model,
            batch_token_ids=batch_token_ids,
            attention_mask=attention_mask,
            enable_grad=False
        )

        # 更新episode数据
        for i, episode in enumerate(episodes):
            episode.old_policy_log_probs = old_policy_log_probs[i, :].to(torch.float32).cpu().numpy()
            episode.ref_policy_log_probs = ref_policy_log_probs[i, :].to(torch.float32).cpu().numpy()

        return episodes

    def serialize_episodes(self, episodes):
        """序列化episodes数据用于网络传输"""
        serialized_data = []
        for episode in episodes:
            data = {
                'prefix': episode.prefix,
                'prefix_tokens': episode.prefix_tokens,
                'prefix_token_ids': episode.prefix_token_ids,
                'generated_token_ids': episode.generated_token_ids,
                'whole_token_ids': episode.whole_token_ids,
                'is_finished': episode.is_finished,
                'text': episode.text,
                'reward': episode.reward,
                'reward_info': episode.reward_info,
                'old_policy_log_probs': episode.old_policy_log_probs,
                'ref_policy_log_probs': episode.ref_policy_log_probs
            }
            serialized_data.append(data)
        return serialized_data

    def run(self):
        """主运行循环"""
        print("开始采样循环...")
        last_sample_time = time.time()
        sample_count = 0

        try:
            while not self.stop_event.is_set():
                # 采样一批数据
                episodes = self.sample_batch()

                # 序列化数据
                serialized_episodes = self.serialize_episodes(episodes)

                # 发送数据到训练进程
                data = pickle.dumps(serialized_episodes)
                self.data_sender.send(data)

                rewards = [episode.reward for episode in episodes]
                sample_count += 1
                print(f"{time.time() - last_sample_time:.2f}s, 采样{self.sample_batch_size}条数据, {self.num_questions_per_batch}个问题, 奖励为: {rewards}")
                last_sample_time = time.time()

                if sample_count % 10 == 0:
                    if self.use_lora:
                        print(f"采样进程已采样 {sample_count} 批数据, 并尝试加载最新LoRA模型参数")
                        if not self.lora_adapter_dir:
                            print(f"最新LoRA模型参数不存在, 跳过")
                        else:
                            try:
                                if isinstance(self.old_policy_model, PeftModel):
                                    self.old_policy_model.load_adapter(self.lora_adapter_dir, adapter_name="default", is_trainable=False)
                                else:
                                    self.old_policy_model = PeftModel.from_pretrained(self.old_policy_model, self.lora_adapter_dir, is_trainable=False)
                                self.old_policy_model.eval()
                                print(f"成功更新最新LoRA模型参数: {self.lora_adapter_dir}")
                            except Exception as e:
                                print(f"加载最新LoRA模型参数失败: {self.lora_adapter_dir}, 错误: {e}")
                    else:
                        print(f"采样进程已采样 {sample_count} 批数据, 并尝试加载最新全量模型参数")
                        ckpt_path = self.ckpt_dir / self.ckpt_file
                        if not ckpt_path:
                            print(f"最新全量模型参数不存在, 跳过")
                        else:
                            try:
                                checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=True)
                                state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
                                # deepspeed保存的模型参数key有module前缀, 加载时需要移除module, 否则键不匹配
                                state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
                                self.old_policy_model.load_state_dict(state_dict)
                                print(f"成功更新最新全量模型参数: {ckpt_path}")
                            except Exception as e:
                                print(f"加载最新全量模型参数失败: {ckpt_path}, 错误: {e}")

                # 短暂休眠避免CPU占用过高
                time.sleep(0.01)

        except KeyboardInterrupt:
            print("采样进程收到中断信号")
        except Exception as e:
            print(f"采样进程错误: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """清理资源"""
        print("清理采样进程资源...")
        self.stop_event.set()
        
        if hasattr(self, 'data_sender'):
            self.data_sender.close()
        if hasattr(self, 'sync_receiver'):
            self.sync_receiver.close()
        if hasattr(self, 'context'):
            self.context.term()
        
        print("采样进程已清理完成")


def main():
    """主函数"""
    config_path = "./config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    print("=== GRPO采样进程 ===")
    print(f"GPU ID: {config["gpu"]["sampling_gpu"]}")
    print(f"数据端口: {config["communication"]["data_port"]}")

    # 创建并运行采样进程
    worker = SamplingWorker(config)
    worker.run()


if __name__ == "__main__":
    main()