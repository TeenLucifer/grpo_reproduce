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
from collections import deque
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from data_types import Gsm8kTasksDataset, Episode
from utils import sample_trajectory, reward_function, get_batch_log_probs, update_old_policy

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
        self.train_batch_size = config["training"]["batch_size"]
        self.batch_size = config["sampling"]["batch_size"]
        self.num_answers_per_question = config["sampling"]["num_answers_per_question"]
        self.num_questions_per_batch = self.batch_size // self.num_answers_per_question
        #self.num_questions_per_batch = config["training"]["num_questions_per_batch"]
        #self.num_answers_per_question = self.batch_size // self.num_questions_per_batch
        self.zmq_data_port = config["communication"]["data_port"]
        self.zmq_sync_port = config["communication"]["sync_port"]

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
            test_size=100
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

        # 模型参数接收socket（PULL模式）
        self.sync_receiver = self.context.socket(zmq.PULL)
        self.sync_receiver.bind(f"tcp://*:{self.zmq_sync_port}")
        print(f"ZeroMQ同步接收端口绑定: {self.zmq_sync_port}")

        # 启动同步监听线程
        self.sync_thread = threading.Thread(target=self.sync_listener)
        self.sync_thread.daemon = True
        self.sync_thread.start()

    def sync_listener(self):
        """监听模型参数同步的线程"""
        print("模型参数同步监听线程启动")
        sync_interval = 5  # 每5批数据同步一次
        batch_count = 0

        while not self.stop_event.is_set():
            try:
                # 非阻塞接收模型参数
                if self.sync_receiver.poll(100):  # 100ms超时
                    print("接收模型参数同步请求...")
                    state_dict_data = self.sync_receiver.recv()
                    state_dict = pickle.loads(state_dict_data)

                    # 更新旧策略模型参数
                    update_old_policy(self.old_policy_model, state_dict)
                    print(f"第 {batch_count} 批数据采样前同步新策略参数")

            except Exception as e:
                print(f"同步监听错误: {e}")

            time.sleep(0.1)

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
        sample_start_time = time.time()
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

                sample_count += 1
                print(f"{time.time() - sample_start_time} 采样{self.batch_size}条数据, {self.num_questions_per_batch}个问题")

                if sample_count % 10 == 0:
                    print(f"采样进程已采样 {sample_count} 批数据")

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
    print(f"同步端口: {config["communication"]["sync_port"]}")

    # 创建并运行采样进程
    worker = SamplingWorker(config)
    worker.run()


if __name__ == "__main__":
    main()