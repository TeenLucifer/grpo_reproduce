import multiprocessing as mp
import time
import os
import torch
import threading
import queue
from collections import deque
import numpy as np
from datasets import load_dataset
from data_types import Gsm8kTasksDataset
from torch.utils.data import DataLoader
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import group_advantages, grpo_loss, sample_trajectory, reward_function, get_batch_log_probs, update_old_policy
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


target_model_path = "./Qwen2.5-3B-Instruct"
ref_model_path = "./Qwen2.5-3B-Instruct"
pretrained_model_path = "./Qwen2.5-3B-Instruct"
data_path = "./gsm8k"
max_gen_len = 700
dtype = torch.bfloat16
BATCH_SIZE = 4
NUM_QUESTIONS_PER_BATCH = 2
NUM_ANSWERS_PER_QUESTION = BATCH_SIZE // NUM_QUESTIONS_PER_BATCH


def setup_gpu_device(gpu_id):
    """设置GPU设备"""
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        print(f"进程 {os.getpid()} 使用 GPU {gpu_id}")
    else:
        print("警告：CUDA不可用，使用CPU")


def reference_policy_worker(gpu_id, raw_data_queue, processed_data_queue, stop_event):
    """参考策略进程：计算参考策略概率分布，防止目标策略偏离过远"""
    setup_gpu_device(gpu_id)
    print(f"参考策略进程启动，PID: {os.getpid()}, GPU: {gpu_id}")

    device = torch.device(gpu_id)
    ref_policy_model = AutoModelForCausalLM.from_pretrained(ref_model_path, torch_dtype=torch.bfloat16, _attn_implementation="sdpa").to('cuda')
    ref_policy_model.eval()
    ref_policy_model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(ref_model_path, padding_side='left')

    process_count = 0
    while not stop_event.is_set():
        try:
            # 从原始数据队列获取旧策略采样数据
            episodes = raw_data_queue.get(timeout=1)

            # 计算参考策略的概率分布
            batch_token_ids = torch.tensor([episode.whole_token_ids for episode in episodes], dtype=torch.long, device=device)
            attention_mask = (batch_token_ids != tokenizer.pad_token_id).long()
            batch_log_probs = get_batch_log_probs(
                model=ref_policy_model,
                batch_token_ids=batch_token_ids,
                attention_mask=attention_mask,
                enable_grad=False  # 参考策略推理，不需要梯度
            )
            for i, episode in enumerate(episodes):
                episode.ref_policy_log_probs = batch_log_probs[i, :].clone().detach().to(device)
            for episode in episodes:
                episode.old_policy_log_probs = episode.old_policy_log_probs.clone().detach().to(device)

            # 将处理后的数据发送到训练队列
            processed_data_queue.put(episodes)
            process_count += 1

            if process_count % 50 == 0:
                print(f"参考策略已处理 {process_count} 条数据")

        except queue.Empty:
            continue
        except Exception as e:
            print(f"参考策略处理错误: {e}")


def old_policy_sampling_worker(gpu_id, raw_data_queue, stop_event, sync_queue=None):
    """旧策略采样进程：持续采样数据到经验回放池"""
    setup_gpu_device(gpu_id)
    print(f"旧策略采样进程启动，PID: {os.getpid()}, GPU: {gpu_id}")

    device = torch.device(gpu_id)
    old_policy_model = AutoModelForCausalLM.from_pretrained(pretrained_model_path, torch_dtype=torch.bfloat16, _attn_implementation="sdpa").to('cuda')
    old_policy_model.eval()
    old_policy_model.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(ref_model_path, padding_side='left')
    train_dataset = Gsm8kTasksDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        split="train",
        test_size=100
    )
    generator = torch.Generator(device="cpu")
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=Gsm8kTasksDataset.collate_fn,
        generator=generator,
        batch_size=NUM_QUESTIONS_PER_BATCH
    )
    train_data_iter = iter(train_dataloader)

    sample_count = 0
    sync_interval = 5  # 每5批数据同步一次
    
    while not stop_event.is_set():
        # 检查是否需要同步新策略参数
        if sync_queue and sample_count % sync_interval == 0 and sample_count > 0:
            try:
                # 非阻塞检查是否有新的模型参数
                new_state_dict = sync_queue.get_nowait()
                update_old_policy(old_policy_model, new_state_dict)
                print(f"第 {sample_count} 批数据采样前同步新策略参数")
            except queue.Empty:
                pass  # 没有新参数，继续采样

        try:
            batch = next(train_data_iter) # 每次只取一批
        except:
            # dataloader到头了重新开始
            train_data_iter = iter(train_dataloader)
            batch = next(train_data_iter)

        # 旧策略采样数据
        episodes = sample_trajectory(
            model=old_policy_model,
            batch=batch,
            tokenizer=tokenizer,
            max_gen_len=max_gen_len,
            num_answer_per_question=NUM_ANSWERS_PER_QUESTION,
            reward_function=reward_function,
            device=device,
            dtype=dtype
        )

        # 计算旧策略概率分布
        batch_token_ids = torch.tensor([episode.whole_token_ids for episode in episodes], dtype=torch.long, device=device)
        attention_mask = (batch_token_ids != tokenizer.pad_token_id).long()
        batch_log_probs = get_batch_log_probs(
            model=old_policy_model,
            batch_token_ids=batch_token_ids,
            attention_mask=attention_mask,
            enable_grad=False  # 旧策略推理，不需要梯度
        )
        for i, episode in enumerate(episodes):
            episode.old_policy_log_probs = batch_log_probs[i, :].clone().detach().to(device)

        # 发送到参考策略处理队列
        raw_data_queue.put(episodes)
        sample_count += 1

        if sample_count % 100 == 0:
            print(f"旧策略已采样 {sample_count} 批数据")


def new_policy_training_worker(gpu_id, processed_data_queue, stop_event, sync_queue=None):
    """新策略训练进程：从处理后的数据队列获取数据训练"""
    setup_gpu_device(gpu_id)
    print(f"新策略训练进程启动，PID: {os.getpid()}, GPU: {gpu_id}")

    device = torch.device(gpu_id)
    new_policy_model = AutoModelForCausalLM.from_pretrained(pretrained_model_path, torch_dtype=torch.bfloat16, _attn_implementation="sdpa").to('cuda')
    new_policy_model.train()  # 设置为训练模式
    new_policy_model.requires_grad_(True)

    tokenizer = AutoTokenizer.from_pretrained(ref_model_path, padding_side='left')

    # 初始化优化器和调度器
    optimizer = AdamW(new_policy_model.parameters(), lr=1e-5, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-6)

    # 梯度裁剪参数
    max_grad_norm = 1.0

    train_step = 0
    sync_interval = 25  # 每25个训练步骤同步一次（5批数据 * 5个训练步骤）

    while not stop_event.is_set():
        try:
            # 从处理后的数据队列获取数据
            episodes = processed_data_queue.get(timeout=0.1)
            prefix_len = len(episodes[0].whole_token_ids) - len(episodes[0].generated_token_ids)

            # 计算新策略的概率分布
            batch_token_ids = torch.tensor([episode.whole_token_ids for episode in episodes], dtype=torch.long, device=device)
            attention_mask = (batch_token_ids != tokenizer.pad_token_id).long()
            new_policy_log_probs = get_batch_log_probs(
                model=new_policy_model,
                batch_token_ids=batch_token_ids,
                attention_mask=attention_mask,
                enable_grad=True  # 新策略训练，需要梯度
            )

            # 计算优势函数
            rewards = torch.tensor([episode.reward for episode in episodes], dtype=dtype, device=device)
            advantages = group_advantages(rewards=rewards, num_answers_per_question=NUM_ANSWERS_PER_QUESTION).to(device)

            # 计算grpo算法的loss
            ref_policy_log_probs = [episode.ref_policy_log_probs for episode in episodes]
            old_policy_log_probs = [episode.old_policy_log_probs for episode in episodes]
            ref_policy_log_probs = torch.stack(ref_policy_log_probs).to(device)
            old_policy_log_probs = torch.stack(old_policy_log_probs).to(device)

            loss = grpo_loss(
                ref_policy_log_probs=ref_policy_log_probs,
                old_policy_log_probs=old_policy_log_probs,
                new_policy_log_probs=new_policy_log_probs,
                attention_mask=attention_mask,
                advantages=advantages,
                prefix_len=prefix_len
            )

            # 反向传播和优化步骤
            optimizer.zero_grad()  # 清除旧梯度
            loss.backward()        # 反向传播计算梯度

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(new_policy_model.parameters(), max_grad_norm)

            optimizer.step()       # 更新模型参数
            scheduler.step()       # 更新学习率

            train_step += 1

            # 定期同步新策略参数到旧策略（非阻塞）
            if sync_queue and train_step % sync_interval == 0:
                try:
                    sync_queue.put_nowait(new_policy_model.state_dict())
                    print(f"第 {train_step} 步训练后发送模型参数同步请求")
                except queue.Full:
                    print("同步队列已满，跳过此次同步")

            # 定期打印训练信息
            if train_step % 10 == 0:
                print(f"训练步骤 {train_step}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")

            # 定期保存模型检查点（每1000步）
            if train_step % 1000 == 0:
                checkpoint_path = f"./checkpoints/model_step_{train_step}.pt"
                os.makedirs("./checkpoints", exist_ok=True)
                torch.save({
                    'step': train_step,
                    'model_state_dict': new_policy_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss.item(),
                }, checkpoint_path)
                print(f"模型检查点已保存: {checkpoint_path}")

            # TODO(wangjintao): 定期evaluate准确率

        except queue.Empty:
            continue


def main():
    print(f"主程序启动，PID: {os.getpid()}")
    print("=== 三进程Off-Policy并行训练框架 ===")
    print("进程1: 旧策略采样数据（每5批同步新策略参数）")
    print("进程2: 参考策略计算概率分布")
    print("进程3: 新策略训练（使用参考策略约束）")

    # 设置GPU设备ID
    OLD_POLICY_GPU = 0     # 卡1用于旧策略采样
    REFERENCE_GPU = 1      # 卡2用于参考策略
    NEW_POLICY_GPU = 2     # 卡3用于新策略训练

    # 检查GPU可用性
    if torch.cuda.device_count() < 3:
        print(f"警告：可用GPU数量少于3个（当前{torch.cuda.device_count()}个）")
        if torch.cuda.device_count() == 2:
            print("使用2个GPU：旧策略+参考策略共用GPU0，新策略使用GPU1")
            OLD_POLICY_GPU = 0
            REFERENCE_GPU = 0
            NEW_POLICY_GPU = 1
        elif torch.cuda.device_count() == 1:
            print("使用1个GPU：所有进程共用GPU0")
            OLD_POLICY_GPU = REFERENCE_GPU = NEW_POLICY_GPU = 0

    # 创建进程间通信队列
    raw_data_queue = mp.Queue(maxsize=1000)        # 旧策略 -> 参考策略
    processed_data_queue = mp.Queue(maxsize=1000)  # 参考策略 -> 新策略训练
    sync_queue = mp.Queue(maxsize=5)               # 新策略 -> 旧策略参数同步

    # 创建停止事件
    stop_event = mp.Event()

    # 创建进程
    processes = []

    # 创建参考策略进程
    reference_process = mp.Process(
        target=reference_policy_worker,
        args=(REFERENCE_GPU, raw_data_queue, processed_data_queue, stop_event),
        name="Reference_Policy"
    )
    processes.append(reference_process)

    # 创建旧策略采样进程
    old_policy_process = mp.Process(
        target=old_policy_sampling_worker,
        args=(OLD_POLICY_GPU, raw_data_queue, stop_event, sync_queue),
        name="OldPolicy_Sampling"
    )
    processes.append(old_policy_process)

    # 创建新策略训练进程
    new_policy_process = mp.Process(
        target=new_policy_training_worker,
        args=(NEW_POLICY_GPU, processed_data_queue, stop_event, sync_queue),
        name="NewPolicy_Training"
    )
    processes.append(new_policy_process)

    # 启动所有进程
    for process in processes:
        process.start()
        print(f"进程 {process.name} 已启动")

    try:
        # 等待所有进程结束
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        print("\n接收到中断信号，正在关闭所有进程...")
        stop_event.set()  # 通知所有进程停止

        for process in processes:
            if process.is_alive():
                process.join(timeout=5)  # 等待进程优雅退出
                if process.is_alive():
                    process.terminate()
                    process.join()
        print("所有进程已关闭")


if __name__ == "__main__":
    main()