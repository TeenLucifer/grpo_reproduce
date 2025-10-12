from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from data_types import Episode, MiniBatch
from typing import Callable, List
import re
from typing import Any, Dict, List, Optional
import time
import numpy as np
from fractions import Fraction

def group_advantages(rewards: torch.Tensor, num_answers_per_question: int):
    batch_size = rewards.shape[0]
    assert batch_size % num_answers_per_question == 0, "batch_size must be divisible by num_answers_per_question"

    num_questions = batch_size // num_answers_per_question

    rewards_grouped = rewards.view(num_questions, num_answers_per_question)

    mean = rewards_grouped.mean(dim=1, keepdim=True)
    std = rewards_grouped.std(dim=1, keepdim=True)
    std = std + 1e-8 # 防止除0

    advantages_grouped = (rewards_grouped - mean) / std
    advantages = advantages_grouped.view(batch_size, 1)

    return advantages

def gspo_loss(
    ref_policy_log_probs: torch.Tensor,
    old_policy_log_probs: torch.Tensor,
    new_policy_log_probs: torch.Tensor,
    attention_mask: torch.Tensor,
    advantages: torch.Tensor,
    prefix_len: int,
    clip_epsilon: float=0.2,
    kl_beta: float=0.04
):
    batch_size = ref_policy_log_probs.shape[0]

    # 取生成部分的概率分布
    ref_policy_log_probs_ = ref_policy_log_probs[:, prefix_len-1:] # token_0裁剪了, 因此需要裁剪的长度为prefix_len-1
    old_policy_log_probs_ = old_policy_log_probs[:, prefix_len-1:]
    new_policy_log_probs_ = new_policy_log_probs[:, prefix_len-1:]
    attention_mask_       = attention_mask[:, prefix_len:]         # attention_mask维度中token_0的位置没裁剪, 因此需要裁剪的长度为prefix_len

    # 计算有效序列, 遮掩pad_token
    valid_seq_len = attention_mask_.sum(dim=1)
    new_old_log_probs_ = (new_policy_log_probs_ - old_policy_log_probs_) * attention_mask_
    ref_new_log_probs_ = (ref_policy_log_probs_ - new_policy_log_probs_) * attention_mask_

    # 序列级别的重要性采样
    importance_ratio = torch.exp(new_old_log_probs_.sum(dim=1) / valid_seq_len).view(batch_size, 1) # batch_size * 1
    cliped_ratio = torch.clip(importance_ratio, 1 - clip_epsilon, 1 + clip_epsilon) # batch_size * 1
    importance_term = importance_ratio * advantages # batch_size * 1
    clip_term = cliped_ratio * advantages # batch_size * 1

    kl_term = torch.exp(ref_new_log_probs_.sum(dim=1) / valid_seq_len) - (ref_new_log_probs_.sum(dim=1) / valid_seq_len) - 1
    kl_term = kl_term.view(batch_size, 1)

    objective_function = torch.min(importance_term, clip_term) - kl_beta * kl_term
    sequence_loss = -objective_function

    # 批次平均损失作为总损失
    loss = sequence_loss.mean()
    return loss

def grpo_loss(
    ref_policy_log_probs: torch.Tensor,
    old_policy_log_probs: torch.Tensor,
    new_policy_log_probs: torch.Tensor,
    attention_mask: torch.Tensor,
    advantages: torch.Tensor,
    prefix_len: int,
    clip_epsilon: float=0.2,
    kl_beta: float=0.04
):
    # 取生成部分的概率分布
    ref_policy_log_probs_ = ref_policy_log_probs[:, prefix_len-1:] # token_0裁剪了, 因此需要裁剪的长度为prefix_len-1
    old_policy_log_probs_ = old_policy_log_probs[:, prefix_len-1:]
    new_policy_log_probs_ = new_policy_log_probs[:, prefix_len-1:]
    attention_mask_       = attention_mask[:, prefix_len:]         # attention_mask维度中token_0的位置没裁剪, 因此需要裁剪的长度为prefix_len

    importance_ratio = torch.exp(new_policy_log_probs_ - old_policy_log_probs_)
    cliped_ratio = torch.clip(importance_ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    importance_term = importance_ratio * advantages
    clip_term = cliped_ratio * advantages

    kl_term = torch.exp(ref_policy_log_probs_ - new_policy_log_probs_) - (ref_policy_log_probs_ - new_policy_log_probs_) - 1

    objective_function = torch.min(importance_term, clip_term) - kl_beta * kl_term
    per_token_loss = -objective_function

    loss = ((per_token_loss * attention_mask_).sum(dim=1) / attention_mask_.sum(dim=1)).mean()
    return loss

def get_batch_log_probs(
    model: AutoModelForCausalLM,
    batch_token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    enable_grad: bool = False
):
    '''
    获取每个token在词表中被选择的概率, 需要获取生成token_{n+1}的概率分布, 即获取prob_{n}

    batch_token_ids:       batch_size * seq_len
    batch_logits:          batch_size * (seq_len - 1) * vocab_size
    batch_token_ids_:      batch_size * (seq_len - 1) * vocab_size
    batch_token_log_probs: batch_size * (seq_len - 1)

    token序列和logit序列分别为:
    [token_0               token_1               token_2,               ...,                   token_n]
    [           logit_0               logit_1               logit_2,    ...,    logit_{n-1},               logit_{n}]

    logit序列和token序列一一对应, 根据logit_{n}分布得到token_{n+1}
    [token_0 -> logit_0 -> token_1 -> logit_1 -> token_2, ..., logit_{n-1} -> token_n]
    [logit_{n} token_{n+1}]
    
    Args:
        enable_grad: 是否启用梯度计算，训练新策略时需要True
    '''
    if enable_grad:
        # 训练模式，需要梯度
        batch_logits = model(input_ids=batch_token_ids, attention_mask=attention_mask).logits
    else:
        # 推理模式，不需要梯度
        with torch.no_grad():
            batch_logits = model(input_ids=batch_token_ids, attention_mask=attention_mask).logits

    batch_logits = batch_logits[:, :-1, :]       # 去掉最后一个logits, 因为最后一个用于预测下一个token
    batch_token_ids_ = batch_token_ids[:, 1:]     # 去掉第一个token, 因为没有对应的logits用于预测它

    batch_log_probs = []
    for logits_row, token_ids_row in zip(batch_logits, batch_token_ids_):
        log_probs_row = logits_row.log_softmax(dim=-1) # 把logits归一化转换为概率分布, log概率分布能把重要性采样/KL散度计算的除法运算转换为减法运算
        token_log_probs_row = torch.gather(log_probs_row, dim=1, index=token_ids_row.unsqueeze(1)).squeeze(1) # 根据token_id检索对应token的概率值, 用于重要性采样/KL散度计算
        batch_log_probs.append(token_log_probs_row)
    batch_log_probs = torch.stack(batch_log_probs)
    return batch_log_probs

@torch.no_grad()
def sample_trajectory(
    model: AutoModelForCausalLM,
    batch: MiniBatch,
    tokenizer: AutoTokenizer,
    max_gen_len: int,
    num_answer_per_question: int,
    reward_function: Callable,
    device: torch.device,
    dtype: torch.dtype
) -> List[Episode]:
    pad_token_id = tokenizer.pad_token_id
    min_prompt_len = min(len(t) for t in batch.prefix_token_ids)
    max_prompt_len = max(len(t) for t in batch.prefix_token_ids)
    total_len = max_gen_len + max_prompt_len
    #batch_size = len(batch.prefix_token_ids)
    num_question_per_batch = len(batch.prefix_token_ids)
    batch_size = num_question_per_batch * num_answer_per_question

    # 对齐prefix的长度, 并转换为tensor (使用左padding)
    batch_prefix_token_ids = torch.full((batch_size, max_prompt_len), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, max_prompt_len), dtype=torch.long, device=device)
    for i, seq in enumerate(batch.prefix_token_ids):
        for j in range(num_answer_per_question):
            # 左padding：将序列放在右边，左边填充pad_token_id
            start_idx = max_prompt_len - len(seq)
            batch_prefix_token_ids[i * num_answer_per_question + j, start_idx:] = torch.tensor(seq, dtype=torch.long, device=device)
            attention_mask[i * num_answer_per_question + j, start_idx:] = 1

    # 根据prefix生成结果
    batch_token_ids = model.generate(
        input_ids=batch_prefix_token_ids,
        attention_mask=attention_mask,
        pad_token_id=pad_token_id,
        max_new_tokens=max_gen_len
    )
    # 获取生成的文本
    batch_texts = [tokenizer.decode(token_ids, skip_special_tokens=True) for token_ids in batch_token_ids]
    batch_response_texts = [tokenizer.decode(token_ids[max_prompt_len:], skip_special_tokens=True) for token_ids in batch_token_ids]

    episodes = []
    for i in range(batch_size):
        batch_idx = i // num_answer_per_question
        response_texts=batch_response_texts[i]
        rewards = reward_function(
            response=response_texts,
            answer=batch.answer[batch_idx],
        )
        episode = Episode(
            prefix=batch.prefix[batch_idx],
            prefix_tokens=batch.prefix_tokens[batch_idx],
            prefix_token_ids=batch.prefix_token_ids[batch_idx],
            generated_token_ids=batch_token_ids[i, max_prompt_len:].tolist(),
            whole_token_ids=batch_token_ids[i, :].tolist(),
            is_finished=True,
            text=batch_texts[i],
            reward=rewards["reward"],
            reward_info=rewards["reward_info"],
            old_policy_log_probs=torch.zeros(batch_token_ids.shape[1], dtype=dtype, device=device),
            ref_policy_log_probs=torch.zeros(batch_token_ids.shape[1], dtype=dtype, device=device)
        )
        episodes.append(episode)

    return episodes

def format_reward_function(response: str) -> float:
    """
    Checks if the response follows the format <think>...</think><answer>...</answer>
    """
    think_regex = r"<think>.*?<\/think>"
    answer_regex = r"<answer>.*?<\/answer>"
    #full_format_regex = r"^<think>.*?<\/think>\n<answer>.*?<\/answer>$"

    think_match = re.search(think_regex, response, re.DOTALL)
    answer_match = re.search(answer_regex, response, re.DOTALL)
    reward = -1.0
    if think_match and answer_match: # think和answer格式都匹配上奖励1.25, 否则-1
        reward = 1.25

    return reward

def smart_float(s: str) -> float:
    s = s.strip()
    # 去掉千分位逗号
    s = s.replace(",", "")
    try:
        # 尝试分数
        return float(Fraction(s))
    except ValueError:
        # 普通数字
        return float(s)

def answer_reward_function(response: str, answer: str = None) -> float:
    """
    Checks if the answer uses all numbers exactly once and evaluates to the target
    """
    answer_regex = r"<answer>(.*?)<\/answer>"
    answer_match = re.search(answer_regex, response, re.DOTALL)
    if not answer_match:
        return 0.0

    answer_content = answer_match.group(1)
    if not answer_content:
        return 0.0

    num_regex = r'\d+\.\d+|\d+/\d+|\d+'
    nums = re.findall(num_regex, answer_content) 
    if len(nums) == 0:
        return -1.0 # 答案中不包含数字, 奖励-1
    num = nums[-1]

    #answer_num = float(answer.replace(',', ''))
    truth_num = smart_float(answer)
    reply_num = smart_float(num)
    if abs(reply_num - truth_num) < 1e-5:
        return 1.0

    return -1.0 # 答案中数字不匹配, 奖励-1


def reward_function(
    response: str,
    answer: str = None,
) -> Dict[str, Any]:
    """Reward function for Countdown Tasks.

    Total reward = 0.1 * format_reward + answer_reward
    """
    format_reward = format_reward_function(response=response)
    answer_reward = answer_reward_function(response=response, answer=answer)
    return {
        "reward": format_reward + answer_reward,
        "reward_info": {
            "format_reward": format_reward,
            "answer_reward": answer_reward,
        },
    }


def update_old_policy(old_policy_model, new_policy_state_dict):
    """将新策略模型的参数同步到旧策略模型"""
    old_policy_model.load_state_dict(new_policy_state_dict)
    print(f"旧策略模型已更新，时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")


def train_accuracy(episodes: list[Episode]):
    batch_size = len(episodes)

    format_correct_num = 0
    answer_correct_num = 0
    for episode in episodes:
        if np.abs(1.25 - episode.reward_info["format_reward"]) < 1e-3:
            format_correct_num = format_correct_num + 1
        if np.abs(1.0 - episode.reward_info["answer_reward"]) < 1e-3:
            answer_correct_num = answer_correct_num + 1
    format_accuracy = float(format_correct_num) / batch_size
    answer_accuracy = float(answer_correct_num) / batch_size

    return format_accuracy, answer_accuracy