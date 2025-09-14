from torch.utils.data import Dataset
from transformers import AutoTokenizer
#from tokenizers import Tokenizer
import numpy as np
import pandas as pd
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import threading
from collections import deque
import torch

SYSTEM_PROMPT = """You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\
The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."""

class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
    
    def add(self, data):
        with self.lock:
            self.buffer.append(data)
    
    def sample(self, batch_size=32):
        with self.lock:
            if len(self.buffer) < batch_size:
                return None
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)

@dataclass
class Episode:
    """Store all relevant information of an episode."""

    prefix: str
    prefix_tokens: List[str]
    prefix_token_ids: List[int]
    generated_token_ids: List[int]
    whole_token_ids: list[int]
    is_finished: bool
    text: str
    reward: float
    reward_info: Dict[str, float]

    old_policy_log_probs: np.ndarray
    ref_policy_log_probs: np.ndarray

@dataclass
class MiniBatch:
    """Batch of data for each training step."""

    prefix: List[str]
    prefix_tokens: List[List[str]]
    prefix_token_ids: List[List[int]]
    question: list[str]
    answer: list[str]

class Gsm8kTasksDataset(Dataset):
    """Prepare GSM8K Tasks for training"""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        data_path: str,
        split: str = "train",
        test_size: int = 100,
    ):
        data = pd.read_parquet(Path(data_path) / "main")
        # use the last `test_size` examples for testing
        self.data = (
            data.iloc[:-test_size] if split == "train" else data.iloc[-test_size:]
        )
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx].to_dict()
        item.update(self.encode_prefix(item["question"]))
        return item

    def encode_prefix(self, question: str):
        """Prefix is the *actual* input to the model."""
        prefix = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}
            ], tokenize=False, add_generation_prompt=False
        )
        tokens = self.tokenizer.tokenize(prefix)
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return {
            "prefix": prefix,
            "prefix_tokens": tokens,
            "prefix_token_ids": tokens_ids
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> MiniBatch:
        """Collate examples into a batch."""
        question = [item["question"] for item in batch]
        answer = [item["answer"].split('####')[-1].strip() for item in batch]
        prefix = [item["prefix"] for item in batch]
        prefix_tokens = [item["prefix_tokens"] for item in batch]
        prefix_token_ids = [item["prefix_token_ids"] for item in batch]
        return MiniBatch(
            question=question,
            answer=answer,
            prefix=prefix,
            prefix_tokens=prefix_tokens,
            prefix_token_ids=prefix_token_ids,
        )