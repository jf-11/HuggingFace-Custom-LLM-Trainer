"""
This file contains helper function to train the models.
Implementation based on:
- https://github.com/karpathy/llama2.c/blob/master/train.py
"""

# Imports ===============================================================================================================
import os
import json
import sys
import pickle
from tqdm import tqdm
from contextlib import nullcontext
import wandb
import math

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from bitsandbytes.optim import AdamW8bit, PagedAdamW8bit
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# PyTorch dataset class =================================================================================================
class dataset(Dataset):
    """
    Parameters:
    - data (dict): A dictionary where keys are indices and values are dictionaries with 'text' and 'label' as keys.
                   The 'text' is the instruction or input text, and the 'label' is the target output for the model.
    - tokenizer: A tokenizer instance compatible with the transformer model in use. The tokenizer should support
                 padding and be capable of returning PyTorch tensors.

    Attributes:
    - len (int): The length of the dataset, i.e., the number of items it contains.
    - data (dict): The input data passed during the instantiation of the dataset.
    - tokenizer: The tokenizer used for preprocessing the input data.
    - max_len (int): The maximum sequence length to which all the sequences will be padded or truncated.

    The `__getitem__` method of this class processes and returns a single tokenized input at a specified index,
    including the input ids, attention mask, and a modified label where the part corresponding to the instruction
    is masked with -100, ensuring that the model does not predict tokens for the instruction part.

    The `__len__` method returns the length of the dataset, allowing its size to be queried with `len(dataset)`. 
    """
    def __init__(self, data, tokenizer, max_len):
        self.len = len(data)
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"                     
        self.instruction = "You are a helpful assistant."

    def __getitem__(self, idx):
        text = self.data[idx]["text"]
        label = self.data[idx]["label"]
        combined = f"<s>[INST] {self.instruction} {text} [/INST]{label}</s>"
        #combined = f"<s><|user|>\n{self.instruction} {text}<|end|>\n<|assistant|>{label}</s>"
        tokenized = self.tokenizer(combined, return_tensors="pt", padding="max_length", max_length=self.max_len, truncation=True, add_special_tokens=False)
        label = tokenized.input_ids.clone()
        instruction_len = len(self.tokenizer(f"<s>[INST] {self.instruction} {text} [/INST]", return_tensors="pt", add_special_tokens=False)["input_ids"][0]) + 1
        #instruction_len = len(self.tokenizer(f"<s><|user|>\n{self.instruction} {text}<|end|>\n<|assistant|>", return_tensors="pt", add_special_tokens=False)["input_ids"][0]) + 1
        label[:,:instruction_len] = -100

        return {
                'ids': tokenized.input_ids[0],
                'mask': tokenized.attention_mask[0],
                'label': label[0],
        }

    def __len__(self):
        return self.len

# Parameter analysis ====================================================================================================
def model_parameter_stats(model):
    """
    Computes the number of trainable and non-trainable parameters of a given model.
    Works with models that have LoRA applied as well.

    Args:
    model (torch.nn.Module): The model to analyze.

    Returns:
    tuple: A tuple containing the number of trainable parameters, the number of non-trainable parameters,
           and the percentage of trainable parameters.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    trainable_percentage = (trainable_params / total_params) * 100 if total_params > 0 else 0

    return trainable_params, non_trainable_params, trainable_percentage

# Get learning rate function ============================================================================================
def get_lr(train_step,config,max_iters):
    """
    Calculates and returns the learning rate based on the current training step, 
    a configuration dictionary, and the maximum number of iterations. This function 
    supports both linear warmup and cosine decay learning rate schedules.

    Parameters:
    - train_step (int): The current training step.
    - config (dict): A configuration dictionary that must contain the following keys:
        - "lr" (float): The base learning rate.
        - "warmup" (int): The number of steps for the linear warmup. If warmup is not 
          used, this should be set to 0.
        - "lr_schedule" (bool): A boolean indicating whether to use a learning rate 
          schedule. If False, the learning rate will remain constant.
    - max_iters (int): The maximum number of training iterations.

    Returns:
    - float: The calculated learning rate for the current training step.

    Raises:
    - AssertionError: If the decay ratio calculated during the cosine decay schedule 
      is not between 0 and 1, inclusive.

    Notes:
    - During warmup, the learning rate increases linearly over the number of warmup steps.
    - If "lr_schedule" is True and the warmup period has passed, the function uses a 
      cosine decay schedule to adjust the learning rate.
    - If "lr_schedule" is False, the learning rate remains at the base learning rate 
      ("lr") after any warmup period.
    """
    if config["warmup"]>0:
        if train_step < config["warmup"]:
            return config["lr"] * train_step / config["warmup"]

    if config["lr_schedule"]:
        decay_ratio = (train_step - config["warmup"]) / (max_iters - config["warmup"])
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
        return 0.0 + coeff * (config["lr"] - 0.0)
            
    else:
        return config["lr"]