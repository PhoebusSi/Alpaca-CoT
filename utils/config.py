import os
from collections import namedtuple
import torch

from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    BloomForCausalLM,
    BloomTokenizerFast)

AVAILABLE_MODEL = ['bloom', 'llama', 'chatglm']
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
DEVICE_MAP = {"": int(os.environ.get("LOCAL_RANK") or 0)} if WORLD_SIZE != 1 else "auto"
DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"

ModelClass = namedtuple("ModelClass", ('tokenizer', 'model'))

MODEL_CLASSES = {
    "llama": ModelClass(**{
        "tokenizer": LlamaTokenizer,
        "model": LlamaForCausalLM,
    }),
    "bloom": ModelClass(**{
        "tokenizer": BloomTokenizerFast,
        "model": BloomForCausalLM,
    }),
    "chatglm": ModelClass(**{
        "tokenizer": AutoTokenizer,
        "model": AutoModel,
    }),
    "Auto": ModelClass(**{
        "tokenizer": AutoTokenizer,
        "model": AutoModel,
    })
}

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
    "prompt_format_before": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
    ),
    "prompt_format_after": (
        "\n\n### Response:"
    )
}

IGNORE_INDEX = -100

COMMON_PATH = ""  # local path for model

MODEL_LORA_TARGET_MODULES = {
    "bloom": ["query_key_value"],
    "llama": ["q_proj", "v_proj"],
    "chatglm": ["query_key_value"],
}

MODEL_PATHS = {
    "llama_7b": "decapoda-research/llama-7b-hf",
    "llama_13b": "decapoda-research/llama-13b-hf",
    "chatglm_6b": "THUDM/chatglm-6b",
    "bloom_7b": "bigscience/bloomz-7b1-mt",
}


GENERATE_CONFIG = {
    "temperature": 0.1,
    "top_p": 0.75,
    "top_k": 40,
    "num_beams": 4,
    "max_new_tokens": 512
}

GENERATE_CONFIG_4_firefly = {
    "temperature": 0.35,
    "top_p": 0.85,
    "do_sample": True,
    "repetition_penalty": 1.2,
    "max_new_tokens": 200
}
