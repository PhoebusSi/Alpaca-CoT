import argparse
import torch
import sys
import os
import platform
from utils.tools import *
from transformers import AutoModelForCausalLM
import peft

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser(description='Process some llm info.')
parser.add_argument('--model_type', type=str, default="chatglm", choices=AVAILABLE_MODEL,
                    help='the base structure (not the model) used for model or fine-tuned model')
parser.add_argument('--size', type=str, default="7b",
                    help='the type for base model or the absolute path for fine-tuned model')
parser.add_argument('--lora_dir', type=str, default="none",
                    help='the path for fine-tuned lora params, none when not in use')
parser.add_argument('--merged_dir', type=str, default="saved_models/tmp")
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed serving')
args = parser.parse_args()

# load model & tokenizer
model, lora_model, lora_type, model_class = get_lora_model(args)

# merge weights
if lora_type == 'q_v_proj':
    first_weight = model.model.layers[0].self_attn.q_proj.weight
    first_weight_old = first_weight.clone()
    lora_weight = lora_model.base_model.model.model.layers[0].self_attn.q_proj.weight
    assert torch.allclose(first_weight_old, first_weight)
    if peft.__version__ > '0.2.0':
        lora_model = lora_model.merge_and_unload()
    else:
        for layer in lora_model.base_model.model.model.layers:
            layer.self_attn.q_proj.merge_weights = True
            layer.self_attn.v_proj.merge_weights = True
    lora_model.train(False)
    assert not torch.allclose(first_weight_old, first_weight)
    lora_model_sd = lora_model.state_dict()
    # print(lora_model_sd)
    deloreanized_sd = {
        k.replace("base_model.model.", ""): v
        for k, v in lora_model_sd.items()
        if "lora" not in k
    }
    if args.model_type in ['llama']:
        model_class.model.save_pretrained(model, args.merged_dir, state_dict=deloreanized_sd, max_shard_size="400MB")
    else:
        model.save_pretrained(args.merged_dir)
elif lora_type == 'query_key_value':
    print("query_key_value type merge is building")
else:
    print("lora info error! check path")