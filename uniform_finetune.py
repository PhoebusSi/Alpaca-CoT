import os
import sys
import copy
import torch
import torch.nn as nn
import bitsandbytes as bnb
from dataclasses import dataclass, field
from datasets import load_dataset
import transformers
from collections import namedtuple

assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import (
    LlamaForCausalLM, LlamaTokenizer, 
    AutoModel, AutoTokenizer,
    BloomForCausalLM, BloomTokenizerFast)


from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)

import argparse

device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} 

ModelClass = namedtuple("ModelClass", ('tokenizer', 'model'))

_MODEL_CLASSES = {
    "llama": ModelClass(**{
        "tokenizer": LlamaTokenizer,
        "model": LlamaForCausalLM,
        
    }),
    "chatglm": ModelClass(**{
        "tokenizer": AutoTokenizer, #ChatGLMTokenizer,
        "model":  AutoModel, #ChatGLMForConditionalGeneration,
    }),
    "bloom": ModelClass(**{
        "tokenizer": BloomTokenizerFast,
        "model": BloomForCausalLM,
    }),
    "Auto": ModelClass(**{
        "tokenizer": AutoTokenizer,
        "model": AutoModel,
    })
}

# add the custom dataset
DATA_PATH = {
             "alpaca": "./data/alpaca_data_cleaned.json",
             "belle": "./data/belle_data_cn.json",
             "alpaca-belle": "./data/alpaca_plus_belle_data.json",
             "cot": "./data/CoT_data.json",
             "alpaca-cot": "./data/alcapa_plus_cot.json",
             "alpaca-belle-cot": "./data/alcapa_plus_belle_plus_cot.json",
             "belle1.5m": "./data/belle_data1.5M_cn.json",
             "finance": "./data/finance_en.json"
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
}

IGNORE_INDEX = -100

def generate_prompt(data_point):
    prompt_ = PROMPT_DICT['prompt_input'] if data_point["input"] else PROMPT_DICT['prompt_no_input'] 
    return prompt_.format_map(data_point) 


def get_data_model(args):

    def get_model_class(model_type):

        if model_type not in ['bloom', 'llama', 'chatglm']:
            model_type = "Auto"

        return _MODEL_CLASSES[model_type] # tokenizer, model

    data_file_path = DATA_PATH.get(args.data, None)

    assert data_file_path, "Error: Wrong type of data."

    data = load_dataset("json", data_files=data_file_path)

    print(data)

    model_class = get_model_class(args.model_type)

    if args.model_type == "chatglm":
        # chatglm can not set load_in_8bit=True: ChatGLMForConditionalGeneration does not support gradient checkpointing.
        model = model_class.model.from_pretrained(args.model_name_or_path, 
                                                trust_remote_code=True,
                                                device_map=device_map)
        tokenizer = model_class.tokenizer.from_pretrained(args.model_name_or_path,trust_remote_code=True) # default add_eos_token=False
    else:
        model = model_class.model.from_pretrained(args.model_name_or_path, 
                                                load_in_8bit=True,
                                                device_map=device_map)
    
        tokenizer = model_class.tokenizer.from_pretrained(args.model_name_or_path) # default add_eos_token=False

    # llama has no pad_id, maybe copy the stanford_alpaca's handling ?
    if args.model_type == 'llama':
        tokenizer.pad_token_id = 0 # unk_id in llama. we want this to be different from the eos token

    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    # the size of trainable parameters for lora modules
    model.print_trainable_parameters() 

    return data, model, tokenizer


def train(args):

    # 1. load data & model_class
    data, model, tokenizer = get_data_model(args)

    if "chatglm" in args.model_type:
        def prompt_tokenize(prompt):    
            input_ids = tokenizer.encode(prompt)
            return {
                "input_ids": input_ids,
                "labels": copy.deepcopy(input_ids)
            }
        def completion_tokenize(completion):         
            if completion[-4:] == '</s>':
                input_ids = tokenizer.encode(completion[:-4]) #, add_special_tokens=False)
            else:
                input_ids = tokenizer.encode(completion) #, add_special_tokens=False)
            return {
                "input_ids": input_ids,
                "labels": copy.deepcopy(input_ids)
            }
    else:
        def tokenize(prompt):
            result = tokenizer(prompt, 
                               truncation=True, 
                               max_length=args.cutoff_len, 
                            #    padding="max_length", 
                               padding=False,
                            )   
        
            return {
                "input_ids": result["input_ids"],
                "attention_mask": result["attention_mask"],
                "labels": copy.deepcopy(result["input_ids"])
            }
    

    def generate_and_tokenize_prompt(data_point):
        prompt_no_resp = generate_prompt(data_point)
        if "chatglm" in args.model_type:
            tokenized_result = prompt_tokenize(prompt_no_resp)
        else:
            tokenized_result = tokenize(prompt_no_resp)
        source_len = len(tokenized_result['input_ids'])

        prompt_with_response = prompt_no_resp + " " + data_point["output"]

        # if "llama" in args.model_type:
        prompt_with_response += " " + tokenizer.eos_token

        if "chatglm" in args.model_type:
            tokenized_with_response = completion_tokenize(prompt_with_response)
        else:
            tokenized_with_response = tokenize(prompt_with_response)

        tokenized_with_response["labels"] = [IGNORE_INDEX] * source_len + tokenized_with_response["labels"][source_len:] 

        return tokenized_with_response

    model_name = args.model_name_or_path.split( '/')[-1]
    output_dir = f"saved_models/{model_name}_{args.data}"


    # 2. split dataset
    if args.val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=args.val_set_size, shuffle=True, seed=42
        )
        train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    # 3. train
    total_batch_size = args.per_gpu_train_batch_size * args.gradient_accumulation_steps * (world_size if ddp else 1) 
    total_optim_steps = train_data.num_rows // total_batch_size

    print("***** Running training *****")
    print(f"  Num Epochs = {args.epochs}", )
    print(f"  Instantaneous batch size per GPU = {args.per_gpu_train_batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Total optimization steps = {total_optim_steps}")

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.per_gpu_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            fp16=True,
            logging_steps=20,
            evaluation_strategy="steps" if args.val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if args.val_set_size > 0 else None,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if args.val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, return_tensors="pt", padding=True),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train()

    model.save_pretrained(output_dir)

    print("\n If there's a warning about missing keys above, please disregard :)")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--size', type=str, help='the size of llama model')
    parser.add_argument('--data', type=str, help='the data used for instructing tuning')
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--model_type', default="llama", choices=['llama', 'chatglm', 'bloom'])
    parser.add_argument('--model_name_or_path', default="decapoda-research/llama-7b-hf", type=str)
    parser.add_argument('--per_gpu_train_batch_size', default=4, type=int, help='Batch size per GPU/CPU for training.')
    parser.add_argument('--gradient_accumulation_steps', default=32, type=int)
    parser.add_argument('--epochs', default=3, type=int)
    parser.add_argument('--learning_rate', default=3e-4, type=float)
    parser.add_argument('--cutoff_len', default=512, type=int)
    parser.add_argument('--lora_r', default=8, type=int)
    parser.add_argument('--lora_alpha', default=16, type=int)
    parser.add_argument('--lora_dropout', default=0.05, type=float)
    parser.add_argument('--val_set_size', default=2000, type=int)
    parser.add_argument('--lora_target_modules', nargs='+', 
                        help="the module to be injected, e.g. q_proj/v_proj/k_proj/o_proj for llama, query_key_value for bloom&GLM", 
                        default=["q_proj", "v_proj"])

    args = parser.parse_args()
    print(args)
    
    train(args)
