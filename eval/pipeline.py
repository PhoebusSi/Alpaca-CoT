import argparse
import json

import torch
from peft import (LoraConfig, PeftModel, TaskType, get_peft_model,
                  prepare_model_for_int8_training)
from tqdm import tqdm
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          BloomForCausalLM, BloomTokenizerFast,
                          LlamaForCausalLM, LlamaTokenizer, pipeline)

def eval(args):
    device_map =  {"":args.device}
    model = None
    if args.model_type in ["llama"]:
        model = LlamaForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True, load_in_8bit=False,  torch_dtype=torch.float16,device_map=device_map, local_files_only=True)
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, local_files_only=True)
        tokenizer.pad_token_id = 0
        if "vicuna" in args.model_name_or_path:
            tokenizer.padding_side = "left"
    elif args.model_type in ["bloom"]:
        model = BloomForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True, load_in_8bit=False, torch_dtype=torch.float16, device_map=device_map, local_files_only=True)
        tokenizer = BloomTokenizerFast.from_pretrained(args.model_name_or_path, trust_remote_code=True, local_files_only=True)
        tokenizer.padding_side = "left"
    elif args.model_type in ["chatglm"]:
        model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True, torch_dtype=torch.float16, device_map=device_map, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, local_files_only=True)
    elif args.model_type in ["moss"]:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True, load_in_8bit=False, torch_dtype=torch.float16, device_map=device_map, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, local_files_only=True)
        tokenizer.pad_token_id = 0
    
    if args.lora_dir != 'none':
        model = PeftModel.from_pretrained(model, args.lora_dir, device_map=device_map, )
        #model = PeftModel.from_pretrained(model, args.lora_dir)
        model = model.to(f"cuda:{args.device}")
        if "adalora" in args.lora_dir:
            model = model.to(f"cuda:{args.device}").half()

    model.eval()
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device_map=device_map)

    input_data, output_data = [], []
    if args.input == "belle":
        fname = "./raw/belle/eval_set.json"
    elif args.input == "mmcu":
        fname = "./raw/mmcu/test.json"

    with open(fname) as f:
        input_data = json.load(f)

    batch_size = args.batch_size
    for i in tqdm(range(0, len(input_data), batch_size)):
        batch = input_data[i: i + batch_size]
        batch_input = ""
        if args.input == "belle":
            batch_input = ["Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n%s\n\n### Response:"%(jtem["question"]) for jtem in batch]
        else:
            batch_input = ["Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n请阅读以下选择题并给出正确选项，不要解释原因。\n%s\n%s\n正确答案的序号是：\n\n### Response:"%(jtem["题目"], jtem["选项"]) for jtem in batch]
        with torch.no_grad():
            result = generator(batch_input, max_new_tokens=512, num_return_sequences=1, batch_size=batch_size)

        for j in range(len(result)): 
            p = result[j][0]['generated_text']
            print(p)
            if args.input == "belle":
                output_data.append({"C": batch[j]["class"], "Q": batch[j]["question"], "A": batch[j]["std_answer"], "P": p})
            else:
                output_data.append({"C": batch[j]["类别"], "Q": batch[j]["题目"], "O":batch[j]["选项"], "A": batch[j]["答案"], "P": p})

    with open("%s/%s_%s_%s_p.json"%(args.result, args.model_name_or_path.split("/")[-1].strip("."), args.lora_dir.replace("/","-").strip("."), args.input), 'w', encoding = 'utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, allow_nan=False, indent=4)  


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some llm info.')
    parser.add_argument('--model_type', default="llama", choices=['llama', 'vicuna', 'chatglm', 'bloom', 'moss'])
    parser.add_argument('--model_name_or_path', default="decapoda-research/llama-7b-hf", type=str)
    parser.add_argument('--input', type=str, default="belle", choices=['belle', 'mmcu'])
    parser.add_argument('--lora_dir', type=str, default="none")
    parser.add_argument('--result', default="eval", type=str)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--device', default=0, type=int)
    args = parser.parse_args()
    print(args)

    eval(args)
