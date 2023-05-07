import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse
import torch
import sys
import platform
from utils.tools import *

parser = argparse.ArgumentParser(description='Process some llm info.')
parser.add_argument('--model_type', type=str, default="chatglm", choices=AVAILABLE_MODEL,
                    help='the base structure (not the model) used for model or fine-tuned model')
parser.add_argument('--size', type=str, default="7b",
                    help='the type for base model or the absolute path for fine-tuned model')
parser.add_argument('--lora_dir', type=str, default="none",
                    help='the path for fine-tuned lora params, none when not in use')
parser.add_argument('--lora_r', default=8, type=int)
parser.add_argument('--lora_alpha', default=16, type=int)
parser.add_argument('--lora_dropout', default=0.05, type=float)
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed serving')
args = parser.parse_args()

# load model & tokenizer
model, tokenizer = get_fine_tuned_model(args)
# model = model.eval()
if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)


if args.model_type != 'chatglm':
    def server(instruction):
        # 1. generate input
        prompt = generate_service_prompt(instruction, args.model_type, args.lora_dir)
        # 2. encoder
        generation_config = get_generation_config(args.model_type)
        inputs_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(DEVICE_TYPE)
        # 3. generate & decoder
        outputs = model.generate(
            input_ids=inputs_ids,
            generation_config=generation_config
        )
        res = tokenizer.decode(outputs[0], skip_special_tokens=True)
        output = generate_service_output(res, prompt, args.model_type, args.lora_dir)
        return output
    while 1:
        print("User:")
        instructions = input()
        print("GPT:\n"+server(instructions))
else:
    def build_prompt(history):
        prompt = "欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"
        for query, response in history:
            prompt += f"\n\n用户：{query}"
        prompt += f"\n\nChatGLM-6B：{response}"
        return prompt

    history = []
    print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
    while 1:
        query = input("\n用户：")
        if query == "stop":
            break
        if query == "clear":
            history = []
            os.system('clear')
            print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            continue
        count = 0
        for response, history in model.stream_chat(tokenizer, query, history=history):
            count += 1
            if count % 8 == 0:
                os.system('clear')
                print(build_prompt(history), flush=True)
        os.system('clear')
        print(build_prompt(history), flush=True)