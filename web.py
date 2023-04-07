from fastapi import FastAPI
import uvicorn
import time
import json
import argparse
import torch
import sys
import os
import platform
from utils.tools import *
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


parser = argparse.ArgumentParser(description='Process some llm info.')
parser.add_argument('--llm', type=str, default="chatglm", choices=AVAILABLE_MODEL,
                    help='the base structure (not the model) used for model or fine-tuned model')
parser.add_argument('--model_path', type=str, default="7b",
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
model = model.eval()
if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)


app = FastAPI()


def server(instruction):
    # 1. generate input
    prompt = generate_service_prompt(instruction, args.llm, args.lora_dir)
    # 2. encoder
    generation_config = get_generation_config(args.llm)
    inputs_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(DEVICE_TYPE)
    # 3. generate & decoder
    outputs = model.generate(
        input_ids=inputs_ids,
        generation_config=generation_config
    )
    res = tokenizer.decode(outputs[0], skip_special_tokens=True)
    output = generate_service_output(res, prompt, args.llm, args.lora_dir)
    return output


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/query/{query}")
def read_item(query: str):
    begin_time = time.time()
    answer = server(query)
    end_time = time.time()
    res = {"query": query, "answer": answer, "time": end_time - begin_time}
    print(json.dumps(res, ensure_ascii=False))
    return res


uvicorn.run(app, host="0.0.0.0", port=8410)