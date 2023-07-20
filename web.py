import datetime

from fastapi import FastAPI, Request
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
parser.add_argument('--model_type', type=str, default="chatglm", choices=AVAILABLE_MODEL,
                    help='the base structure (not the model) used for model or fine-tuned model')
parser.add_argument('--model_path', type=str, default="7b",
                    help='the type for base model or the absolute path for fine-tuned model')
parser.add_argument('--lora_dir', type=str, default="none",
                    help='the path for fine-tuned lora params, none when not in use')
parser.add_argument('--lora_r', default=8, type=int)
parser.add_argument('--lora_alpha', default=16, type=int)
parser.add_argument('--lora_dropout', default=0.05, type=float)
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed serving')
parser.add_argument('--quantization_bit', default=None, type=int, help="The number of bits to quantize the model.")
parser.add_argument('--compute_dtype', default="fp16", type=str)
args = parser.parse_args()

# GPU count
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else None
device = torch.device("cuda") if NUM_GPUS>0 else torch.device("cpu")

# load model & tokenizer
model, tokenizer = get_fine_tuned_model(args)
model = model.eval()
if torch.__version__ >= "2" and sys.platform != "win32" and sys.version_info < (3, 11):
    model = torch.compile(model)


app = FastAPI()


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


# garbage collection
def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

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

@app.post("/")
async def create_item(request: Request):
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get("prompt")
    messages = []
    messages.append({"role": "user", "content": prompt})
    response = model.chat(tokenizer, messages)

    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")

    answer = {
        "response": response,
        "status": 200,
        "time": time
    }
    log = "["+ time +"]" + '",prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)
    torch_gc()
    return answer

uvicorn.run(app, host="0.0.0.0", port=8410)