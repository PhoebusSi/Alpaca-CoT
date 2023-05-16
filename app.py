import gradio as gr
from transformers import AutoModel, AutoTokenizer
import sys
import torch
import argparse
from peft import PeftModel
import transformers
from collections import namedtuple

from transformers import (
    LlamaForCausalLM, LlamaTokenizer, 
    AutoModel, AutoTokenizer,
    BloomForCausalLM, BloomTokenizerFast, GenerationConfig)

tokenizer=None
model=None
LOAD_8BIT = False

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

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def get_model_class(model_type, 
                    model_name_or_path, 
                    lora_model_path):
    global model, tokenizer

    model_class = _MODEL_CLASSES[model_type] # tokenizer, model

    model_base = model_class.model.from_pretrained(model_name_or_path, 
                                             load_in_8bit=LOAD_8BIT,
                                             torch_dtype=torch.float16,
                                             device_map="auto")
    tokenizer = model_class.tokenizer.from_pretrained(model_name_or_path) # default add_eos_token=False

    model = PeftModel.from_pretrained(
        model_base,
        lora_model_path,
        torch_dtype=torch.float16,
        device_map={"": device}
    )
    if not LOAD_8BIT: 
        model.half()


def predict(
            instruction,
            top_p=0.9,
            temperature=1.0,
            history=None, 
            max_new_tokens=512,
            top_k=40,
            num_beams=4,
            **kwargs,
            ):
    
    history = history or []

    prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{0}\n\n### Response:"
    ).format(instruction)

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        do_sample=True,
        no_repeat_ngram_size=6,
        repetition_penalty=1.8,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)

    print('模型回复', output)

    bot_response = output.split("### Response:")[1].strip()

    history.append((instruction, bot_response))

    return "", history, history


def predict_test(message, top_p, temperature, history):

    history = history or []

    user_message = f"{message} {top_p}, {temperature}"
    print(user_message)

    history.append((message, user_message))
    return history, history

def clear_session():
    return '', '', None
 
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model_type', default="llama", choices=['llama', 'chatglm', 'bloom'])
parser.add_argument('--model_name_or_path', default="decapoda-research/llama-7b-hf", type=str)
parser.add_argument('--lora_name_or_path', default="", type=str)

args = parser.parse_args()

get_model_class(args.model_type, args.model_name_or_path, args.lora_name_or_path)

block = gr.Blocks(css = """#col_container { margin-left: auto; margin-right: auto;} #chatbot {height: 520px; overflow: auto;}""")

with block as demo:

    #top_p, temperature
    with gr.Accordion("Parameters", open=False):
        top_p = gr.Slider( minimum=-0, maximum=1.0, value=0.75, step=0.05, interactive=True, label="Top-p (nucleus sampling)",)
        temperature = gr.Slider( minimum=-0, maximum=5.0, value=1.0, step=0.1, interactive=True, label="Temperature",)

    chatbot = gr.Chatbot(label="Alpaca-CoT")
    message = gr.Textbox()
    state = gr.State()

    message.submit(predict, inputs=[message, top_p, temperature, state], outputs=[message, chatbot, state])
    
    with gr.Row():
        clear_history = gr.Button("🗑 清除历史对话 | Clear History")
        clear = gr.Button('🧹 清除输入 | Clear Input')
        send = gr.Button("🚀 发送 | Send")
        regenerate = gr.Button("🚗 重新生成 | regenerate")
    
    # regenerate.click(regenerate, inputs=[message], outputs=[chatbot])      
    regenerate.click(fn=clear_session , inputs=[], outputs=[chatbot, state], queue=False)
    send.click(predict, inputs=[message, top_p, temperature, state], outputs=[message, chatbot, state])
    clear.click(lambda: None, None, message, queue=False)
    clear_history.click(fn=clear_session , inputs=[], outputs=[message, chatbot, state], queue=False)
 
demo.queue(max_size=20, concurrency_count=20).launch(server_name="0.0.0.0", server_port=7890, debug=True, inbrowser=False, share=True)
