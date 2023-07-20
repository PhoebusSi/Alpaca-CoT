import copy
import logging
from typing import List, Optional

import torch
from datasets import load_dataset
from transformers import GenerationConfig, AutoConfig, BitsAndBytesConfig, PreTrainedModel

from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType
)
from .config import *
from .device import get_device_map
from transformers.utils.versions import require_version

logging = logging.getLogger(__name__)


def generate_prompt(data_point):
    prompt_ = PROMPT_DICT['prompt_input'] if data_point["input"] else PROMPT_DICT['prompt_no_input']
    return prompt_.format_map(data_point)


def generate_prompt_dict(data_point):
    prompt_ = PROMPT_DICT['prompt_input'] if data_point["input"] else PROMPT_DICT['prompt_no_input']
    result_ = data_point["output"]
    return {
        "origin": data_point,
        "input": prompt_.format_map(data_point),
        "output": result_
    }


def generate_prompt_4_chatglm(data_point):
    prompt_ = PROMPT_DICT['chatglm_prompt_input']
    return prompt_.format_map(data_point)


def generate_prompt_4_chatglm_dict(data_point):
    prompt_ = PROMPT_DICT['chatglm_prompt_input']
    result_ = data_point["output"]
    return {
        "origin": data_point,
        "input": prompt_.format_map(data_point),
        "output": result_
    }


def get_data_model(args):
    def _get_model_class(llm_type, model_path):
        if llm_type not in AVAILABLE_MODEL:
            llm_type = "Auto"
            return MODEL_CLASSES[llm_type], model_path
        else:
            load_path = llm_type + "_" + model_path
            return MODEL_CLASSES[llm_type], COMMON_PATH + MODEL_PATHS[load_path]

    data = load_dataset("json", data_files=args.data)
    print(data)

    model_class, model_path = _get_model_class(args.model_type, args.size)

    if args.model_type == "chatglm":
        model = model_class.model.from_pretrained(model_path,
                                                  trust_remote_code=True,
                                                  device_map=DEVICE_MAP)
        tokenizer = model_class.tokenizer.from_pretrained(model_path,
                                                          trust_remote_code=True)
    else:
        model = model_class.model.from_pretrained(model_path,
                                                  load_in_8bit=False,
                                                  device_map=DEVICE_MAP)
        tokenizer = model_class.tokenizer.from_pretrained(model_path)
        tokenizer.pad_token_id = 0
    model = prepare_model_for_int8_training(model)
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=MODEL_LORA_TARGET_MODULES[args.model_type],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return data, model, tokenizer


def get_tokenize_func(args, tokenizer):
    if "chatglm" in args.model_type:
        def prompt_tokenize(prompt):
            input_ids = tokenizer.encode(prompt)
            return {
                "input_ids": input_ids,
                "labels": copy.deepcopy(input_ids)
            }

        def completion_tokenize(completion):
            if completion[-4:] == '</s>':
                input_ids = tokenizer.encode(completion[:-4])  # , add_special_tokens=False)
            else:
                input_ids = tokenizer.encode(completion)  # , add_special_tokens=False)
            return {
                "input_ids": input_ids,
                "labels": copy.deepcopy(input_ids)
            }

        return prompt_tokenize, completion_tokenize
    else:
        def tokenize(prompt):
            result = tokenizer(prompt,
                               truncation=True,
                               max_length=args.cutoff_len,
                               padding=False,
                               )
            return {
                "input_ids": result["input_ids"],
                "attention_mask": result["attention_mask"],
                "labels": copy.deepcopy(result["input_ids"])
            }

        return tokenize, tokenize


def get_train_val_data(args, data, tokenizer):
    def _generate_and_tokenize_prompt(data_point):
        prompt_no_resp = generate_prompt(data_point)
        prompt_tokenize, completion_tokenize = get_tokenize_func(args, tokenizer)
        tokenized_result = prompt_tokenize(prompt_no_resp)
        source_len = len(tokenized_result['input_ids'])
        prompt_with_response = prompt_no_resp + " " + data_point["output"]
        prompt_with_response += " " + tokenizer.eos_token
        tokenized_with_response = completion_tokenize(prompt_with_response)
        tokenized_with_response["labels"] = [IGNORE_INDEX] * source_len + tokenized_with_response["labels"][source_len:]
        return tokenized_with_response

    def _generate_and_tokenize_prompt_4_chatglm(data_point):
        prompt_no_resp = generate_prompt_4_chatglm(data_point)
        prompt_tokenize, completion_tokenize = get_tokenize_func(args, tokenizer)
        tokenized_result = prompt_tokenize(prompt_no_resp)
        source_len = len(tokenized_result['input_ids'])
        prompt_with_response = prompt_no_resp + " " + data_point["output"]
        prompt_with_response += " " + tokenizer.eos_token
        tokenized_with_response = completion_tokenize(prompt_with_response)
        tokenized_with_response["labels"] = [IGNORE_INDEX] * source_len + tokenized_with_response["labels"][source_len:]
        return tokenized_with_response

    if args.val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=args.val_set_size, shuffle=True, seed=42
        )
        if args.model_type in ['chatglm']:
            train_data = train_val["train"].shuffle().map(_generate_and_tokenize_prompt_4_chatglm)
            val_data = train_val["test"].shuffle().map(_generate_and_tokenize_prompt_4_chatglm)
        else:
            train_data = train_val["train"].shuffle().map(_generate_and_tokenize_prompt)
            val_data = train_val["test"].shuffle().map(_generate_and_tokenize_prompt)
    else:
        if args.model_type in ['chatglm']:
            train_data = data["train"].shuffle().map(_generate_and_tokenize_prompt_4_chatglm)
        else:
            train_data = data["train"].shuffle().map(_generate_and_tokenize_prompt)
        val_data = None
    return train_data, val_data


def get_predict_data(args):
    data = load_dataset("json", data_files=args.data)
    print(data)
    if args.model_type in ['chatglm']:
        predict_data = data["train"].shuffle().map(generate_prompt_4_chatglm_dict)
    else:
        predict_data = data["train"].shuffle().map(generate_prompt_dict)
    return predict_data


def get_fine_tuned_model(args):
    if (args.model_type == "baichuan"):
        def _get_model_class(llm_type):
            if llm_type not in AVAILABLE_MODEL:
                llm_type = "Auto"
            return MODEL_CLASSES[llm_type]

        model_path = args.model_path
        model_class = _get_model_class(args.model_type)
    else:
        def _get_model_class(llm_type, model_path):
            if llm_type not in AVAILABLE_MODEL:
                llm_type = "Auto"
                return MODEL_CLASSES[llm_type], model_path
            else:
                load_path = llm_type + "_" + model_path
                if llm_type in ['moss']:
                    load_path = llm_type
                return MODEL_CLASSES[llm_type], COMMON_PATH + MODEL_PATHS[load_path]

        model_class, model_path = _get_model_class(args.model_type, args.size)

    if args.model_type == "chatglm":
        model = model_class.model.from_pretrained(model_path,
                                                  trust_remote_code=True,
                                                  torch_dtype=torch.float16,
                                                  device_map=DEVICE_MAP)
        tokenizer = model_class.tokenizer.from_pretrained(model_path,
                                                          trust_remote_code=True)
        if args.lora_dir != 'none':
            peft_path = args.lora_dir + "/adapter_model.bin"
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=True,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=MODEL_LORA_TARGET_MODULES[args.model_type],
            )
            model = get_peft_model(model, peft_config)
            model.load_state_dict(torch.load(peft_path, map_location="cpu"), strict=False)
            # torch.set_default_tensor_type(torch.cuda.FloatTensor)
    elif args.model_type == "moss":
        model = model_class.model.from_pretrained(model_path,
                                                  trust_remote_code=True,
                                                  load_in_8bit=False,
                                                  torch_dtype=torch.float16,
                                                  device_map=get_device_map(model_type="moss", load_in_8bit=True))

        tokenizer = model_class.tokenizer.from_pretrained(model_path, trust_remote_code=True)
        if args.lora_dir != 'none':
            model = PeftModel.from_pretrained(
                model,
                args.lora_dir,
                device_map={"": DEVICE_TYPE}
            )
    elif args.model_type == "baichuan":
        baichuan_config = AutoConfig.from_pretrained(model_path,
                                                     trust_remote_code=True, )
        tokenizer = model_class.tokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = 0  # set as the <unk> token

        config_kwargs = {}
        # Quantization configurations by bitsandbytes
        if args.quantization_bit is not None:
            if args.quantization_bit == 8:
                require_version("bitsandbytes>=0.37.0", "To fix: pip install bitsandbytes>=0.37.0")
                config_kwargs["load_in_8bit"] = True
                config_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0
                )

            elif args.quantization_bit == 4:
                require_version("bitsandbytes>=0.39.0", "To fix: pip install bitsandbytes>=0.39.0")
                require_version("transformers>=4.30.1", "To fix: pip install transformers>=4.30.1")
                require_version("accelerate>=0.20.3", "To fix: pip install accelerate>=0.20.3")
                require_version("peft>=0.4.0.dev0", "To fix: pip install git+https://github.com/huggingface/peft.git")
                config_kwargs["load_in_4bit"] = True
                config_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=None,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )

            config_kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK", "0"))}
            print("Quantizing model to {} bit.".format(args.quantization_bit))

        # `device_map=auto` should be used for inference only
        config_kwargs["device_map"] = "auto"

        # Load and prepare pretrained models (without valuehead).
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=baichuan_config,
            torch_dtype=torch.bfloat16 if args.compute_dtype == "bf16" else torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            **config_kwargs
        )
        model.generation_config = GenerationConfig.from_pretrained(model_path)

        # Register auto class to save the custom code files.
        if hasattr(baichuan_config, "auto_map") and "AutoConfig" in baichuan_config.auto_map:
            baichuan_config.__class__.register_for_auto_class()
        if hasattr(baichuan_config, "auto_map") and "AutoTokenizer" in baichuan_config.auto_map:
            tokenizer.__class__.register_for_auto_class()
        if hasattr(baichuan_config, "auto_map") and "AutoModelForCausalLM" in baichuan_config.auto_map:
            model.__class__.register_for_auto_class()

        if args.lora_dir != "none":
            print("loading LoRA weight")
            model = PeftModel.from_pretrained(
                model,
                args.lora_dir,
                device_map={"": DEVICE_TYPE}
            )
            model.requires_grad_(False)

    else:
        model = model_class.model.from_pretrained(model_path,
                                                  load_in_8bit=False,
                                                  torch_dtype=torch.float16,
                                                  device_map=DEVICE_MAP)

        tokenizer = model_class.tokenizer.from_pretrained(model_path)
        if args.lora_dir != 'none':
            model = PeftModel.from_pretrained(
                model,
                args.lora_dir,
                device_map={"": DEVICE_TYPE}
            )
    model = model.half() if args.quantization_bit is None else model
    return model, tokenizer


def get_lora_model(args):
    def _get_model_class(llm_type, model_path):
        if llm_type not in AVAILABLE_MODEL:
            llm_type = "Auto"
            return MODEL_CLASSES[llm_type], model_path
        else:
            load_path = llm_type + "_" + model_path
            return MODEL_CLASSES[llm_type], COMMON_PATH + MODEL_PATHS[load_path]

    model_class, model_path = _get_model_class(args.model_type, args.size)

    if args.model_type == "chatglm":
        model = model_class.model.from_pretrained(model_path,
                                                  trust_remote_code=True,
                                                  device_map={"": "cpu"}, )
    else:
        model = model_class.model.from_pretrained(model_path,
                                                  load_in_8bit=False,
                                                  torch_dtype=torch.float16,
                                                  device_map={"": "cpu"}, )
    if args.lora_dir != 'none':
        lora_model = PeftModel.from_pretrained(
            model,
            args.lora_dir,
            torch_dtype=torch.float16,
            device_map={"": "cpu"},
        )
    else:
        lora_model = None

    if 'q_proj' in MODEL_LORA_TARGET_MODULES[args.model_type] and 'v_proj' in MODEL_LORA_TARGET_MODULES[
        args.model_type]:
        lora_type = 'q_v_proj'
    elif 'query_key_value' in MODEL_LORA_TARGET_MODULES[args.model_type]:
        lora_type = 'query_key_value'
    else:
        lora_type = None
    return model, lora_model, lora_type, model_class


def generate_service_prompt(instruction, llm, lora):
    if lora == 'none':
        if llm in ['chatglm']:
            return instruction
        else:
            return PROMPT_DICT['prompt_format_before'] + instruction + PROMPT_DICT['prompt_format_after']
    else:
        if llm in ['moss']:
            return META_INSTRUCTION.get('moss', "") + PROMPT_DICT['prompt_format_before'] + instruction + PROMPT_DICT[
                'prompt_format_after']
        return PROMPT_DICT['prompt_format_before'] + instruction + PROMPT_DICT['prompt_format_after']


def get_generation_config(llm):

    generation_configs = GenerationConfig(
        temperature=GENERATE_CONFIG['temperature'],
        top_p=GENERATE_CONFIG['top_p'],
        top_k=GENERATE_CONFIG['top_k'],
        num_beams=GENERATE_CONFIG['num_beams'],
        max_new_tokens=GENERATE_CONFIG['max_new_tokens']
    )
    return generation_configs


def generate_service_output(output, prompt, llm, lora):
    if lora == 'none':
        if llm in ['llama']:
            return output.replace(prompt, '', 1).strip()
        elif llm in ['chatglm']:
            return output.replace(prompt, '', 1).strip()
        else:
            return output.split("### Response:")[1].strip()
    else:
        return output.split("### Response:")[1].strip()


def prepare_model_for_training(
        model: PreTrainedModel,
        output_embedding_layer_name: Optional[str] = "lm_head",
        use_gradient_checkpointing: Optional[bool] = True,
        layer_norm_names: Optional[List[str]] = ["norm", "ln_f", "ln_attn", "ln_mlp"]
        # for LLaMA, BLOOM and Falcon settings
) -> PreTrainedModel:
    for name, param in model.named_parameters():
        if param.ndim == 1 and any(layer_norm_name in name for layer_norm_name in layer_norm_names):
            param.data = param.data.to(torch.float32)

    if use_gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        model.gradient_checkpointing_enable()
        model.config.use_cache = False  # turn off when gradient checkpointing is enabled

    if hasattr(model, output_embedding_layer_name):
        output_embedding_layer: torch.nn.Linear = getattr(model, output_embedding_layer_name)
        input_dtype = output_embedding_layer.weight.dtype

        class CastOutputToFloat(torch.nn.Sequential):

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return super().forward(x.to(input_dtype)).to(torch.float32)

        setattr(model, output_embedding_layer_name, CastOutputToFloat(output_embedding_layer))

    return model
