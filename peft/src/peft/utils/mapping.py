from .other import bloom_model_postprocess_past_key_value


TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING = {
    "t5": ["q", "v"],
    "mt5": ["q", "v"],
    "bart": ["q_proj", "v_proj"],
    "gpt2": ["c_attn"],
    "bloom": ["query_key_value"],
    "blip-2": ["q", "v", "q_proj", "v_proj"], # *
    "opt": ["q_proj", "v_proj"],
    "gptj": ["q_proj", "v_proj"],
    "gpt_neox": ["query_key_value"],
    "gpt_neo": ["q_proj", "v_proj"],
    "bert": ["query", "value"],
    "roberta": ["query", "value"],
    "xlm-roberta": ["query", "value"],
    "electra": ["query", "value"],
    "deberta-v2": ["query_proj", "value_proj"],
    "deberta": ["in_proj"],
    "layoutlm": ["query", "value"],
    "llama": ["q_proj", "v_proj"], # **
    "chatglm": ["query_key_value"], # **
}

TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING = {
    "t5": ["q", "k", "v", "o", "wi", "wo"],
    "mt5": ["q", "k", "v", "o", "wi_0", "wi_1", "wo"],
    "bart": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    # "gpt2": ["c_attn"],
    # "bloom": ["query_key_value"],
    "opt": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    # "gptj": ["q_proj", "v_proj"],
    # "gpt_neox": ["query_key_value"],
    # "gpt_neo": ["q_proj", "v_proj"],
    # "bert": ["query", "value"],
    "roberta": ["query", "key", "value", "dense"],
    # "xlm-roberta": ["query", "value"],
    # "electra": ["query", "value"],
    "deberta-v2": ["query_proj", "key_proj", "value_proj", "dense"],
    # "deberta": ["in_proj"],
    # "layoutlm": ["query", "value"],
}

TRANSFORMERS_MODELS_TO_BOTTLENECK_TARGET_MODULES_MAPPING = {
    "bloom": ["dense_h_to_4h", "dense_4h_to_h"],
    "gptj": ["fc_in", "fc_out"],
    "gpt_neo": ["c_fc", "c_proj"],
    "llama": ["gate_proj", "up_proj", "down_proj"],
    "opt": ["fc1", "fc2"],
    "chatglm": ["dense_h_to_4h", "dense_4h_to_h"],
}

TRANSFORMERS_MODELS_TO_ADAPTERP_TARGET_MODULES_MAPPING = {
    "bloom": ["dense_4h_to_h"],
    "gptj": ["fc_out"],
    "gpt_neo": ["c_proj"],
    "llama": ["down_proj"],
    "opt": ["fc2"],
    "chatglm": ["dense_4h_to_h"],
}

TRANSFORMERS_MODELS_TO_PARALLEL_ADAPTER_TARGET_MODULES_MAPPING = {
    "bloom": ["query_key_value"],
    "gptj": ["q_proj", "v_proj", "k_proj"],
    "gpt_neo": ["q_proj", "v_proj", "k_proj"],
    "llama": ["q_proj", "v_proj", "k_proj"],
    "opt": ["q_proj", "v_proj", "k_proj"],
    "chatglm": ["query_key_value"],
}

TRANSFORMERS_MODELS_TO_ADAPTER_TYPE_MAPPING = {
    "bloom": {"dense_h_to_4h": "mh_adapter", "dense_4h_to_h": "output_adapter"},
    "gptj": {"fc_in":"mh_adapter", "fc_out":"output_adapter"},
    "gpt_neo": {"c_fc":"mh_adapter", "c_proj":"output_adapter"},
    "llama": {"gate_proj": "mh_adapter", "up_proj":"mh_adapter", "down_proj":"output_adapter"},
    "opt": {"fc1":"mh_adapter", "fc2":"output_adapter"},
    "chatglm": {"dense_h_to_4h": "mh_adapter", "dense_4h_to_h": "output_adapter"},
}

TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING = {
    "bloom": bloom_model_postprocess_past_key_value,
}

WEIGHTS_NAME = "adapter_model.bin"
CONFIG_NAME = "adapter_config.json"
