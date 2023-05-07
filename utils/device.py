import torch
from accelerate import init_empty_weights
from accelerate.utils import get_balanced_memory, infer_auto_device_map
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.modeling_utils import no_init_weights
from transformers.utils import ContextManagers


def get_device_map(model_type="moss", load_in_8bit=False):
    if model_type == "moss":
        cls = get_class_from_dynamic_module(
            class_reference="fnlp/moss-moon-003-sft--modeling_moss.MossForCausalLM", pretrained_model_name_or_path="fnlp/moss-moon-003-sft")
        config = AutoConfig.from_pretrained(
            "fnlp/moss-moon-003-sft", return_unused_kwargs=True, trust_remote_code=True)[0]
        with ContextManagers([no_init_weights(_enable=True), init_empty_weights()]):
            model = cls(config)
            max_memory = get_balanced_memory(model, dtype=torch.int8 if load_in_8bit else None,
                                             low_zero=False, no_split_module_classes=model._no_split_modules)
            device_map = infer_auto_device_map(
                model, dtype=torch.float16 if not load_in_8bit else torch.int8, max_memory=max_memory, no_split_module_classes=model._no_split_modules)
            device_map["transformer.wte"] = 0
            device_map["transformer.drop"] = 0
            device_map["transformer.ln_f"] = 0
            device_map["lm_head"] = 0
            return device_map
    return "auto"
