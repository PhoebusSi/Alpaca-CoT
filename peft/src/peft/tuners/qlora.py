import copy
import bitsandbytes as bnb
import importlib
import re
import warnings
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from ..utils import (
    TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING,
    PeftType,
    _freeze_adapter,
    _get_submodules,
    transpose,
    is_bnb_available
)

from .lora import (
    LoraConfig,
    LoraLayer,
    LoraModel,
    mark_only_lora_as_trainable,
)

if is_bnb_available():
    import bitsandbytes as bnb


def is_ipex_available():
    def get_major_and_minor_from_version(full_version):
        return str(version.parse(full_version).major) + "." + str(version.parse(full_version).minor)

    _torch_version = importlib.metadata.version("torch")
    if importlib.util.find_spec("intel_extension_for_pytorch") is None:
        return False
    _ipex_version = "N/A"
    try:
        _ipex_version = importlib.metadata.version("intel_extension_for_pytorch")
    except importlib.metadata.PackageNotFoundError:
        return False
    torch_major_and_minor = get_major_and_minor_from_version(_torch_version)
    ipex_major_and_minor = get_major_and_minor_from_version(_ipex_version)
    if torch_major_and_minor != ipex_major_and_minor:
        warnings.warn(
            f"Intel Extension for PyTorch {ipex_major_and_minor} needs to work with PyTorch {ipex_major_and_minor}.*,"
            f" but PyTorch {_torch_version} is found. Please switch to the matching version and run again."
        )
        return False
    return True
    


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="EleutherAI/pythia-12b"
    )
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."}
    )
    use_auth_token: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables using Huggingface auth token from Git Credentials."}
    )

@dataclass
class DataArguments:
    eval_dataset_size: int = field(
        default=1024, metadata={"help": "Size of validation dataset."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    source_max_len: int = field(
        default=1024,
        metadata={"help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    target_max_len: int = field(
        default=256,
        metadata={"help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    dataset: str = field(
        default='alpaca',
        metadata={"help": "Which dataset to finetune on. See datamodule for options."}
    )
    dataset_format: Optional[str] = field(
        default=None,
        metadata={"help": "Which dataset format is used. [alpaca|chip2|self-instruct|hh-rlhf]"}
    )

def find_all_linear_names(args, model):
    cls = bnb.nn.Linear4bit if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        print('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)

        touch(join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)

def get_accelerate_model(args, checkpoint_dir):

    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
    if is_ipex_available() and torch.xpu.is_available():
        n_gpus = torch.xpu.device_count()
        
    max_memory = f'{args.max_memory_MB}MB'
    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = "auto"

    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
        max_memory = {'': max_memory[local_rank]}


    if args.full_finetune: assert args.bits in [16, 32]

    print(f'loading base model {args.model_name_or_path}...')
    compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        load_in_4bit=args.bits == 4,
        load_in_8bit=args.bits == 8,
        device_map=device_map,
        max_memory=max_memory,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.double_quant,
            bnb_4bit_quant_type=args.quant_type,
        ),
        torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
        trust_remote_code=args.trust_remote_code,
        use_auth_token=args.use_auth_token
    )
    if compute_dtype == torch.float16 and args.bits == 4:
        if torch.cuda.is_bf16_supported():
            print('='*80)
            print('Your GPU supports bfloat16, you can accelerate training with the argument --bf16')
            print('='*80)
            
    if compute_dtype == torch.float16 and (is_ipex_available() and torch.xpu.is_available()):
        compute_dtype = torch.bfloat16
        print('Intel XPU does not support float16 yet, so switching to bfloat16')

    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    model.config.torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        padding_side="right",
        use_fast=False, # Fast tokenizer giving issues.
        tokenizer_type='llama' if 'llama' in args.model_name_or_path else None, # Needed for HF name change
        trust_remote_code=args.trust_remote_code,
        use_auth_token=args.use_auth_token,
    )
    if tokenizer._pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    if 'llama' in args.model_name_or_path or isinstance(tokenizer, LlamaTokenizer):
        # LLaMA tokenizer may not have correct special tokens set.
        # Check and add them if missing to prevent them from being parsed into different tokens.
        # Note that these are present in the vocabulary.
        # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.
        print('Adding special tokens.')
        tokenizer.add_special_tokens({
                "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
                "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
                "unk_token": tokenizer.convert_ids_to_tokens(
                    model.config.pad_token_id if model.config.pad_token_id != -1 else tokenizer.pad_token_id
                ),
        })
    
    if not args.full_finetune:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

    if not args.full_finetune:
        if checkpoint_dir is not None:
            print("Loading adapters from checkpoint.")
            model = PeftModel.from_pretrained(model, join(checkpoint_dir, 'adapter_model'), is_trainable=True)
        else:
            print(f'adding LoRA modules...')
            modules = find_all_linear_names(args, model)
            config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=modules,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    return model, tokenizer

def print_trainable_parameters(args, model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    if args.bits == 4: trainable_params /= 2
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    
    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg

@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [f"{self.tokenizer.bos_token}{example['input']}" for example in instances]
        targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt['input_ids'],
            tokenized_targets['input_ids']
        ):
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    labels.append(
                        torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
                    )
                else:
                    labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target)))
            else:
                input_ids.append(torch.tensor(tokenized_source))
        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) if not self.predict_with_generate else None
        data_dict = {
            'input_ids': input_ids,
            'attention_mask':input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict['labels'] = labels
        return data_dict

def extract_unnatural_instructions_data(examples, extract_reformulations=False):
    out = {
        'input': [],
        'output': [],
    }
    for example_instances in examples['instances']:
        for instance in example_instances:
            out['input'].append(instance['instruction_with_input'])
            out['output'].append(instance['output'])
    if extract_reformulations:
        for example_reformulations in examples['reformulations']:
            if example_reformulations is not None:
                for instance in example_reformulations:
                    out['input'].append(instance['instruction_with_input'])
                    out['output'].append(instance['output'])
    return out

ALPACA_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    ),
}


@dataclass
class QLoraConfig(LoraConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.AdaLora`].

    Args:
        target_r (`int`): The target average rank of incremental matrix.
        init_r (`int`): The initial rank for each incremental matrix.
        tinit (`int`): The steps of initial fine-tuning warmup.
        tfinal (`int`): The step of final fine-tuning.
        deltaT (`int`): The time internval between two budget allocations.
        beta1 (`float`): The hyperparameter of EMA for sensitivity smoothing.
        beta2 (`float`): The hyperparameter of EMA for undertainty quantification.
        orth_reg_weight (`float`): The coefficient of orthogonal regularization.
        total_step (`int`): The total training steps that should be specified before training.
        rank_pattern (`list`): The allocated rank for each weight matrix by RankAllocator.
    """

    target_r: int = field(default=8, metadata={"help": "Target Lora matrix dimension."})
    init_r: int = field(default=12, metadata={"help": "Intial Lora matrix dimension."})
    tinit: int = field(default=0, metadata={"help": "The steps of initial warmup."})
    tfinal: int = field(default=0, metadata={"help": "The steps of final warmup."})
    deltaT: int = field(default=1, metadata={"help": "Step interval of rank allocation."})
    beta1: float = field(default=0.85, metadata={"help": "Hyperparameter of EMA."})
    beta2: float = field(default=0.85, metadata={"help": "Hyperparameter of EMA."})
    orth_reg_weight: float = field(default=0.5, metadata={"help": "The orthogonal regularization coefficient."})
    total_step: Optional[int] = field(default=None, metadata={"help": "The total training steps."})
    rank_pattern: Optional[dict] = field(default=None, metadata={"help": "The saved rank pattern."})

    def __post_init__(self):
        self.peft_type = PeftType.QLORA

class QLoraModel(LoraModel):
    """
    Creates AdaLoRA (Adaptive LoRA) model from a pretrained transformers model. Paper:
    https://openreview.net/pdf?id=lq62uWRJjiY

    Args:
        model ([`transformers.PreTrainedModel`]): The model to be adapted.
        config ([`AdaLoraConfig`]): The configuration of the AdaLora model.

    Returns:
        `torch.nn.Module`: The AdaLora model.

    Example::

        >>> from transformers import AutoModelForSeq2SeqLM, LoraConfig >>> from peft import AdaLoraModel, AdaLoraConfig
        >>> config = AdaLoraConfig(
                peft_type="ADALORA", task_type="SEQ_2_SEQ_LM", r=8, lora_alpha=32, target_modules=["q", "v"],
                lora_dropout=0.01,
            )
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base") >>> model = AdaLoraModel(config, model)

    **Attributes**:
        - **model** ([`transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`AdaLoraConfig`]): The configuration of the AdaLora model.
    """

    def __init__(self, model, config, adapter_name):
        nn.Module.__init__(self)
        self.model = model
        self.peft_config = config
        self.add_adapter(adapter_name, self.peft_config[adapter_name])

    def add_adapter(self, adapter_name, config=None):
        if config is not None:
            model_config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
            config = self._prepare_adalora_config(config, model_config)
            self.peft_config[adapter_name] = config
        self._find_and_replace(adapter_name)
        if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
            raise ValueError(
                "AdaLoraModel supports only 1 adapter with bias. When using multiple adapters, set bias to 'none' for all adapters."
            )
        traininable_mode_counter = 0
        for config in self.peft_config.values():
            if not config.inference_mode:
                traininable_mode_counter += 1

        if traininable_mode_counter > 1:
            raise ValueError(
                "AdaLoraModel supports only 1 trainable adapter. "
                "When using multiple adapters, set inference_mode to True for all adapters except the one you want to train."
            )

        mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)
        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)
        else:
            self.trainable_adapter_name = adapter_name
            self.rankallocator = RankAllocator(self.model, self.peft_config[adapter_name], self.trainable_adapter_name)

    def _find_and_replace(self, adapter_name):
        lora_config = self.peft_config[adapter_name]
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if loaded_in_8bit and not is_bnb_available():
            raise ImportError(
                "To use Lora with 8-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )
        is_target_modules_in_base_model = False
        kwargs = {
            "r": lora_config.init_r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
        }
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if isinstance(lora_config.target_modules, str):
                target_module_found = re.fullmatch(lora_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in lora_config.target_modules)
            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = _get_submodules(self.model, key)
                bias = target.bias is not None
                if isinstance(target, LoraLayer):
                    target.update_layer(
                        adapter_name,
                        lora_config.init_r,
                        lora_config.lora_alpha,
                        lora_config.lora_dropout,
                        lora_config.init_lora_weights,
                    )
                else:
                    if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
                        kwargs.update(
                            {
                                "has_fp16_weights": target.state.has_fp16_weights,
                                "memory_efficient_backward": target.state.memory_efficient_backward,
                                "threshold": target.state.threshold,
                                "index": target.index,
                            }
                        )
                        new_module = SVDLinear8bitLt(
                            adapter_name, target.in_features, target.out_features, bias=bias, **kwargs
                        )
                    else:
                        if isinstance(target, torch.nn.Linear):
                            in_features, out_features = target.in_features, target.out_features
                            if kwargs["fan_in_fan_out"]:
                                warnings.warn(
                                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                                    "Setting fan_in_fan_out to False."
                                )
                                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
                        elif isinstance(target, Conv1D):
                            in_features, out_features = (
                                target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                            )
                            if not kwargs["fan_in_fan_out"]:
                                warnings.warn(
                                    "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                                    "Setting fan_in_fan_out to True."
                                )
                                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
                        else:
                            raise ValueError(
                                f"Target module {target} is not supported. "
                                f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
                            )
                        new_module = SVDLinear(adapter_name, in_features, out_features, bias=bias, **kwargs)

                    self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {lora_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def forward(self, *args, **kwargs):
        outputs = self.model.forward(*args, **kwargs)

        # Calculate the orthogonal regularization
        orth_reg_weight = self.peft_config[self.trainable_adapter_name].orth_reg_weight
        assert orth_reg_weight > 0

        if hasattr(outputs, "loss"):
            regu_loss = 0
            num_param = 0
            for n, p in self.model.named_parameters():
                if ("lora_A" in n or "lora_B" in n) and self.trainable_adapter_name in n:
                    para_cov = p @ p.T if "lora_A" in n else p.T @ p
                    I = torch.eye(*para_cov.size(), out=torch.empty_like(para_cov))
                    I.requires_grad = False
                    num_param += 1
                    regu_loss += torch.norm(para_cov - I, p="fro").to("cuda:0")# set default gpu
            regu_loss = regu_loss / num_param
            outputs.loss += orth_reg_weight * regu_loss
        return outputs

    def resize_modules_by_rank_pattern(self, rank_pattern, adapter_name):
        lora_config = self.peft_config[adapter_name]
        for name, rank_idx in rank_pattern.items():
            if isinstance(rank_idx, list):
                rank = sum(rank_idx)
            elif isinstance(rank_idx, torch.Tensor):
                rank_idx = rank_idx.view(-1)
                rank = rank_idx.sum().item()
            else:
                raise ValueError("Unexcepted type of rank_idx")
            key = ".".join(name.split(".")[0:-2]) if adapter_name in name else ".".join(name.split(".")[0:-1])
            _, target, _ = _get_submodules(self.model, key)
            lora_E_weights = target.lora_E[adapter_name][rank_idx]
            lora_A_weights = target.lora_A[adapter_name][rank_idx]
            lora_B_weights = target.lora_B[adapter_name][:, rank_idx]
            ranknum = target.ranknum[adapter_name]
            target.update_layer(
                adapter_name,
                rank,
                lora_config.lora_alpha,
                lora_config.lora_dropout,
                lora_config.init_lora_weights,
            )
            with torch.no_grad():
                if rank > 0:
                    target.lora_E[adapter_name].copy_(lora_E_weights)
                    target.lora_A[adapter_name].copy_(lora_A_weights)
                    target.lora_B[adapter_name].copy_(lora_B_weights)
                    # The scaling is exactly as the previous
                    target.ranknum[adapter_name].copy_(ranknum)

    def resize_state_dict_by_rank_pattern(self, rank_pattern, state_dict, adapter_name):
        for name, rank_idx in rank_pattern.items():
            rank = sum(rank_idx)
            prefix = ".".join(name.split(".")[0:-2]) if adapter_name in name else ".".join(name.split(".")[0:-1])
            for layer in ["lora_E", "lora_A", "lora_B"]:
                key = f"base_model.model.{prefix}.{layer}.{adapter_name}"
                if layer != "lora_B":
                    state_dict[key] = (
                        state_dict[key][rank_idx] if rank != state_dict[key].shape[0] else state_dict[key]
                    )
                else:
                    state_dict[key] = (
                        state_dict[key][:, rank_idx] if rank != state_dict[key].shape[1] else state_dict[key]
                    )
        return state_dict

    def update_and_allocate(self, global_step):
        lora_config = self.peft_config[self.trainable_adapter_name]
        # Update the importance score and allocate the budget
        if global_step < lora_config.total_step - lora_config.tfinal:
            _, rank_pattern = self.rankallocator.update_and_allocate(self.model, global_step)
            if rank_pattern:
                lora_config.rank_pattern = rank_pattern
        # Finalize the budget allocation
        elif global_step == lora_config.total_step - lora_config.tfinal:
            _, rank_pattern = self.rankallocator.update_and_allocate(self.model, global_step, force_mask=True)
            # for some reason, this freezes the trainable parameters and nothing gets updates
            # self.resize_modules_by_rank_pattern(rank_pattern, self.trainable_adapter_name)
            lora_config.rank_pattern = rank_pattern
            self.rankallocator.reset_ipt()
        # Currently using inefficient way to mask the unimportant weights using the rank pattern
        #  due to problem mentioned above
        elif global_step > lora_config.total_step - lora_config.tfinal:
            self.rankallocator.mask_using_rank_pattern(self.model, lora_config.rank_pattern)
        # Pass the function and do forward propagation
        else:
            return None

    @staticmethod
    def _prepare_adalora_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING[
                model_config["model_type"]
            ]
        if peft_config.inference_mode:
            peft_config.merge_weights = True
        return peft_config