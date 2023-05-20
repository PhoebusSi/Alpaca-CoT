import os
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

class SavePeftModelCallback(TrainerCallback):
    def on_save(self,args: TrainingArguments,state: TrainerState,control: TrainerControl,**kwargs,):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        kwargs["model"].save_pretrained(checkpoint_folder)
        return control