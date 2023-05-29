import os
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

class SavePeftModelCallback(TrainerCallback):
    def on_save(self,args: TrainingArguments,state: TrainerState,control: TrainerControl,**kwargs,):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        kwargs["model"].save_pretrained(checkpoint_folder)
        
        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            try:
                os.remove(pytorch_model_path)
            except:
                pass
        return control