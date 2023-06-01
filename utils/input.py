import torch
import numpy as np

class ChatGLMCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list) -> dict:
        seq_length = max([len(feature["input_ids"]) for feature in features]) + 1
        input_ids_list, attention_mask_list, position_ids_list, labels_list = [], [], [], []
        for feature in features:
            input_ids = feature["input_ids"] + [self.tokenizer.eos_token_id] * (seq_length - len(feature["input_ids"]))
            input_ids_list.append(input_ids)

            context_length = feature["input_ids"].index(self.tokenizer.bos_token_id)
            attention_mask = np.ones((1, seq_length, seq_length))
            attention_mask = np.tril(attention_mask)
            attention_mask[:, :, :context_length] = 1
            attention_mask = np.bool_(attention_mask < 0.5)
            attention_mask_list.append(attention_mask)

            labels = feature["labels"] + [-100] * (seq_length - len(feature["labels"]))
            labels_list.append(labels)

            position_ids = [np.append(np.arange(context_length), np.ones([seq_length-context_length])*(context_length-1))]
            position_ids.append(np.append(np.zeros([context_length-1]), np.arange(seq_length-context_length+1)))
            position_ids_list.append(position_ids)
        return {"input_ids":      torch.LongTensor(np.array(input_ids_list)),
                "labels":         torch.LongTensor(np.array(labels_list)),
                "attention_mask": torch.BoolTensor(np.array(attention_mask_list)),
                "position_ids":   torch.LongTensor(np.array(position_ids_list)),
                }
