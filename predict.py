import argparse
from transformers import pipeline
from utils.tools import *
import json

def predict(args):
    model, tokenizer = get_fine_tuned_model(args)
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    input_data = get_predict_data(args)
    def predict_and_write_to_file(input_data, batch_size):
        with open(args.result_dir, 'w') as f:
            for i in range(0, len(input_data['input']), batch_size):
                batch = input_data['input'][i:i + batch_size]
                origin = input_data['origin'][i:i + batch_size]
                generated_text = generator(batch, max_length=args.cutoff_len, num_return_sequences=1)
                for instruction, prompt, result in zip(origin, batch, generated_text):
                    res = result[0]['generated_text']
                    filter_res = generate_service_output(res, prompt, args.model_type, args.lora_dir)
                    instruction['generate'] = filter_res
                    str_info = json.dumps(instruction, ensure_ascii=False)
                    f.write(str_info + "\n")
                    f.flush()
    predict_and_write_to_file(input_data, args.predict_batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some llm info.')
    parser.add_argument('--model_type', type=str, default="belle_bloom", choices=AVAILABLE_MODEL,
                        help='the base structure (not the model) used for model or fine-tuned model')
    parser.add_argument('--size', type=str, default="7b",
                        help='the type for base model or the absolute path for fine-tuned model')
    parser.add_argument('--data', type=str, default="test", help='the data used for predicting')
    parser.add_argument('--lora_dir', type=str, default="none",
                        help='the path for fine-tuned lora params, none when not in use')
    parser.add_argument('--result_dir', default="saved_result.txt", type=str)
    parser.add_argument('--predict_batch_size', default=128, type=int)
    parser.add_argument('--lora_r', default=8, type=int)
    parser.add_argument('--lora_alpha', default=16, type=int)
    parser.add_argument('--lora_dropout', default=0.05, type=float)
    parser.add_argument('--cutoff_len', default=512, type=int)
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed serving')
    args = parser.parse_args()
    print(args)
    predict(args)
