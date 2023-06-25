CUDA_VISIBLE_DEVICES=0 python pipeline.py --input belle --model_type llama --model_name_or_path <path2basemodel> --lora_dir <path2LoraModel> --result ./result/belle --device 0 --batch_size 20

CUDA_VISIBLE_DEVICES=1 python pipeline.py --input mmcu --model_type llama --model_name_or_path <path2basemodel> --lora_dir <path2LoraModel> --result ./result/mmcu  --device 0 --batch_size 20

CUDA_VISIBLE_DEVICES=2 python generate.py --input belle --model_type llama --model_name_or_path <path2basemodel> --lora_dir <path2LoraModel> --result ./result/belle --device 0 --batch_size 1

CUDA_VISIBLE_DEVICES=3 python generate.py --input mmcu --model_type llama --model_name_or_path <path2basemodel> --lora_dir <path2LoraModel> --result ./result/mmcu  --device 0 --batch_size 1