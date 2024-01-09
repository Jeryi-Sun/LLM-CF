#!/bin/bash

# 运行命令，使用 beauty 相关的参数
python llama2_generate.py --model_path "LLMs/llama2-rec/amazon_beauty/rec_gen_ctr" \
                    --dataset_path "LLaMA-Factory/data/beauty/ready-to-use/ctr/train_generate_again.pkl" \
                    --output_dataset_path "LLaMA-Factory/data/beauty/ready-to-use/ctr/train_cot_v3.pkl" \
                    --output_dataset_vector "LLaMA-Factory/data/beauty/ready-to-use/ctr/train_cot.npz" \
                    --gpu_ids "4,5,6,7"


