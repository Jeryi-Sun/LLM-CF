#!/bin/bash

# 运行命令，使用 sports 相关的参数
python llama2_generate.py --model_path "LLMs/llama2-rec/amazon_sports/rec_gen_ctr" \
                    --dataset_path "LLaMA-Factory/data/sports/ready-to-use/ctr/train_generate_again.pkl" \
                    --output_dataset_path "LLaMA-Factory/data/sports/ready-to-use/ctr/train_cot_v3.pkl" \
                    --output_dataset_vector "LLaMA-Factory/data/sports/ready-to-use/ctr/train_cot.npz" \
                    --gpu_ids "0,1,2,3"
