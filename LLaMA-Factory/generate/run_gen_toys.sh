#!/bin/bash
python llama2_generate.py --model_path "LLMs/llama2-rec/amazon_toys/rec_gen_ctr" \
                    --dataset_path "LLaMA-Factory/data/toys/ready-to-use/ctr/train_generate_again.pkl"  \
                    --output_dataset_path "LLaMA-Factory/data/toys/ready-to-use/ctr/train_cot_v3.pkl" \
                    --output_dataset_vector "LLaMA-Factory/data/toys/ready-to-use/ctr/train_cot.npz" \
                    --gpu_ids "4,5,6,7"