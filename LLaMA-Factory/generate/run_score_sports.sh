#!/bin/bash

# 运行命令，使用 sports 相关的参数
python score_cot.py \
                    --output_dataset_path "LLaMA-Factory/data/sports/ready-to-use/ctr/train_cot_scored.pkl" \
                    --dataset_path "LLaMA-Factory/data/sports/ready-to-use/ctr/train_cot.pkl" \
