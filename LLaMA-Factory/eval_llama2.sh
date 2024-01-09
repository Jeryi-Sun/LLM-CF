python src/evaluate.py \
    --model_name_or_path  /media/disk1/fordata/web_server/LLMs/llama2-rec/amazon_toys/rec_ctr/   \
    --finetuning_type none \
    --template vanilla \
    --task mmlu \
    --split test \
    --lang en \
    --n_shot 5 \
    --batch_size 16 \
    --save_name /home/web_server/code/RRecagent/LLaMA-Factory/eval_result/toys/mmlu-rec-ctr

python src/evaluate.py \
    --model_name_or_path  /media/disk1/fordata/web_server/LLMs/llama2-rec/amazon_sports/rec_ctr/    \
    --finetuning_type none \
    --template vanilla \
    --task mmlu \
    --split test \
    --lang en \
    --n_shot 5 \
    --batch_size 16 \
    --save_name /home/web_server/code/RRecagent/LLaMA-Factory/eval_result/sports/mmlu-rec-ctr

python src/evaluate.py \
    --model_name_or_path  /media/disk1/fordata/web_server/LLMs/llama2-rec/amazon_beauty/rec_ctr/  \
    --finetuning_type none \
    --template vanilla \
    --task mmlu \
    --split test \
    --lang en \
    --n_shot 5 \
    --batch_size 16 \
    --save_name /home/web_server/code/RRecagent/LLaMA-Factory/eval_result/beauty/mmlu-rec-ctr




python src/evaluate.py \
    --model_name_or_path  /media/disk1/fordata/web_server/LLMs/llama2-rec/amazon_toys/rec_gen_ctr/   \
    --finetuning_type none \
    --template vanilla \
    --task mmlu \
    --split test \
    --lang en \
    --n_shot 5 \
    --batch_size 16 \
    --save_name /home/web_server/code/RRecagent/LLaMA-Factory/eval_result/toys/mmlu-rec-gen-ctr

python src/evaluate.py \
    --model_name_or_path  /media/disk1/fordata/web_server/LLMs/llama2-rec/amazon_sports/rec_gen_ctr/    \
    --finetuning_type none \
    --template vanilla \
    --task mmlu \
    --split test \
    --lang en \
    --n_shot 5 \
    --batch_size 16 \
    --save_name /home/web_server/code/RRecagent/LLaMA-Factory/eval_result/sports/mmlu-rec-gen-ctr

python src/evaluate.py \
    --model_name_or_path  /media/disk1/fordata/web_server/LLMs/llama2-rec/amazon_beauty/rec_gen_ctr/  \
    --finetuning_type none \
    --template vanilla \
    --task mmlu \
    --split test \
    --lang en \
    --n_shot 5 \
    --batch_size 16 \
    --save_name /home/web_server/code/RRecagent/LLaMA-Factory/eval_result/beauty/mmlu-rec-gen-ctr



# # half

# python src/evaluate.py \
#     --model_name_or_path  /media/disk1/fordata/web_server/LLMs/llama2-rec/amazon_toys/rec_ctr_half/   \
#     --finetuning_type none \
#     --template vanilla \
#     --task mmlu \
#     --split test \
#     --lang en \
#     --n_shot 5 \
#     --batch_size 16 \
#     --save_name /home/web_server/code/RRecagent/LLaMA-Factory/eval_result/toys/mmlu-rec-ctr_half

# python src/evaluate.py \
#     --model_name_or_path  /media/disk1/fordata/web_server/LLMs/llama2-rec/amazon_sports/rec_ctr_half/    \
#     --finetuning_type none \
#     --template vanilla \
#     --task mmlu \
#     --split test \
#     --lang en \
#     --n_shot 5 \
#     --batch_size 16 \
#     --save_name /home/web_server/code/RRecagent/LLaMA-Factory/eval_result/sports/mmlu-rec-ctr_half

# python src/evaluate.py \
#     --model_name_or_path  /media/disk1/fordata/web_server/LLMs/llama2-rec/amazon_beauty/rec_ctr_half/  \
#     --finetuning_type none \
#     --template vanilla \
#     --task mmlu \
#     --split test \
#     --lang en \
#     --n_shot 5 \
#     --batch_size 16 \
#     --save_name /home/web_server/code/RRecagent/LLaMA-Factory/eval_result/beauty/mmlu-rec-ctr_half




# python src/evaluate.py \
#     --model_name_or_path  /media/disk1/fordata/web_server/LLMs/llama2-rec/amazon_toys/rec_gen_ctr_half/   \
#     --finetuning_type none \
#     --template vanilla \
#     --task mmlu \
#     --split test \
#     --lang en \
#     --n_shot 5 \
#     --batch_size 16 \
#     --save_name /home/web_server/code/RRecagent/LLaMA-Factory/eval_result/toys/mmlu-rec-gen-ctr_half

# python src/evaluate.py \
#     --model_name_or_path  /media/disk1/fordata/web_server/LLMs/llama2-rec/amazon_sports/rec_gen_ctr_half/    \
#     --finetuning_type none \
#     --template vanilla \
#     --task mmlu \
#     --split test \
#     --lang en \
#     --n_shot 5 \
#     --batch_size 16 \
#     --save_name /home/web_server/code/RRecagent/LLaMA-Factory/eval_result/sports/mmlu-rec-gen-ctr_half

# python src/evaluate.py \
#     --model_name_or_path  /media/disk1/fordata/web_server/LLMs/llama2-rec/amazon_beauty/rec_gen_ctr_half/  \
#     --finetuning_type none \
#     --template vanilla \
#     --task mmlu \
#     --split test \
#     --lang en \
#     --n_shot 5 \
#     --batch_size 16 \
#     --save_name /home/web_server/code/RRecagent/LLaMA-Factory/eval_result/beauty/mmlu-rec-gen-ctr_half

# # base
# python src/evaluate.py \
#     --model_name_or_path  /media/disk1/fordata/web_server/LLMs/llama2-hf/Llama-2-7b-chat-hf/  \
#     --finetuning_type none \
#     --template vanilla \
#     --task mmlu \
#     --split test \
#     --lang en \
#     --n_shot 5 \
#     --batch_size 16 \
#     --save_name /home/web_server/code/RRecagent/LLaMA-Factory/eval_result/beauty/mmlu-base
