python src/eval_rec.py \
    --base_model /media/disk1/fordata/web_server/LLMs/llama2-rec/amazon_toys/rec_gen_ctr/    \
    --test_data_path /home/web_server/code/RRecagent/LLaMA-Factory/data/toys/ready-to-use/ctr/test.json \
    --result_json_data /home/web_server/code/RRecagent/LLaMA-Factory/eval_result/toys/amazon-rec-gen-ctr.json

python src/eval_rec.py \
    --base_model /media/disk1/fordata/web_server/LLMs/llama2-rec/amazon_sports/rec_gen_ctr/    \
    --test_data_path /home/web_server/code/RRecagent/LLaMA-Factory/data/sports/ready-to-use/ctr/test.json \
    --result_json_data /home/web_server/code/RRecagent/LLaMA-Factory/eval_result/sports/amazon-rec-gen-ctr.json

python src/eval_rec.py \
    --base_model /media/disk1/fordata/web_server/LLMs/llama2-rec/amazon_beauty/rec_gen_ctr/    \
    --test_data_path /home/web_server/code/RRecagent/LLaMA-Factory/data/beauty/ready-to-use/ctr/test.json \
    --result_json_data /home/web_server/code/RRecagent/LLaMA-Factory/eval_result/beauty/amazon-rec-gen-ctr.json


