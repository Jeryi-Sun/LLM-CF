#!/bin/bash
set -e 
bash ./sft_llama2_rec_gen.sh
bash ./sft_llama2_rec.sh
bash ./eval_llama2.sh
bash ./eval_rec.sh & bash ./eval_rec2.sh &
wait
echo "Both processes have finished executing."