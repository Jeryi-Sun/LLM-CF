# python3 main_aug.py --model SRGNN_aug --name SRGNN_aug --dataset beauty --workspace wksp_beauty/ --gpu_id 1 --epochs 50  --random_seed 2023
# python3 main_aug.py --model GRU4REC_aug --name GRU4REC_aug --dataset beauty --workspace wksp_beauty/ --gpu_id 1 --epochs 50  --random_seed 2023
# python3 main_aug.py --model SASREC_aug --name SASREC_aug --dataset beauty --workspace wksp_beauty/ --gpu_id 1 --epochs 50  --random_seed 2023
# python3 main_aug.py --model YoutubeDNN_aug --name YoutubeDNN_aug --dataset beauty --workspace wksp_beauty/ --gpu_id 1 --epochs 50  --random_seed 2023

python3 main.py --model SRGNN --name SRGNN --dataset beauty --workspace wksp_beauty/ --gpu_id 1 --epochs 50  --random_seed 2023
python3 main.py --model GRU4REC --name GRU4REC --dataset beauty --workspace wksp_beauty/ --gpu_id 1 --epochs 50  --random_seed 2023
python3 main.py --model SASREC --name SASREC --dataset beauty --workspace wksp_beauty/ --gpu_id 1 --epochs 50  --random_seed 2023
python3 main.py --model YoutubeDNN --name YoutubeDNN --dataset beauty --workspace wksp_beauty/ --gpu_id 1 --epochs 50  --random_seed 2023