import subprocess
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, help='which model to use', default='DeepFM')
parser.add_argument('--name', type=str, default='default')
parser.add_argument('--dataset', type=str, default=None)
parser.add_argument('--workspace', type=str, default=None)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--cot_name', type=str, default='default')
parser.add_argument('--num_aug', type=int, default='4')
parser.add_argument('--prefix_dir', type=str, default="augment_data")
parser.add_argument('--postfix_dir', type=str, default="/")

args = parser.parse_args()

random_seed_ls = [2023, 2024, 2025, 2026, 2027]

batch_cmd_ls = [
    f'python3 main_aug.py --gpu_id {args.gpu_id} --epochs 30 '+ \
    f'--name {args.name+"_"+str(seed)} --model {args.model} --dataset {args.dataset} '+\
    f'--workspace {args.workspace} --random_seed {seed} --cot_name {args.cot_name} --num_aug {args.num_aug} ' +\
    f'--prefix_dir {args.prefix_dir} --postfix_dir {args.postfix_dir}' 
    for seed in random_seed_ls
]


'''run batch experiments'''

for cmd in batch_cmd_ls:

    print("running cmd: ", cmd)
    p = subprocess.Popen(cmd, shell=True)
    p.wait()  
