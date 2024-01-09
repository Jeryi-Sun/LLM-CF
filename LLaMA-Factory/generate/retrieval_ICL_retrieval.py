"""
梳理一下现在的数据
1. f"{root_path}/{data_name}/items2text_emb.pkl" item2emb 的数据
2. cot_data_user_pos_path = "/home/web_server/code/RRecagent/LLaMA-Factory/data/{data_name}/ready-to-use/ctr/train_cot_user_pos.pkl" cot 的数据
3. 将 cot data 根据item2emb转化为 embedding，每条数据两个 embedding，一个 history 的一个 Target item 的
4. /home/web_server/code/RRecagent/LLaMA-Factory/data/beauty/dataset 这里包含了 train.tsv test.tsv valid.tsv 三个，现在需要为其中的每一个都去train_cot_user_pos中算分数
5. 算完分数之后需要 
    a. 删除后来的历史 
        在 user id 相同的情况下，需要保持 pos id 小于 当前的 len(eval(history))
    b. 保持多样性 正例子负例子都有
"""
import os
import random
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import pickle
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
root_path = "/home/web_server/code/RRecagent/LLaMA-Factory/data/"
model = SentenceTransformer('/media/disk1/fordata/web_server/LLMs/match_model/')


def convert_cot2embedding(cot_data, item2emb):
    cot_emb_history_list = []
    cot_emb_target_item_list = []
    for item in cot_data:
        history = eval(item["history"])
        temp_history_emb = [item2emb[str(i)] for i in history]
        temp_history_emb_np = np.array(temp_history_emb)
        temp_history_emb_list = np.mean(temp_history_emb_np, axis=0).tolist()
        cot_emb_history_list.append(temp_history_emb_list)
        cot_emb_target_item_list.append(item2emb[str(item["item"])])
    return torch.tensor(cot_emb_history_list,device='cuda'), torch.tensor(cot_emb_target_item_list,device='cuda')    


def get_top_k(n, cot_data, score_index, user_id, pos_id):
    click_list = []
    no_click_list = []
    for idx in score_index:
        if str(cot_data[idx]['user_id'])==str(user_id) and int(pos_id)<int(cot_data[idx]['pos_id']):
            continue
        if len(click_list)<n//2:
            if cot_data[idx]["label"]==1.0:
                click_list.append(idx)

        if len(no_click_list)<n//2:
            if cot_data[idx]["label"]==0.0:
                no_click_list.append(idx)
        if len(click_list)>=n//2 and len(no_click_list)>=n//2:
            combined_list = click_list + no_click_list
            # Randomly shuffling the combined list
            random.shuffle(combined_list)
            return combined_list
        



def get_icl_list(tsv_data, cot_emb_history_cuda, cot_emb_target_item_cuda, cot_data):
    icl_list = []
    for i in tqdm(range(len(tsv_data))):
        history = eval(tsv_data.loc[i]['history'])
        target_item = tsv_data.loc[i]['item']
        history_emb = [item2emb[str(i)] for i in history]
        history_emb_np = np.array(history_emb)
        history_emb_cuda = torch.tensor(np.mean(history_emb_np, axis=0), device='cuda')
        target_emb_cuda = torch.tensor(item2emb[str(target_item)], device='cuda')

        """
        history_emb_cuda 与 target_emb_cuda 都是size=(768,) torch 向量；cot_emb_history_cuda size=(10000, 768), cot_emb_target_item_cuda  size=(10000, 768);
        请分别计算:
        history_emb_cuda与cot_emb_history_cuda每一行相乘求和得到的 history_score size=(10000,)
        target_emb_cuda 与cot_emb_target_item_cuda每一行相乘求和得到的 target_score size=(10000,)
        """
        history_score = torch.matmul(cot_emb_history_cuda.float(), history_emb_cuda.float())
        target_score = torch.matmul(cot_emb_target_item_cuda.float(), target_emb_cuda.float())
        # Calculating the weighted average of history_score and target_score
        weighted_avg = history_score # important !!!

        # Getting indices of the sorted array in descending order
        sorted_indices = torch.argsort(weighted_avg, descending=True).tolist()

        user_id = tsv_data.loc[i]['user_id']
        pos_id = len(eval(tsv_data.loc[i]['history']))
        icl_ids = get_top_k(4, cot_data, sorted_indices, user_id, pos_id)
        icl_list.append(icl_ids)
    tsv_data["ICL"] = icl_list
    return tsv_data




for data_name in {"beauty", "sports", "toys"}:
    cot_data_user_pos_path = root_path + f"{data_name}/ready-to-use/ctr/train_cot_user_pos.pkl"
    item2emb_path = f"{root_path}/{data_name}/items2text_emb.pkl"
    with open(cot_data_user_pos_path, 'rb') as file:
        cot_data = pickle.load(file)
    with open(item2emb_path, 'rb') as file:
        item2emb = pickle.load(file)

    cot_emb_history_cuda, cot_emb_target_item_cuda = convert_cot2embedding(cot_data, item2emb)
    print("cot_emb_history_cuda: ", cot_emb_history_cuda.shape)
    print("cot_emb_target_item_cuda: ", cot_emb_history_cuda.shape)
    train_data_path =f"/home/web_server/code/RRecagent/LLaMA-Factory/data/{data_name}/dataset/training.tsv"
    valid_data_path = f"/home/web_server/code/RRecagent/LLaMA-Factory/data/{data_name}/dataset/validation.tsv"
    test_data_path = f"/home/web_server/code/RRecagent/LLaMA-Factory/data/{data_name}/dataset/test.tsv"

    train_data = pd.read_csv(train_data_path, sep='\t')
    valid_data = pd.read_csv(valid_data_path, sep='\t')
    test_data = pd.read_csv(test_data_path, sep='\t')

    train_data_cot = get_icl_list(train_data, cot_emb_history_cuda, cot_emb_target_item_cuda, cot_data)
    print(train_data_cot.head())
    valid_data_cot = get_icl_list(valid_data, cot_emb_history_cuda, cot_emb_target_item_cuda, cot_data)
    test_data_cot = get_icl_list(test_data, cot_emb_history_cuda, cot_emb_target_item_cuda, cot_data)
    train_data_save_path =f"/home/web_server/code/RRecagent/LLaMA-Factory/data/{data_name}/dataset/train_icl_recall.tsv"
    valid_data_save_path = f"/home/web_server/code/RRecagent/LLaMA-Factory/data/{data_name}/dataset/valid_icl_recall.tsv"
    test_data_save_path = f"/home/web_server/code/RRecagent/LLaMA-Factory/data/{data_name}/dataset/test_icl_recall.tsv"

    train_data_cot.to_csv(train_data_save_path, index=False, sep='\t')
    valid_data_cot.to_csv(valid_data_save_path, index=False, sep='\t')
    test_data_cot.to_csv(test_data_save_path, index=False, sep='\t')









    
    





