
import os
import random
import pickle
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
root_path = "RRecagent/LLaMA-Factory/data/"
model = SentenceTransformer('LLMs/match_model/')


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
    return torch.tensor(cot_emb_history_list).to('cuda:0'), torch.tensor(cot_emb_target_item_list).to('cuda:0')    


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
        history_emb_cuda = torch.tensor(np.mean(history_emb_np, axis=0)).to('cuda:0')
        target_emb_cuda = torch.tensor(item2emb[str(target_item)]).to('cuda:0')

    
        history_score = torch.matmul(cot_emb_history_cuda.float(), history_emb_cuda.float())
        target_score = torch.matmul(cot_emb_target_item_cuda.float(), target_emb_cuda.float())
        # Calculating the weighted average of history_score and target_score
        weighted_avg = 0.5 * history_score + 0.5 * target_score

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

    combined_label = []

    

    cot_emb_history_cuda, cot_emb_target_item_cuda = convert_cot2embedding(cot_data, item2emb)
    print("cot_emb_history_cuda: ", cot_emb_history_cuda.shape)
    print("cot_emb_target_item_cuda: ", cot_emb_history_cuda.shape)
    train_data_path =f"LLaMA-Factory/data/{data_name}/dataset/training.tsv"
    valid_data_path = f"LLaMA-Factory/data/{data_name}/dataset/validation.tsv"
    test_data_path = f"LLaMA-Factory/data/{data_name}/dataset/test.tsv"

    train_data = pd.read_csv(train_data_path, sep='\t')
    valid_data = pd.read_csv(valid_data_path, sep='\t')
    test_data = pd.read_csv(test_data_path, sep='\t')

    train_data_cot = get_icl_list(train_data, cot_emb_history_cuda, cot_emb_target_item_cuda, cot_data)
    print(train_data_cot.head())
    valid_data_cot = get_icl_list(valid_data, cot_emb_history_cuda, cot_emb_target_item_cuda, cot_data)
    test_data_cot = get_icl_list(test_data, cot_emb_history_cuda, cot_emb_target_item_cuda, cot_data)
    train_data_save_path =f"LLaMA-Factory/data/{data_name}/dataset/train_icl.tsv"
    valid_data_save_path = f"LLaMA-Factory/data/{data_name}/dataset/valid_icl.tsv"
    test_data_save_path = f"LLaMA-Factory/data/{data_name}/dataset/test_icl.tsv"

    train_data_cot.to_csv(train_data_save_path, index=False, sep='\t')
    valid_data_cot.to_csv(valid_data_save_path, index=False, sep='\t')
    test_data_cot.to_csv(test_data_save_path, index=False, sep='\t')
    print("percent of value 1", sum(combined_label)/len(combined_label))










    
    





