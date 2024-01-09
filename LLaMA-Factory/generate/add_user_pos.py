import pickle
import json
import pandas as pd
from tqdm import tqdm

def add_user_pos_train():

    with open(train_data_path, 'r') as file:
        train_data = json.load(file)
    ref_data = pd.read_csv(ref_data_path,sep='\t')
    for idx in tqdm(range(len(train_data))):
        train_data[idx]["user_id"] = int(ref_data.loc[idx]['user_id'])
        train_data[idx]["pos_id"] = len(eval(ref_data.loc[idx]['history']))
        train_data[idx]["history"] = ref_data.loc[idx]['history']
        train_data[idx]["item"] = int(ref_data.loc[idx]['item'])
        train_data[idx]["label"] = float(ref_data.loc[idx]['label'])


    print(train_data[0])
    with open(train_user_pos_data_path, 'w') as file:
        json.dump(train_data, file, ensure_ascii=False)

def add_user_pos_train_cot():

    with open(train_user_pos_data_path, 'r') as file:
        train_data = json.load(file)
    with open(cot_data_path, 'rb') as file:
        cot_data = pickle.load(file)

    for cot_item in tqdm(cot_data):
        flag = False
        for train_item in train_data:
            if train_item['instruction'] == cot_item['instruction'] and \
            train_item['input'] == cot_item['input'] and \
            train_item['Output'] == cot_item['Output']:
                cot_item['user_id'] = train_item['user_id']
                cot_item['pos_id'] = train_item['pos_id']
                cot_item["history"] = train_item["history"]
                cot_item["item"] = train_item["item"]
                cot_item["label"] = train_item["label"]
                flag=True
                break
        if not flag:
            print("error miss!")
    print(cot_data[0])
    with open(cot_data_user_pos_path, 'wb') as file:
        pickle.dump(cot_data, file)

for data_name in ["sports","beauty",'toys']:

    cot_data_path = f"LLaMA-Factory/data/{data_name}/ready-to-use/ctr/train_cot_scored_final.pkl"
    cot_data_user_pos_path = f"LLaMA-Factory/data/{data_name}/ready-to-use/ctr/train_cot_user_pos.pkl"
    train_data_path = f"LLaMA-Factory/data/{data_name}/ready-to-use/ctr/train.json"
    train_user_pos_data_path = f"LLaMA-Factory/data/{data_name}/ready-to-use/ctr/train_user_pos.json"
    ref_data_path = f"LLaMA-Factory/data/{data_name}/dataset/training.tsv"
    add_user_pos_train_cot()



