import os
os.environ['CUDA_VISIBLE_DEVICES'] = "7"
import json
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
root_path = "LLaMA-Factory/data"
model = SentenceTransformer('LLMs/match_model/')
for data_name in ["toys", "sports", "beauty"]:
    cot_emb = []
    file_path_cot = f'{root_path}/{data_name}/ready-to-use/ctr/train_cot_base.pkl'
    with open(file_path_cot, "rb") as file:
        cot_data = pickle.load(file)
    for item in tqdm(cot_data):
        cot_emb.append(model.encode(item["Base_COT_text"], normalize_embeddings=True))
    save_path = f'{root_path}/{data_name}/ready-to-use/ctr/cot_emb_base.pkl'
    with open(save_path, "wb") as file:
        pickle.dump(cot_emb, file)
    







        