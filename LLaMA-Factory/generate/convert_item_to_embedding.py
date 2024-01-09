import os
os.environ['CUDA_VISIBLE_DEVICES'] = "5"
import json
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
root_path = "LLaMA-Factory/data/"

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)
    
def convert_list_into_str(ls):
    ret = ''
    if len(ls) == 0:
        return ret
    if len(ls)>1:
        for name in ls[:-1]:
            ret += f'{name}, '
    ret += ls[-1]
    return ret

def transform_item_text(itemID):
    item_feat = items2attributes[itemID]
    feats = ['title','brand','category']
    item_text = f'title: {item_feat[feats[0]]}; '+ \
                f'brand: {convert_list_into_str(item_feat[feats[1]])}; '+ \
                f'category: {convert_list_into_str(item_feat[feats[2]])}'
    return item_text

model = SentenceTransformer('LLMs/match_model/')
for data_name in ["toys", "sports", "beauty"]:
    items2text_emb = {}
    items2attributes = load_json(f'{root_path}/{data_name}/new_item_attributes.json')
    items_ids = list(items2attributes.keys())
    for id in tqdm(items_ids):
        item_text = transform_item_text(id)
        item_text_emb = model.encode(item_text, normalize_embeddings=True)
        items2text_emb[id] = item_text_emb.tolist()

    save_path = f"{root_path}/{data_name}/items2text_emb.pkl"

    # Save the dictionary as a pickle file
    """
    with open(save_path, 'rb') as file:
        data_loaded = pickle.load(file)
    """
    with open(save_path, "wb") as file:
        pickle.dump(items2text_emb, file)








        