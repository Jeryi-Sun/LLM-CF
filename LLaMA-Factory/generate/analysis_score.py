import pickle
import numpy as np


def analysis_score(original_data_path, final_data_path):
# 加载原始数据
    with open(original_data_path, 'rb') as file:
        data = pickle.load(file)

    # 处理数据
    filtered_data = []
    for item in data:
        if 'cot_scores' in item and 'COT_text' in item and isinstance(item['cot_scores'], list):
            max_score_index = np.argmax(item['cot_scores'])
            if item['cot_scores'][max_score_index] > 0:
                # 只保留最大cot_scores对应的COT_text
                item['cot_scores'] = item['cot_scores'][max_score_index]
                item['COT_text'] =  item['COT_text'][max_score_index]
                filtered_data.append(item)
            else:
                item['cot_scores'] = item['cot_scores']
                item['COT_text'] =  item['COT_text']
                filtered_data.append(item)

    print("original_data len", len(data))
    print("filtered_data len", len(filtered_data))
    # 保存过滤后的数据
    with open(final_data_path, 'wb') as file:
        pickle.dump(filtered_data, file)

# sports

original_data_path = "LLaMA-Factory/data/sports/ready-to-use/ctr/train_cot_scored.pkl" 
final_data_path = "LLaMA-Factory/data/sports/ready-to-use/ctr/train_cot_scored_final.pkl"

analysis_score(original_data_path, final_data_path)


# beauty
original_data_path = "LLaMA-Factory/data/beauty/ready-to-use/ctr/train_cot_scored.pkl" 
final_data_path = "LLaMA-Factory/data/beauty/ready-to-use/ctr/train_cot_scored_final.pkl"

analysis_score(original_data_path, final_data_path)

# toys
original_data_path = "LLaMA-Factory/data/toys/ready-to-use/ctr/train_cot_scored.pkl" 
final_data_path = "LLaMA-Factory/data/toys/ready-to-use/ctr/train_cot_scored_final.pkl"

analysis_score(original_data_path, final_data_path)