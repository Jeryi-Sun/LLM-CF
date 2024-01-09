import pickle
import copy
import json
cot_data_path = "LLaMA-Factory/data/toys/ready-to-use/ctr/train_cot_scored_final.pkl"
cot_data_save_path = "LLaMA-Factory/data/toys/ready-to-use/ctr/train_cot_tuning.json"
with open(cot_data_path, 'rb') as file:
    cot_data = pickle.load(file)
cot_tuning_data = []
for item in cot_data:
    item_cot = copy.deepcopy(item)
    item_cot_answer = copy.deepcopy(item)
    item_cot["input"] = item["input"] + " Let's think step by step. "
    item_cot["Output"] = item["COT_text"]

    item_cot_answer["input"] = item["input"] + " Let's think step by step. " + item["COT_text"] \
        + " Therefore, the answer (Yes or No) is " 

    cot_tuning_data.append(item_cot)
    cot_tuning_data.append(item_cot_answer)

with open(cot_data_save_path, 'w') as file:
    json.dump(cot_tuning_data, file)


"""
{'instruction': 'Based on the following purchase history of a user, please determine whether the user is likely to purchase the target new item. Answer with "Yes" or "No".', 
 'input': "Target new item: [title: Ardell Lashgrip Strip Adhesive, Clear, 0.25 Ounce; brand: Ardell; category: Beauty, Tools & Accessories, Makeup Brushes & Tools, Eyelash Tools, Fake Eyelashes & Adhesives]. \n The user's purchased items: [title: PanOxyl Bar 10% - 4 oz - 1 Bar; brand: Panoxyl; category: Beauty, Skin Care, Face, Cleansers][title: BH Cosmetics 120 Color Eyeshadow Palette 2nd Edition; brand: BHCosmetics; category: Beauty, Makeup, Eyes, Eye Shadow][title: Neutrogena SkinClearing Mineral Powder, Classic Ivory 10; brand: Neutrogena; category: Beauty, Makeup, Face, Foundation].\n", 
 'Output': 'No', 
 'COT_text': " 1. The user's interaction history and review comments indicate that they have a strong interest in makeup and beauty products.\n2. The target new item, Ardell Lashgrip Strip Adhesive, Clear, 0.25 Ounce, is a product related to the user's interest in makeup and beauty products.\n3. The user's past purchases of BH Cosmetics 120 Color Eyeshadow Palette 2nd Edition and Neutrogena SkinClearing Mineral Powder, Classic Ivory 10 suggest that the user is interested in a variety of makeup products and skincare.\n4. The user's potential desire for diversity in their purchases could be a factor in their decision not to purchase the Ardell Lashgrip Strip Adhesive, Clear, 0.25 Ounce.\n5. Popularity bias and position bias could also influence the user's decision not to purchase the Ardell Lashgrip Strip Adhesive, Clear, 0.25 Ounce.\n6. The user's decision not to purchase the Ard", 
 'cot_scores': 0.6157596627009052} 
"""

"""
input: input + "Let's think step by step."
'Output' = {COT_text}
"""

"""
input: input + "Let's think step by step." + {COT_text} + Therefore, the answer (Yes or No) is 
"""



