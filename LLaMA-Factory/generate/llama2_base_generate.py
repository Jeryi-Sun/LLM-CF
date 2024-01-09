
import argparse
import random
parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
parser.add_argument('--gpu_ids', type=str, required=True, help='such as 0,1,2')
parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset')
parser.add_argument('--output_dataset_path', type=str, required=True, help='Path for the output dataset')
parser.add_argument('--output_dataset_vector', type=str, required=True, help='Path for the output dataset vector')

args = parser.parse_args()

model_path = args.model_path
dataset_path = args.dataset_path
output_dataset_path = args.output_dataset_path
output_dataset_vector = args.output_dataset_vector


import os
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

from transformers import AutoTokenizer
import torch
from transformers import LlamaForCausalLM
import json
from tqdm import tqdm  
import pickle
import numpy as np


tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.padding_side = "left"
model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map='auto'
    )
model.eval()

model.config.pad_token_id = tokenizer.pad_token_id = 0  # </s>
model.config.bos_token_id = 1
model.config.eos_token_id = 2

device = "cuda"


def generate_prompt(Output, input=None):
    if Output =='Yes':
        return """[INST] <<SYS>>
        As an AI model developed for analyzing consumer behavior, your task is to generate a chain of thought that considers the following points:

        1. Utilize the user's interaction history and review comments to summarize their profiles.
        2. Introduce the target new item and detail its features precisely. In addition, integrate information about items related to the current target new item to enhance understanding of its context and potential applications.
        3. Contemplate the alignment between the user's profile and the features of the target item.
        4. Reflect on the user's potential desire for diversity in their purchases.

        Your output should be a clear and logical chain of thought that elucidates the reasoning behind each consideration according to the user's decision-making given below. Ensure your analysis is impartial and free from any prejudicial content. The focus should be on understanding the factors that could influence a user's decision-making process regarding the target item.
        <</SYS>>

        Please generate a chain of thought based on the user's history of their past purchases, considering how these might relate to their interest in the target new item.
        {} 
        User's decision-making: The user purchases the target new item.
        Let's think step by step and develop the chain of thought for the above considerations. Commence with the chain of thought immediately.
        [/INST]""".format(input)
    else:
        return """[INST] <<SYS>>
        As an AI model developed for analyzing consumer behavior, your task is to generate a chain of thought that considers the following points:

        1. Utilize the user's interaction history and review comments to summarize their profiles.
        2. Introduce the target new item and detail its features precisely. In addition, integrate information about items related to the current target new item to enhance understanding of its context and potential applications.
        3. Contemplate the alignment between the user's profile and the features of the target item.
        4. Reflect on the user's potential desire for diversity in their purchases.

        Your output should be a clear and logical chain of thought that elucidates the reasoning behind each consideration according to the user's decision-making given below. Ensure your analysis is impartial and free from any prejudicial content. The focus should be on understanding the factors that could influence a user's decision-making process regarding the target item.
        <</SYS>>

        Please generate a chain of thought based on the user's history of their past purchases, considering how these might relate to their interest in the target new item.
        {} 
        User's decision-making: The user dose not purchase the target new item.
        Let's think step by step and develop the chain of thought for the above considerations. Commence with the chain of thought immediately.
        [/INST]""".format(input)


def generate_and_get_hidden_states(
    dataset_path,
    output_path,
    batch_size=16,  
    temperature=0.6,
    num_beams=1,
    max_new_tokens=256,
    cutoff_len=1024,
    num_return_sequences=1  
):
    with open(dataset_path, 'rb') as file:
        data = pickle.load(file)

    for i in tqdm(range(0, len(data), batch_size), desc="Processing"):
        data_subset = data[i:i+batch_size]

        prompts = [generate_prompt(item["Output"], item["input"]) for item in data_subset]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=cutoff_len).to(device)

        # 生成序列
        generated_sequences = model.generate(
            **inputs,
            num_beams=num_beams,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_return_sequences,
            use_cache=True,
            # penalty_alpha=0.5,
            # repetition_penalty=1.1,
            # min_new_tokens = 10
        )

        # outputs = model(generated_sequences, output_hidden_states=True)
        # hidden_states = outputs.hidden_states
        # mean_pooled = torch.mean(hidden_states[-1], dim=1)

        generated_texts = tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)
        processed_texts = [text.split("[/INST]", 1)[1] if "[/INST]" in text else text for text in generated_texts]
        for j, item in enumerate(data_subset):
            item["Base_COT_text"] = processed_texts[j]
            #item["COT_vector"] = [mean_pooled[k].tolist() for k in range(j*num_return_sequences, (j+1)*num_return_sequences)]
        # cot_vector+=mean_pooled.tolist()

    with open(output_path, 'wb') as file:
        pickle.dump(data, file)


generate_and_get_hidden_states(
    dataset_path,
    output_dataset_path
)