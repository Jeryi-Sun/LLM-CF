import argparse
import pickle
from textblob import TextBlob
from collections import Counter
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "4"

parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset')
parser.add_argument('--output_dataset_path', type=str, required=True, help='Path for the output dataset')
# 解析命令行参数
args = parser.parse_args()

dataset_path = args.dataset_path
output_dataset_path = args.output_dataset_path

def check_coherence(cot):
    blob = TextBlob(cot)
    sentences = blob.sentences

    total_variation = 0
    for i in range(len(sentences) - 1):
        polarity_change = abs(sentences[i].sentiment.polarity - sentences[i+1].sentiment.polarity)
        subjectivity_change = abs(sentences[i].sentiment.subjectivity - sentences[i+1].sentiment.subjectivity)
        total_variation += polarity_change + subjectivity_change
    coherence_score = 1 - min(total_variation / len(sentences), 1)

    words = [word.lower() for word in blob.words]
    word_counts = Counter(words)
    repetitions = sum(count for word, count in word_counts.items() if count > 1)
    repetition_score = 1 - min(repetitions / len(words), 1)

    combined_score = (coherence_score + repetition_score) / 2
    return combined_score


def similarity(cot, Output, user_history, match_model):
    texts = ["Utilize the user's interaction history and review comments to summarize their profile.", 
    "Contemplate the alignment between the user's profile and the features of the target item.",
    "Reflect on the user's potential desire for diversity in their purchases.",
    "Examine any biases such as popularity bias or position bias that may influence the user's preferences or behavior.",
    user_history,
    "The user dose not purchase the target new item" if Output=='No' else "The user purchases the target new item."
    ]
    cots = [cot for _ in texts]

    embeddings_1 = match_model.encode(cots, normalize_embeddings=True)
    embeddings_2 = match_model.encode(texts, normalize_embeddings=True)
    scores = (embeddings_1 @ embeddings_2.T)[0]
    check_completeness = np.mean(scores[:4])
    relevance_score = scores[4]
    consistency_score = scores[5]

    return check_completeness, relevance_score, consistency_score



weights = {
    'coherence': 0.25,
    'completeness': 0.25,
    'relevance': 0.25,
    'consistency': 0.25
}


model = SentenceTransformer('LLMs/match_model/')
with open(dataset_path, 'rb') as file:
    data = pickle.load(file)

cot_scores = {}

for s in tqdm(data):
    cot_scores = []
    for cot in s['COT_text']:
        if len(cot) <10:  
            total_score = 0
        elif "[INST]" in cot:
            total_score = 0
        else:
            coherence_score = check_coherence(cot)

            completeness_score, relevance_score, consistency_score = similarity(cot, s['Output'], s['input'], model)
            # Combine scores with a chosen weighting system
            total_score = coherence_score * weights['coherence'] + \
                        completeness_score * weights['completeness'] + \
                        relevance_score * weights['relevance'] + \
                            consistency_score * weights['consistency']
        
        cot_scores.append(total_score)
    s['cot_scores'] = cot_scores


with open(output_dataset_path, 'wb') as file:
    pickle.dump(data, file)
