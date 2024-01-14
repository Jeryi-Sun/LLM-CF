
# LLM-CF
# This is the implementation for the paper "Large Language Model Enhanced Collaborative Filtering".

# Overview
Recent advancements in Large Language Models (LLMs) have attracted considerable interest among researchers to leverage these models to enhance Recommender Systems (RSs).  Existing work predominantly utilizes LLMs to generate knowledge-rich texts or to utilize LLM-derived embeddings as features to improve RSs. Although the extensive world knowledge embedded in LLMs generally benefits RSs, the application is still limited due to the constraints in leveraging a large number of users/items in a single prompt. Considering its crucial role in RSs, one key challenge in enhancing RSs with LLMs lies in providing better collaborative filtering information through LLMs. In this paper, drawing inspiration from the in-context learning and chain of thought reasoning in LLMs, we propose the **W**orld knowledge and **R**easoning guided **C**ollaborative **F**iltering (**LLM-CF**) framework, which distils the world knowledge and reasoning capabilities of LLMs into collaborative filtering through an in-context, chain of thought methodology. We also explored a concise and efficient instruction-tuning method, which improves the recommendation capabilities of LLMs while preserving their general functionalities. Our framework is model-agnostic and efficient for deployment. Comprehensive experiments on three real-world datasets demonstrate that LLM-CF significantly enhances several backbone recommendation models and consistently outperforms competitive baselines, showcasing its effectiveness in distilling the world knowledge and reasoning capabilities of LLM into collaborative filtering.

# Trainining and Evaluation

## Train RecGen-LLaMA
The training process of RecGen-LLaMA is based on the open-sourced [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory/tree/main).

The LLaMA-2-7b-chat can be downloaded from [huggingface](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf).


```python
cd LLaMA-Factory/
bash run_sfts.sh
```

## CoT Reasoning Generation and In-context CoT Examples Retrieval

The matching model can be downloaded from [huggingface](https://huggingface.co/BAAI/bge-large-en-v1.5).

cd LLaMA-Factory/generate/

```python
bash run_gen_beauty.sh run_gen_sports.sh  run_gen_toys.sh 
bash run_score_beauty.sh run_score_sports.sh run_score_toys.sh
python analysis_score.py
python add_user_pos.py
python retrieval_ICL.py
python retrieval_ICL_retrieval.py
python convert_cot_embedding.py
```


## ranking task

```python
cd ranking_task/src_code
bash run_beauty.sh run_sports.sh  run_toys.sh 
```

## retrieval task

```python
cd retrieval_task/src_code
bash run_beauty.sh run_sports.sh  run_toys.sh 
```

The parameters used in above code are shown in their own files as default parameters.
