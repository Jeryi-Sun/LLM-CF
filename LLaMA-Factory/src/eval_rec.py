import sys

import fire
import gradio as gr
import numpy as np
import torch
torch.set_num_threads(1)
import transformers
import json
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
# from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from sklearn.metrics import roc_auc_score
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(
    load_8bit: bool = False,
    base_model: str = "/media/disk1/fordata/web_server/sunzhongxiang/LLM_models/llama-2-hf/Llama-2-7b-hf/",
    #lora_weights: str = "tloen/alpaca-lora-7b",
    test_data_path: str = "/home/web_server/sunzhongxiang/codes/RRecAgent/LLaMA-Factory/data/amazon_test.json",
    result_json_data: str = "/home/web_server/sunzhongxiang/codes/RRecAgent/LLaMA-Factory/eval_result/llama-7B-base.json",
    batch_size: int = 32,
    share_gradio: bool = False,
):
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

    #model_type = lora_weights.split('/')[-1]
    #model_name = '_'.join(model_type.split('_')[:2])

    # if model_type.find('book') > -1:
    #     train_sce = 'book'
    # else:
    #     train_sce = 'movie'
    
    # if test_data_path.find('book') > -1:
    #     test_sce = 'book'
    # else:
    #     test_sce = 'movie'
    
    #temp_list = model_type.split('_')
    seed = 1024 #temp_list[-2]
    sample = 1 #temp_list[-1]
    
    # if os.path.exists(result_json_data):
    #     f = open(result_json_data, 'r')
    #     data = json.load(f)
    #     f.close()
    # else:
    #     data = dict()

    # if not data.__contains__(train_sce):
    #     data[train_sce] = {}
    # if not data[train_sce].__contains__(test_sce):
    #     data[train_sce][test_sce] = {}
    # if not data[train_sce][test_sce].__contains__(model_name):
    #     data[train_sce][test_sce][model_name] = {}
    # if not data[train_sce][test_sce][model_name].__contains__(seed):
    #     data[train_sce][test_sce][model_name][seed] = {}
    # if data[train_sce][test_sce][model_name][seed].__contains__(sample):
    #     exit(0)
        # data[train_sce][test_sce][model_name][seed][sample] = 


    tokenizer = LlamaTokenizer.from_pretrained(base_model,)
    if "cuda" in device:
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map='auto'
        )
    # elif device == "mps":
    #     model = LlamaForCausalLM.from_pretrained(
    #         base_model,
    #         device_map={"": device},
    #         torch_dtype=torch.float16,
    #     )
    #     model = PeftModel.from_pretrained(
    #         model,
    #         lora_weights,
    #         device_map={"": device},
    #         torch_dtype=torch.float16,
    #     )
    # else:
    #     model = LlamaForCausalLM.from_pretrained(
    #         base_model, device_map={"": device}, low_cpu_mem_usage=True
    #     )
    #     model = PeftModel.from_pretrained(
    #         model,
    #         lora_weights,
    #         device_map={"": device},
    #     )


    tokenizer.padding_side = "left"
    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # </s>
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        instructions,
        inputs=None,
        temperature=0,
        top_p=1.0,
        top_k=40,
        num_beams=1,
        max_new_tokens=1,
        cutoff_len=1024,
        batch_size=1,
        **kwargs,
    ):
        prompt = [generate_prompt(instruction, input) for instruction, input in zip(instructions, inputs)]
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=cutoff_len).to(device)
        

        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                **inputs,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
                # batch_size=batch_size,
            )
        s = generation_output.sequences
        scores = generation_output.scores[0].softmax(dim=-1)
        selected_scores = scores[:, [tokenizer.encode(' Yes', add_special_tokens=False)[-1], tokenizer.encode(' No', add_special_tokens=False)[-1]]].clone().detach().to(torch.float32)
        logits = selected_scores.softmax(dim=-1).cpu().numpy()
        input_ids = inputs["input_ids"].to(device)
        L = input_ids.shape[1]
        s = generation_output.sequences
        output = tokenizer.batch_decode(s, skip_special_tokens=True)
        output = [_.split('Response:\n')[-1] for _ in output]
        
        return output, logits.tolist()
        
    # testing code for readme
    outputs = []
    logits = []
    from tqdm import tqdm
    gold = []
    pred = []

    # with open(test_data_path, 'r') as f:
    #     test_data = json.load(f)
    #     instructions = [_['instruction'][0] for _ in test_data]
    #     inputs = [_['input'] for _ in test_data]
    #     gold = [int(_['Output'] == 'Yes') for _ in test_data]
    #     def batch(list, batch_size=64):
    #         chunk_size = (len(list) - 1) // batch_size + 1
    #         for i in range(chunk_size):
    #             yield list[batch_size * i: batch_size * (i + 1)]
    #     for i, batch in tqdm(enumerate(zip(batch(instructions), batch(inputs)))):
    #         instructions, inputs = batch
    #         output, logit = evaluate(instructions, inputs)
    #         outputs = outputs + output
    #         logits = logits + logit
    #     for i, test in tqdm(enumerate(test_data)):
    #         test_data[i]['predict'] = outputs[i]
    #         test_data[i]['logits'] = logits[i]
    #         pred.append(logits[i][0])

    # from sklearn.metrics import roc_auc_score

    # result = roc_auc_score(gold, pred)
    # f = open(result_json_data, 'w')
    # json.dump({"AUC":result}, f, indent=4)
    # print("AUC: ", result)
    # f.close()
    import json
    from tqdm import tqdm
    from sklearn.metrics import roc_auc_score, log_loss

    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
        instructions = [_['instruction'][0] for _ in test_data]
        inputs = [_['input'] for _ in test_data]
        gold = [int(_['Output'] == 'Yes') for _ in test_data]

        def batch(list, batch_size=64):
            chunk_size = (len(list) - 1) // batch_size + 1
            for i in range(chunk_size):
                yield list[batch_size * i: batch_size * (i + 1)]

        outputs = []
        logits = []
        for i, batch in tqdm(enumerate(zip(batch(instructions), batch(inputs)))):
            instructions, inputs = batch
            output, logit = evaluate(instructions, inputs)  # Assuming evaluate() is defined elsewhere
            outputs = outputs + output
            logits = logits + logit
        for i, test in tqdm(enumerate(test_data)):
            test_data[i]['predict'] = outputs[i]
            test_data[i]['logits'] = logits[i]
            pred.append(logits[i][0])

    result_auc = roc_auc_score(gold, pred)
    result_logloss = log_loss(gold, pred)

    with open(result_json_data, 'w') as f:
        results = {
            "AUC": result_auc,
            "LogLoss": result_logloss
        }
        json.dump(results, f, indent=4)
        print("AUC: ", result_auc)
        print("LogLoss: ", result_logloss)


def generate_prompt(instruction, input=None):
   return """[INST] <<SYS>>
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
    <</SYS>>

    {}
    {} [/INST]""".format(instruction, input)


if __name__ == "__main__":
    fire.Fire(main)