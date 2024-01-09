import copy
import torch
import torch.nn as nn

import os, logging
from tqdm import tqdm   
import numpy as np

import torchsnooper

from .transformer_decoder import ModelArgs, Transformer

class GPTDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, batch_first=True):
        super(GPTDecoderLayer, self).__init__()
        self.batch_first = batch_first
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=batch_first)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, src):
        if not self.batch_first:
            src = src.transpose(0, 1)  # Convert to (sequence_length, batch_size, feature_number)

        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(nn.functional.relu(self.linear1(src2))))
        src = src + self.dropout2(src2)

        if not self.batch_first:
            src = src.transpose(0, 1)  # Convert back to (batch_size, sequence_length, feature_number)

        return src


class BaseModel(nn.Module):
    def __init__(self, init_para) -> None:
        super().__init__()

        self.wksp = init_para['workspace']
        self.name = init_para['name']
        self.device = init_para['device']
        self.patience = init_para['patience'] # patience for early stop
        # self.loss_func = init_para['loss_function']
        # self.eval_metrics = init_para['eval_metric_ls']
        # self.metric_names = init_para['metric_name_ls']
        self.model_config = init_para['model_config']
        self.input_data_ls = init_para['input_data']
        self.num_neg = init_para['num_neg']
        self.register_embedding_tables()
        self.set_device()


        '''if use augmentation data'''
        if 'aug' in self.name:
            self.aug_input_data_ls = init_para['input_aug_data']
            self.num_aug_samples = init_para['num_aug']
            from .DeepFM import MLPLayers
            self.bge_emb_mapping = MLPLayers(
                self.model_config['cot_mapping_mlp_hidden_size'],
                last_activation=False
            )
            self.cot_emb_reconstruct_mapping = MLPLayers(
                self.model_config['cot_mapping_mlp_hidden_size'][::-1],
                last_activation=False
            )
            params = ModelArgs()
            params.dim = self.model_config['decoder_emb_dim']
            params.n_heads = self.model_config['decoder_nhead']
            params.dim_feedforward = 2*self.model_config['decoder_emb_dim']
            self.context_transformer = Transformer(params=params)
            self.history_mlp = MLPLayers(
                self.model_config['ICL_history_mlp'],
                last_activation=False
            )


    def register_embedding_tables(self):
        self.emb_tab_dict = nn.ModuleDict()
        for feat in self.input_data_ls:
            if feat.name.startswith('history'):
                continue
            self.emb_tab_dict[f'{feat.name}_emb_tab'] = \
                    nn.Embedding(num_embeddings=feat.vocabulary_size, embedding_dim=feat.embedding_dim, padding_idx=0)
            # print(f'{feat.name}_emb_tab')
            if feat.name == 'item_id':
                self.num_items = feat.vocabulary_size
    
    
    def embed_user_behaviors(self, user_id, history):

        history_mask = None
        history_emb, user_emb = None, None

        history_mask = history == 0.0
        history_emb = self.emb_tab_dict[f'item_id_emb_tab'](history)
        user_emb = self.emb_tab_dict[f'user_id_emb_tab'](user_id)
            
        return user_emb, history_emb, history_mask

    def embed_input_fields(self, input_data, input_data_name_ls=None, use_item_textual_data=False):
        emb_data_ls = []

        def mean_pooling(tensor_data, id_data, dim):
            mask = id_data != 0
            tensor_data = torch.sum(tensor_data*mask.unsqueeze(-1), dim=dim)
            tensor_data = tensor_data / (torch.sum(mask, dim=dim, keepdim=True)+1e-9) # 1e-9 to avoid dividing zero

            return tensor_data
        
        input_data_name_ls = self.input_data_ls if input_data_name_ls is None else input_data_name_ls

        for data_filed, data in zip(input_data_name_ls, input_data):
            if data_filed.name.startswith('history'):
                history_data = self.emb_tab_dict[f'{data_filed.name[len("history")+1:]}_emb_tab'](data)
                # mean pooling for history behaviors
                if len(data.size()) == 2:
                    history_data = mean_pooling(history_data, data, 1)
                elif len(data.size()) == 3:
                    history_data = mean_pooling(history_data.reshape(history_data.size(0), -1, history_data.size(-1)), 
                                                    data.reshape(data.size(0), -1), 1)
                else:
                    raise ValueError(f'wrong dimension of input data {data_filed.name} with size {data.size()}')
                emb_data_ls.append(history_data)
                if data_filed.name == 'history_item_id' and use_item_textual_data:
                    history_data = self.emb_tab_dict[f'item_text_id_emb_tab'](data)
                    history_data = mean_pooling(history_data, data, 1)
                    emb_data_ls.append(history_data)
            else:
                emb_data_ls.append(
                    self.emb_tab_dict[f'{data_filed.name}_emb_tab'](data)
                ) 
                if data_filed.name == 'item_id' and use_item_textual_data:
                    emb_data_ls.append(
                        self.emb_tab_dict[f'item_text_id_emb_tab'](data)
                    ) 

        return emb_data_ls

   

    def set_emb_tab_for_aug_data(self, cot_emb: torch.tensor, item_text_emb_dict: dict):
        
        '''BGE embeddings of item textual feature'''
        item_text_emb = torch.zeros((len(item_text_emb_dict)+1, 768)) # zero for padding
        for item_id, text_emb in item_text_emb_dict.items():
            item_text_emb[int(item_id)] = torch.tensor(text_emb, dtype=torch.float32)
        
        self.emb_tab_dict['item_text_id_emb_tab'] = nn.Embedding.from_pretrained(
            embeddings=item_text_emb, freeze=True
        ).to(self.device)

        '''BGE embeddings of COT(chain of thoughts)'''
        cot_emb = torch.nn.functional.normalize(cot_emb, p=2, dim=1)

        self.emb_tab_dict['cot_id_emb_tab'] = nn.Embedding.from_pretrained(
            embeddings=cot_emb, freeze=True
        ).to(self.device)

        self.emb_tab_dict['label_id_emb_tab'] = nn.Embedding(
            2, self.model_config['aug_label_id_dim']
        ).to(self.device)
      
    def embed_orig_and_aug_data_fileds(self, x):
        '''embed both orignal and augmentation data input'''
        orig_data_input, aug_data_input = x

        # orig_user_emb, orig_his_emb, orig_his_mask = self.embed_user_behaviors(orig_data_input[0], orig_data_input[1])
        aug_data_emb = [
            self.embed_input_fields(aug_data, self.aug_input_data_ls, True) 
            for index, aug_data in enumerate(aug_data_input) if index < self.num_aug_samples
        ]
        '''
        each augmented data sample in aug_data_emb:
            * user_id_emb
            * history_item_id_emb
            * history_item_texutual_id_emb: if use textual embedding.  768 dim BGE embedding
            * item_id_emb
            * item_textual_emb : if use textual embedding.  768 dim BGE embedding
            * cot_emb: 768 dim BGE embedding
            * label_emb
        '''
        # set target item as the padding item to avoid data leakage
        dummy_item_id = torch.zeros_like(orig_data_input[2], dtype=torch.long) 
        orig_data_input = (orig_data_input[0], orig_data_input[1], dummy_item_id)

        orig_data_w_text = self.embed_input_fields(orig_data_input, use_item_textual_data=True)
        orig_data_w_text[2] = self.bge_emb_mapping(orig_data_w_text[2])
        orig_data_w_text[4] = self.bge_emb_mapping(orig_data_w_text[4])

        # 这边加一个 MLP 去映射到我们的大小上
        cot_data_emb_raw = [data[-2] for data in aug_data_emb]
        for data in aug_data_emb:
            data[-2] = self.bge_emb_mapping(data[-2]) # mapping cot emb
            data[4] = self.bge_emb_mapping(data[4]) # mapping item_textual emb
            data[2] = self.bge_emb_mapping(data[2]) # mapping history_item textual emb

        final_aug_data_emb = [ [self.history_mlp(torch.cat(data[:-2], dim=-1))] + data[-2:] for data in aug_data_emb]

        

        ICL_data_emb = torch.cat(
            [
               torch.stack(data, dim=1) for data in final_aug_data_emb
            ], dim=1
        )
        # if dont use textual data
        # ICL_data_emb = torch.cat(
        #     [ICL_data_emb, self.history_mlp(torch.cat(orig_data_emb, dim=1)).unsqueeze(1)],
        #     dim=1)
        # else if use textual data
        ICL_data_emb = torch.cat(
            [ICL_data_emb, self.history_mlp(torch.cat(orig_data_w_text, dim=1)).unsqueeze(1)],
            dim=1)
             
        ICL_hidden_states = self.context_transformer(ICL_data_emb)

        # 增加对于 cot 的学习能力
        positions = [0, 3, 6, 9]
        cot_outputs = self.cot_emb_reconstruct_mapping(ICL_hidden_states[:, positions, :])
        cot_inputs = torch.stack(cot_data_emb_raw, dim=1)

        # Calculate cosine similarity
        cosine_sim = nn.CosineSimilarity(dim=-1)
        similarity = cosine_sim(cot_outputs, cot_inputs)

        # Calculate loss as 1 - mean cosine similarity
        cot_reconstruct_loss = 1 - similarity.mean()

        ICL_feat = ICL_hidden_states[:, -1, :]

        return ICL_feat, cot_reconstruct_loss

    # def embed_orig_and_aug_data_wocot_fileds(self, x):
    #     '''embed both orignal and augmentation data input'''
    #     orig_data_input, aug_data_input = x

    #     orig_data_emb = self.embed_input_fields(orig_data_input)
    #     aug_data_emb = [
    #         self.embed_input_fields(aug_data, self.aug_input_data_ls) 
    #         for index, aug_data in enumerate(aug_data_input) if index < self.num_aug_samples
    #     ]
    #     # 这边加一个 MLP 去映射到我们的大小上
    #     aug_data_emb_wocot = []
    #     for data in aug_data_emb:
    #         aug_data_emb_wocot.append(data[:-2]+data[-1:])

    #     final_aug_data_emb = [ [self.history_mlp(torch.cat(data[:-1], dim=-1))] + data[-1:] for data in aug_data_emb_wocot]

        

    #     ICL_data_emb = torch.cat(
    #         [
    #            torch.stack(data, dim=1) for data in final_aug_data_emb
    #         ], dim=1
    #     )
    #     ICL_data_emb = torch.cat(
    #         [ICL_data_emb, self.history_mlp(torch.cat(orig_data_emb, dim=1)).unsqueeze(1)],
    #         dim=1)
             
    #     ICL_hidden_states = self.context_transformer(ICL_data_emb)
    #     #ICL_feat = torch.mean(ICL_hidden_states, dim=1)
    #     ICL_feat = ICL_hidden_states[:, -1, :]

    #     return orig_data_emb, ICL_feat

    def set_device(self):
        if self.device == 'cpu':
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(self.device)

    def save_ckpt(self):

        model_path = os.path.join(self.wksp, 'best.pth')
        torch.save(self.state_dict(), model_path)

    def load_ckpt(self, assigned_path=None):

        ckpt_path = self.wksp
        model_path = None
        if assigned_path is not None:
            '''specific assigned path'''
            model_path = assigned_path
        else:
            '''default path'''   
            model_path = os.path.join(ckpt_path, 'best.pth')
        self.load_state_dict(torch.load(model_path, map_location=self.device))


    def fit(self, epochs, training_dataloader, validation_dataloader, optimizer):

        self.optimizer = optimizer

        best_metric, num_stop_increasing_epochs = 0, 0
        
        for epoch in range(epochs):

            self.train_one_epoch(epoch, training_dataloader)
            watch_metric_value = self.test(epoch, validation_dataloader)
            if watch_metric_value > best_metric:
                self.save_ckpt()
                logging.info('save ckpt at epoch {}'.format(epoch))
                print('save ckpt at epoch {}'.format(epoch))
                best_metric = watch_metric_value
                num_stop_increasing_epochs = 0
            else:
                num_stop_increasing_epochs += 1
                if num_stop_increasing_epochs >= self.patience:
                    logging.info('early stop at epoch {}'.format(epoch))
                    print('early stop at epoch {}'.format(epoch))
                    break

    def train_one_epoch(self, epoch, training_dataloader):
        
        self.train()
        tqdm_ = tqdm(iterable=training_dataloader, mininterval=1, ncols=100)

        epoch_loss = 0.0

        for step, sample in enumerate(tqdm_):
            
            self.optimizer.zero_grad()

            x, y = sample['x'], sample['y']
            x = tuple(data.to(self.device) for data in x)
            y = y.to(self.device)

            if 'aug_x' in sample:
                aug_x = sample['aug_x']
                aug_x = [tuple(data.to(self.device) for data in i_aug_x) for i_aug_x in aug_x]


            if 'aug' in self.name:
                '''for retrieval models with augmented data'''
                loss = self.train_step((x, aug_x), y)
            else:
                '''for retrieval models'''
                loss = self.train_step(x, y)

            if torch.isnan(loss):
                raise ValueError('loss is NaN!')

            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            
            if step % (training_dataloader.__len__() // 50) == 0 and step != 0:
                tqdm_.set_description(
                        "epoch {:d} , step {:d} , loss: {:.4f}".format(epoch, step+1, epoch_loss / (step+1)))
                
        logging.info('epoch {}:  loss = {}'.format(epoch, epoch_loss / (step+1)))

    def train_step(self, orig_data, aug_data=None):

        raise NotImplementedError
    
    def sample_negative_per_batch(self):
        numbers = np.arange(1, self.num_items)

        unique_IDs = np.random.choice(numbers, size=self.num_neg, replace=False)
        unique_IDs = torch.tensor(unique_IDs, dtype=torch.long, device=self.device)

        return unique_IDs
    
    # @torchsnooper.snoop()
    def calulate_loss(self, 
                      user_embedding: torch.tensor, 
                      target_item_emb: torch.tensor, 
                      target_item_ID: torch.tensor):
        '''
        user_embedding: (batch, dim)
        target_item_emb: (batch, dim)
        target_item_ID: (batch)
        '''

        # (number of neg)
        negative_items_IDs = self.sample_negative_per_batch()
        # (num of neg, dim)
        neg_item_emb = self.emb_tab_dict['item_id_emb_tab'](negative_items_IDs)
        mask = negative_items_IDs.unsqueeze(dim=0) != target_item_ID[:, None] # True denotes valid negative samples 

        target_embedding = torch.sum(target_item_emb * user_embedding, dim=1, keepdim=True)
        product = torch.einsum('bd,nd->bn', [user_embedding, neg_item_emb])
        loss = torch.exp(target_embedding ) / (
                    torch.sum(mask * torch.exp(product ), dim=1, keepdim=True) + torch.exp(target_embedding))
        loss = torch.mean(-torch.log(loss))

        return loss

    @torch.no_grad()
    def test(self, epoch, test_dataloader):
        self.eval()

        pred_items, ground_truth_items = [], []

        for batch_data in tqdm(iterable=test_dataloader, mininterval=1, ncols=100):
            
            x, y = batch_data['x'], batch_data['y']
            
            x = tuple(data.to(self.device) for data in x)
            y = y.to(self.device)

            if 'aug_x' in batch_data:
                aug_x = batch_data['aug_x']
                aug_x = [tuple(data.to(self.device) for data in i_aug_x) for i_aug_x in aug_x]

            if 'aug' in self.name:
                '''for retrieval models with augmented data'''
                step_pred_items = self.predict((x,aug_x), y) #B
                # if isinstance(step_pred_logits, tuple):
                #     step_pred_logits = step_pred_logits[0]
            else:
                '''for retrieval models'''
                step_pred_items = self.predict(x, y) #B

            pred_items.extend(step_pred_items.detach().cpu().tolist())
            ground_truth_items.extend(y.detach().cpu().tolist())


        metrics = calculate_metrics(pred_items, ground_truth_items, position_ls=[1,5,10])

        logging.info(f'results at epoch {epoch}')
        print(f'results at epoch {epoch}')
        for name, metric in metrics.items():
            logging.info(f"     {name} : {metric}")
            print(f"      {name} : {metric}")

        if epoch == 'test':
            table = f'''\n| hit@5 | hit@10 | ndcg@10 | ndcg@10 |
|-----|-----|-----|-----|
| {metrics['hit@5']} | {metrics['hit@10']} | {metrics['ndcg@5']} | {metrics['ndcg@10']} |'''
            logging.info(table)

        return metrics['hit@10']
            

def calculate_metrics(pre, ground_truth, position_ls = [1, 5, 10]):

    def cal_metric_at_position_k(k=1):

        hit_k, recall_k, NDCG_k = (0, 0, 0)
        epsilon = 0.1 ** 10
        for i in range(len(ground_truth)):
            one_DCG_k, one_recall_k, IDCG_k  =  0, 0, 0
            top_k_item = pre[i][0:k]
            positive_item = set([ground_truth[i]]) # only one postive item

            for pos, iid in enumerate(top_k_item): 
                if iid in positive_item:
                    one_recall_k += 1
                    one_DCG_k += 1 / np.log2(pos + 2)

            '''caculate according to the formal defination'''
            for pos in range(1): # only one postive item in our experiments
                IDCG_k += 1 / np.log2(pos + 2)

            NDCG_k += one_DCG_k / max(IDCG_k, epsilon) # avoid dividing zero
            top_k_item = set(top_k_item)
            if len(top_k_item & positive_item) > 0:
                hit_k += 1
            recall_k += len(top_k_item & positive_item) / max(len(positive_item), epsilon)
    
        hit_k, recall_k, NDCG_k = hit_k / len(ground_truth), recall_k / len(ground_truth), NDCG_k / len(ground_truth)

        return hit_k, recall_k, NDCG_k
    
    results = {}
    
    for pos_k in position_ls:
        hit_k, recall_k, ndcg_k = cal_metric_at_position_k(pos_k)
        results[f'hit@{pos_k}'] = round(hit_k, 4)
        # recall is the same as hit under this test settign
        # results[f'recall@{pos_k}'] = round(recall_k, 4)
        results[f'ndcg@{pos_k}'] = round(ndcg_k, 4)

    return results

