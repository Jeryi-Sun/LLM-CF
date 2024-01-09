import numpy as np
import torch
import torch.nn as nn
import torchsnooper
from .base import BaseModel


class YoutubeDNN(BaseModel):
    def __init__(self, init_para):
        super(YoutubeDNN, self).__init__(init_para)

        self.dense = nn.Sequential(
            nn.Linear(self.model_config['item_id_dim']+self.model_config['user_id_dim'],
                       self.model_config['item_id_dim']),
            nn.ReLU(),
            nn.Linear(self.model_config['item_id_dim'], self.model_config['item_id_dim']),
            nn.ReLU(),
            nn.Linear(self.model_config['item_id_dim'], self.model_config['item_id_dim'])
        )
        
        self._init_weights()

    def _init_weights(self):
        # weight initialization xavier_normal (or glorot_normal in keras, tf)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.Parameter):
                nn.init.xavier_normal_(m)

    # @torchsnooper.snoop()
    def log2feats(self, seqs, log_seqs_mask):

        seq_len = torch.sum(~log_seqs_mask, dim=1) # B

        seq_sum = torch.sum(seqs*(~log_seqs_mask[...,None]), dim=1) #[B, dim]

        his_emb = seq_sum / seq_len[...,None]

        return his_emb
    

    # @torchsnooper.snoop()
    def predict(self, x, y, topK = 50):
        '''
        prediction for testing (full item setting)
        '''

        user_id_emb, rec_his_emb, rec_his_mask = self.embed_user_behaviors(x[0], x[1])
        
        his_emb = self.log2feats(rec_his_emb, rec_his_mask)

        user_emb = self.dense(torch.cat([user_id_emb, his_emb], dim=-1))

        # item_num, dim
        all_item_emb = self.emb_tab_dict['item_id_emb_tab'](
            torch.arange(1, self.num_items, device=self.device))

        #B, item_num
        logits =  torch.matmul(user_emb, torch.t(all_item_emb))
        logit_value, logit_index = torch.topk(logits, k=topK, dim=1)
        logit_index += 1

        return logit_index

    # @torchsnooper.snoop()
    def forward(self, x):

        user_id_emb, rec_his_emb, rec_his_mask = self.embed_user_behaviors(x[0], x[1])
        # user_id not used in SASREC

        his_emb = self.log2feats(rec_his_emb, rec_his_mask)

        user_emb = self.dense(torch.cat([user_id_emb, his_emb], dim=-1))

        return user_emb
    

    def train_step(self, x, y):

        user_emb = self.forward(x)

        target_item_emb = self.emb_tab_dict['item_id_emb_tab'](y)

        loss = self.calulate_loss(user_emb, target_item_emb, y)

        return loss
        
class YoutubeDNN_aug(YoutubeDNN):
    def __init__(self, init_para):
        super().__init__(init_para)

        self.intergrate_user_emb = nn.Sequential(
            nn.Linear(self.model_config['embedding_size']+self.model_config['aug_dense_feat_dim'],
                       self.model_config['embedding_size']),
            nn.GELU(), 
            nn.Linear(self.model_config['embedding_size'], self.model_config['embedding_size'])
        )

        self._init_weights()

    def get_user_emb(self, x, return_recon_loss=False):

        aug_feat, recon_loss = self.embed_orig_and_aug_data_fileds(x)
        x, aug_x = x
        _, rec_his_emb, rec_his_mask = self.embed_user_behaviors(x[0], x[1])
        rec_his_len = (~rec_his_mask).sum(-1)

        user_id_emb, rec_his_emb, rec_his_mask = self.embed_user_behaviors(x[0], x[1])
        # user_id not used in SASREC

        his_emb = self.log2feats(rec_his_emb, rec_his_mask)

        user_emb = self.dense(torch.cat([user_id_emb, his_emb], dim=-1))

        user_emb = self.intergrate_user_emb(torch.cat([user_emb, aug_feat], dim=-1))

        if return_recon_loss:
            return user_emb, recon_loss
        else:
            return user_emb

    def train_step(self, x, y):

        user_emb, cot_reconstruct_loss = self.get_user_emb(x, return_recon_loss=True)

        target_item_emb = self.emb_tab_dict['item_id_emb_tab'](y)

        sampled_softmax_loss = self.calulate_loss(user_emb, target_item_emb, y)

        loss = sampled_softmax_loss + self.model_config['reconstruct_weight']*cot_reconstruct_loss

        return loss
    
    def predict(self, x, y, topK = 50):
        '''
        prediction for testing (full item setting)
        '''

        user_emb = self.get_user_emb(x)

        # item_num, dim
        all_item_emb = self.emb_tab_dict['item_id_emb_tab'](
            torch.arange(1, self.num_items, device=self.device))

        #B, item_num
        logits =  torch.matmul(user_emb, torch.t(all_item_emb))
        logit_value, logit_index = torch.topk(logits, k=topK, dim=1)
        logit_index += 1

        return logit_index
    
    def embed_input_fields(self, input_data, input_data_name_ls=None, use_item_textual_data=False):

        emb_data_ls = []
        
        input_data_name_ls = self.input_data_ls if input_data_name_ls is None else input_data_name_ls
        def mean_pooling(tensor_data, id_data, dim):
            mask = id_data != 0
            tensor_data = torch.sum(tensor_data*mask.unsqueeze(-1), dim=dim)
            tensor_data = tensor_data / (torch.sum(mask, dim=dim, keepdim=True)+1e-9) # 1e-9 to avoid dividing zero

            return tensor_data
        
        for index, (data_filed, data) in enumerate(zip(input_data_name_ls, input_data)):
            if data_filed.name.startswith('history_item_id'):
                user_id_emb, rec_his_emb, rec_his_mask = self.embed_user_behaviors(input_data[index-1], data)
                # user_id not used in SASREC
                history_data = self.log2feats(rec_his_emb, rec_his_mask)
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