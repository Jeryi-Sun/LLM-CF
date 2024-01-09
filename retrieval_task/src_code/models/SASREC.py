import numpy as np
import torch
import torch.nn as nn
from .base import BaseModel


## reference code: https://github.com/pmixer/SASRec.pytorch
## reference code: https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/sequential_recommender/sasrec.py


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs




class SASREC(BaseModel):
    '''
        Wang-Cheng Kang, Julian McAuley (2018). 
        Self-Attentive Sequential Recommendation. 
        In Proceedings of IEEE International Conference on Data Mining (ICDM'18)
    '''
    def __init__(self, init_para):
        super(SASREC, self).__init__(init_para)

        config = self.model_config

        # self.item_feat = 
        
        self.pos_emb = torch.nn.Embedding(config['max_len'], config['hidden_dim']) 
        self.emb_dropout = torch.nn.Dropout(p = config['dropout_rate'])

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(config['hidden_dim'], eps=1e-8)

        for _ in range(config['num_blocks']):
            new_attn_layernorm = torch.nn.LayerNorm(config['hidden_dim'], eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(config['hidden_dim'],
                                                            config['num_heads'],
                                                            config['dropout_rate'])
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(config['hidden_dim'], eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(config['hidden_dim'], config['dropout_rate'])
            self.forward_layers.append(new_fwd_layer)

        # self.sigmoid = torch.nn.Sigmoid()

        #self._init_weights()

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

        seqs *= seqs.size(-1) ** 0.5
        positions = np.tile(np.array(range(log_seqs_mask.shape[1])), [log_seqs_mask.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(seqs.device))
        seqs = self.emb_dropout(seqs)

        timeline_mask = log_seqs_mask #torch.where(log_seqs==self.padding_idx, 1, 0).bool()
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=seqs.device))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) 
        
        timeline_mask = torch.sum(~timeline_mask, dim=-1) - 1
        log_feats = self.gather_indexes(log_feats, timeline_mask)
        return log_feats

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    # @torchsnooper.snoop()
    def predict(self, x, y, topK = 50):
        '''
        prediction for testing (full item setting)
        '''
        user_emb = self.forward(x)
        #B, dim

        # item_num, dim
        all_item_emb = self.emb_tab_dict['item_id_emb_tab'](
            torch.arange(1, self.num_items, device=self.device))

        #B, item_num
        logits = torch.matmul(user_emb, torch.t(all_item_emb))
        logit_value, logit_index = torch.topk(logits, k=topK, dim=1)
        logit_index += 1

        return logit_index


    
    def train_step(self, x, y):

        user_emb = self.forward(x)

        target_item_emb = self.emb_tab_dict['item_id_emb_tab'](y)

        loss = self.calulate_loss(user_emb, target_item_emb, y)

        return loss
    
    def forward(self, x):

        _, rec_his_emb, rec_his_mask = self.embed_user_behaviors(x[0], x[1])
        # user_id not used in SASREC

        user_emb = self.log2feats(rec_his_emb, rec_his_mask)

        return user_emb
    

class SASREC_aug(SASREC):
    def __init__(self, init_para):
        super().__init__(init_para)

        self.linear_transform = nn.Sequential(
            nn.Linear(self.model_config['hidden_dim'], #+self.model_config['aug_dense_feat_dim'],
                       self.model_config['hidden_dim']),
            nn.GELU(), 
            nn.Linear(self.model_config['hidden_dim'], self.model_config['hidden_dim'])
        )

        self._init_weights()
        

    def forward(self, x, return_recon_loss=False):

        aug_feat, recon_loss = self.embed_orig_and_aug_data_fileds(x)
        x, aug_x = x
        _, rec_his_emb, rec_his_mask = self.embed_user_behaviors(x[0], x[1])
        # user_id not used in SASREC

        user_emb = self.log2feats(rec_his_emb, rec_his_mask)
        user_emb = user_emb + 0.1*aug_feat
        #user_emb = self.linear_transform(torch.cat([user_emb, aug_feat], dim=-1))
        #user_emb = self.linear_transform(user_emb)


        if return_recon_loss:
            return user_emb, recon_loss
        else:
            return user_emb

    def train_step(self, x, y):

        user_emb, cot_reconstruct_loss = self.forward(x, return_recon_loss=True)

        target_item_emb = self.emb_tab_dict['item_id_emb_tab'](y)

        sampled_softmax_loss = self.calulate_loss(user_emb, target_item_emb, y)

        loss = sampled_softmax_loss + self.model_config['reconstruct_weight']*cot_reconstruct_loss

        return loss
    
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
                _, rec_his_emb, rec_his_mask = self.embed_user_behaviors(input_data[index-1], data)
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

