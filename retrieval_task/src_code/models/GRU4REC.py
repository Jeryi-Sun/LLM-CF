
r"""
GRU4Rec
################################################

Reference:
    Yong Kiam Tan et al. "Improved Recurrent Neural Networks for Session-based Recommendations." in DLRS 2016.

"""
import numpy as np
import torch
from torch import nn
from torch.nn.init import xavier_uniform_, xavier_normal_
from .base import BaseModel

class GRU4REC(BaseModel):
    r"""GRU4Rec is a model that incorporate RNN for recommendation.

    Note:

        Regarding the innovation of this article,we can only achieve the data augmentation mentioned
        in the paper and directly output the embedding of the item,
        in order that the generation method we used is common to other sequential models.
    """

    def __init__(self, init_para):
        super().__init__(init_para)

        # load parameters info
        self.hidden_size = self.model_config["hidden_size"]
        self.num_layers = self.model_config["num_layers"]
        self.dropout_prob = self.model_config["dropout_prob"]

        # define layers and loss
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.gru_layers = nn.GRU(
            input_size=self.model_config['item_id_dim'],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        self.dense = nn.Linear(self.hidden_size, self.model_config['item_id_dim'])

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)
    

    def his2feat(self, item_seq, item_seq_len):
        item_seq_emb = item_seq
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        gru_output = self.dense(gru_output)
        # the embedding of the predicted item, shape of (batch_size, embedding_size)
        seq_output = self.gather_indexes(gru_output, item_seq_len - 1)
        return seq_output #B, dim

    # @torchsnooper.snoop()
    def predict(self, x, y, topK = 50):
        '''
        prediction for testing (full item setting)
        '''

        _, rec_his_emb, rec_his_mask = self.embed_user_behaviors(x[0], x[1])
        rec_his_len = (~rec_his_mask).sum(-1)
        
        user_emb = self.his2feat(rec_his_emb, rec_his_len)

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
        # user_id not used in GRU4REC
        rec_his_len = (~rec_his_mask).sum(-1)

        user_emb = self.his2feat(rec_his_emb, rec_his_len)

        return user_emb
    

    def train_step(self, x, y):

        user_emb = self.forward(x)

        target_item_emb = self.emb_tab_dict['item_id_emb_tab'](y)

        loss = self.calulate_loss(user_emb, target_item_emb, y)

        return loss


class GRU4REC_aug(GRU4REC):
    def __init__(self, init_para):
        super().__init__(init_para)

        self.linear_transform = nn.Sequential(
            nn.Linear(self.model_config['hidden_size']+self.model_config['aug_dense_feat_dim'],
                       self.model_config['hidden_size']),
            nn.GELU(), 
            nn.Linear(self.model_config['hidden_size'], self.model_config['hidden_size'])
        )

        self.apply(self._init_weights)

    def forward(self, x, return_recon_loss=False):

        aug_feat, recon_loss = self.embed_orig_and_aug_data_fileds(x)
        x, aug_x = x
        _, rec_his_emb, rec_his_mask = self.embed_user_behaviors(x[0], x[1])
        rec_his_len = (~rec_his_mask).sum(-1)

        user_emb = self.his2feat(rec_his_emb, rec_his_len)
        user_emb = user_emb + 0.1*aug_feat
        #user_emb = self.linear_transform(torch.cat([user_emb, aug_feat], dim=-1))

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
    
    def predict(self, x, y, topK = 50):
        '''
        prediction for testing (full item setting)
        '''

        user_emb = self.forward(x)

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
                _, rec_his_emb, rec_his_mask = self.embed_user_behaviors(input_data[index-1], data)
                rec_his_len = (~rec_his_mask).sum(-1)

                history_data = self.his2feat(rec_his_emb, rec_his_len)
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
