# -*- coding: utf-8 -*-
# @Time   : 2020/9/30 14:07
# @Author : Yujie Lu
# @Email  : yujielu1998@gmail.com

r"""
SRGNN
################################################

Reference:
    Shu Wu et al. "Session-based Recommendation with Graph Neural Networks." in AAAI 2019.

Reference code:
    https://github.com/CRIPAC-DIG/SR-GNN

"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F

from .base import BaseModel



class GNN(nn.Module):
    r"""Graph neural networks are well-suited for session-based recommendation,
    because it can automatically extract features of session graphs with considerations of rich node connections.
    """

    def __init__(self, embedding_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.embedding_size = embedding_size
        self.input_size = embedding_size * 2
        self.gate_size = embedding_size * 3
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.embedding_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.embedding_size))
        self.b_ioh = Parameter(torch.Tensor(self.embedding_size))

        self.linear_edge_in = nn.Linear(
            self.embedding_size, self.embedding_size, bias=True
        )
        self.linear_edge_out = nn.Linear(
            self.embedding_size, self.embedding_size, bias=True
        )

        self._reset_parameters()

    def _reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.embedding_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def GNNCell(self, A, hidden):
        r"""Obtain latent vectors of nodes via graph neural networks.

        Args:
            A(torch.FloatTensor):The connection matrix,shape of [batch_size, max_session_len, 2 * max_session_len]

            hidden(torch.FloatTensor):The item node embedding matrix, shape of
                [batch_size, max_session_len, embedding_size]

        Returns:
            torch.FloatTensor: Latent vectors of nodes,shape of [batch_size, max_session_len, embedding_size]

        """

        input_in = (
            torch.matmul(A[:, :, : A.size(1)], self.linear_edge_in(hidden)) + self.b_iah
        )
        input_out = (
            torch.matmul(
                A[:, :, A.size(1) : 2 * A.size(1)], self.linear_edge_out(hidden)
            )
            + self.b_ioh
        )
        # [batch_size, max_session_len, embedding_size * 2]
        inputs = torch.cat([input_in, input_out], 2)

        # gi.size equals to gh.size, shape of [batch_size, max_session_len, embedding_size * 3]
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        # (batch_size, max_session_len, embedding_size)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        reset_gate = torch.sigmoid(i_r + h_r)
        input_gate = torch.sigmoid(i_i + h_i)
        new_gate = torch.tanh(i_n + reset_gate * h_n)
        hy = (1 - input_gate) * hidden + input_gate * new_gate
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class SRGNN(BaseModel):
    r"""SRGNN regards the conversation history as a directed graph.
    In addition to considering the connection between the item and the adjacent item,
    it also considers the connection with other interactive items.

    Such as: A example of a session sequence(eg:item1, item2, item3, item2, item4) and the connection matrix A

    Outgoing edges:
        === ===== ===== ===== =====
         \    1     2     3     4
        === ===== ===== ===== =====
         1    0     1     0     0
         2    0     0    1/2   1/2
         3    0     1     0     0
         4    0     0     0     0
        === ===== ===== ===== =====

    Incoming edges:
        === ===== ===== ===== =====
         \    1     2     3     4
        === ===== ===== ===== =====
         1    0     0     0     0
         2   1/2    0    1/2    0
         3    0     1     0     0
         4    0     1     0     0
        === ===== ===== ===== =====
    """

    def __init__(self, init_para):
        super(SRGNN, self).__init__(init_para)

        # load parameters info
        self.embedding_size = self.model_config["embedding_size"]

        # define layers and loss
        # item embedding
        # define layers and loss
        self.step = 1
        self.gnn = GNN(self.embedding_size, self.step)
        self.linear_one = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.linear_two = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.linear_three = nn.Linear(self.embedding_size, 1, bias=False)
        self.linear_transform = nn.Linear(
            self.embedding_size * 2, self.embedding_size, bias=True
        )
        self.linear_transform = nn.Sequential(
            nn.Linear(self.embedding_size*2, self.embedding_size),
            nn.ReLU(),
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.ReLU(),
            nn.Linear(self.embedding_size, self.embedding_size)
        )

        # parameters initialization
        self._init_weights()
        self.gnn._reset_parameters()

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


    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def _get_slice(self, item_seq):
        # Mask matrix, shape of [batch_size, max_session_len]
        mask = item_seq.gt(0)
        items, n_node, A, alias_inputs = [], [], [], []
        max_n_node = item_seq.size(1)
        item_seq = item_seq.cpu().numpy()
        for u_input in item_seq:
            node = np.unique(u_input)
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            u_A = np.zeros((max_n_node, max_n_node))

            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break

                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1

            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)

            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
        # The relative coordinates of the item node, shape of [batch_size, max_session_len]
        alias_inputs = torch.LongTensor(alias_inputs).to(self.device)
        # The connecting matrix, shape of [batch_size, max_session_len, 2 * max_session_len]
        A = torch.FloatTensor(np.array(A)).to(self.device)
        # The unique item nodes, shape of [batch_size, max_session_len]
        items = torch.LongTensor(items).to(self.device)

        return alias_inputs, A, items, mask

    def forward(self, item_seq, item_seq_len, item_embeddings):
        alias_inputs, A, items, mask = self._get_slice(item_seq)
        hidden = item_embeddings
        hidden = self.gnn(A, hidden)
        alias_inputs = alias_inputs.view(-1, alias_inputs.size(1), 1).expand(
            -1, -1, self.embedding_size
        )
        seq_hidden = torch.gather(hidden, dim=1, index=alias_inputs)
        # fetch the last hidden state of last timestamp
        ht = self.gather_indexes(seq_hidden, item_seq_len - 1)
        q1 = self.linear_one(ht).view(ht.size(0), 1, ht.size(1))
        q2 = self.linear_two(seq_hidden)

        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * seq_hidden * mask.view(mask.size(0), -1, 1).float(), 1)
        seq_output = self.linear_transform(torch.cat([a, ht], dim=1))
        return seq_output

    
    def predict(self, x, y, topK = 50):
        '''
        prediction for testing (full item setting)
        '''

        _, rec_his_emb, rec_his_mask = self.embed_user_behaviors(x[0], x[1])
        rec_his_len = (~rec_his_mask).sum(-1)
        
        user_emb = self.forward(x[1], rec_his_len, rec_his_emb)

        # item_num, dim
        all_item_emb = self.emb_tab_dict['item_id_emb_tab'](
            torch.arange(1, self.num_items, device=self.device))

        #B, item_num
        logits =  torch.matmul(user_emb, torch.t(all_item_emb))
        logit_value, logit_index = torch.topk(logits, k=topK, dim=1)
        logit_index += 1

        return logit_index
    

    def train_step(self, x, y):

        _, rec_his_emb, rec_his_mask = self.embed_user_behaviors(x[0], x[1])
        rec_his_len = (~rec_his_mask).sum(-1)

        user_emb = self.forward(x[1], rec_his_len, rec_his_emb)

        target_item_emb = self.emb_tab_dict['item_id_emb_tab'](y)

        loss = self.calulate_loss(user_emb, target_item_emb, y)

        return loss


class SRGNN_aug(SRGNN):
    def __init__(self, init_para):
        super().__init__(init_para)

        self.intergrate_user_emb = nn.Sequential(
            nn.Linear(self.model_config['embedding_size']+self.model_config['aug_dense_feat_dim'],
                       self.model_config['embedding_size']),
            nn.GELU(), 
            nn.Linear(self.model_config['embedding_size'], self.model_config['embedding_size'])
        )

        # parameters initialization
        self._init_weights()
        self.gnn._reset_parameters()

    def get_user_emb(self, x, return_recon_loss=False):

        aug_feat, recon_loss = self.embed_orig_and_aug_data_fileds(x)
        x, aug_x = x
        _, rec_his_emb, rec_his_mask = self.embed_user_behaviors(x[0], x[1])
        rec_his_len = (~rec_his_mask).sum(-1)

        user_emb = self.forward(x[1], rec_his_len, rec_his_emb)

        #user_emb = self.intergrate_user_emb(torch.cat([user_emb, aug_feat], dim=-1))
        user_emb = user_emb + 0.1*aug_feat

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
                _, rec_his_emb, rec_his_mask = self.embed_user_behaviors(input_data[index-1], data)
                rec_his_len = (~rec_his_mask).sum(-1)

                history_data = self.forward(data, rec_his_len, rec_his_emb)
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
