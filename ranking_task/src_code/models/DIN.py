import torch.nn as nn
import torch

import torchsnooper

from .base import BaseModel


'''
referred to https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/sequential_recommender/din.py
'''

class DIN(BaseModel):
    def __init__(self, init_para):
        super().__init__(init_para)

        self.item_feat_size = sum([feat.embedding_dim for feat in self.input_data_ls if feat.name.startswith('item')])
        self.user_feat_size = sum([feat.embedding_dim for feat in self.input_data_ls if feat.name.startswith('user')])

        if 'aug_dense_feat_dim' in self.model_config:
            self.total_dim_of_all_fileds = self.user_feat_size + 2 * self.item_feat_size + self.model_config['aug_dense_feat_dim']
        else:
            self.total_dim_of_all_fileds = self.user_feat_size + 2 * self.item_feat_size

        if self.name[-3:] == 'kar':
            self.total_dim_of_all_fileds += self.model_config['convert_arch'][-1] * 2



        self.attn = AttentionSequencePoolingLayer(embedding_dim=self.item_feat_size)
        self.fc_layer = FullyConnectedLayer(input_size=self.total_dim_of_all_fileds,
                                            hidden_unit=self.model_config['hid_units'],
                                            batch_norm=False,
                                            sigmoid = True,
                                            activation='relu',
                                            dropout=self.model_config['dropout'],
                                            dice_dim=2)

        self.loss_func = nn.BCELoss()

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
    def forward(self, input_data):

        user_emb, item_emb, rec_his_emb, rec_his_mask = self.embed_input_fields_for_attention_pooling(input_data)
       
        browse_atten = self.attn(item_emb.unsqueeze(dim=1),
                            rec_his_emb, rec_his_mask) 
        concat_feature = torch.cat([item_emb, browse_atten.squeeze(dim=1), user_emb], dim=-1)
        
        # fully-connected layers
        output = self.fc_layer(concat_feature).squeeze(dim=-1) #batch,1 --> batch 

        return output
        
    def train_step(self, x, y):
        loss = self.loss_func( self.predict(x), y)

        return loss
    
    def predict(self, x):

        return self.forward(x)
      
    

class AttentionSequencePoolingLayer(nn.Module):
    def __init__(self, embedding_dim=4):
        super(AttentionSequencePoolingLayer, self).__init__()

        self.local_att = LocalActivationUnit(hidden_unit=[64, 16], embedding_dim=embedding_dim, batch_norm=False)

    
    def forward(self, query_ad, user_behavior, mask=None):
        # query ad            : size -> batch_size * 1 * embedding_size
        # user behavior       : size -> batch_size * time_seq_len * embedding_size
        # mask                : size -> batch_size * time_seq_len
        # output              : size -> batch_size * 1 * embedding_size
        
        attention_score = self.local_att(query_ad, user_behavior)
        attention_score = torch.transpose(attention_score, 1, 2)  # B * 1 * T
        
        if mask is not None:
            attention_score = attention_score.masked_fill(mask.unsqueeze(1), torch.tensor(0))
        

        # multiply weight
        output = torch.matmul(attention_score, user_behavior)

        return output
        

class LocalActivationUnit(nn.Module):
    def __init__(self, hidden_unit=[80, 40], embedding_dim=4, batch_norm=False):
        super(LocalActivationUnit, self).__init__()
        self.fc1 = FullyConnectedLayer(input_size=4*embedding_dim,
                                       hidden_unit=hidden_unit,
                                       batch_norm=batch_norm,
                                       sigmoid=False,
                                       activation='relu',
                                       dice_dim=3)

        self.fc2 = nn.Linear(hidden_unit[-1], 1)

    # @torchsnooper.snoop()
    def forward(self, query, user_behavior):
        # query ad            : size -> batch_size * 1 * embedding_size
        # user behavior       : size -> batch_size * time_seq_len * embedding_size

        user_behavior_len = user_behavior.size(1)
        
        queries = query.expand(-1, user_behavior_len, -1)
        
        attention_input = torch.cat([queries, user_behavior, queries-user_behavior, queries*user_behavior],
             dim=-1) # as the source code, subtraction simulates verctors' difference
        
        attention_output = self.fc1(attention_input)
        attention_score = self.fc2(attention_output) # [B, T, 1]

        return attention_score


class FullyConnectedLayer(nn.Module):
    def __init__(self, input_size, hidden_unit, batch_norm=False, activation='relu', sigmoid=False, dropout=None, dice_dim=None):
        super(FullyConnectedLayer, self).__init__()
        assert len(hidden_unit) >= 1 
        self.sigmoid = sigmoid

        layers = []
        layers.append(nn.Linear(input_size, hidden_unit[0]))
        
        for i, h in enumerate(hidden_unit[:-1]):
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_unit[i]))
            
            if activation.lower() == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation.lower() == 'tanh':
                layers.append(nn.Tanh())
            elif activation.lower() == 'leakyrelu':
                layers.append(nn.LeakyReLU())
            # elif activation.lower() == 'dice':
            #     assert dice_dim
            #     layers.append(Dice(hidden_unit[i], dim=dice_dim))
            else:
                raise NotImplementedError

            if dropout is not None:
                layers.append(nn.Dropout(dropout))
            
            layers.append(nn.Linear(hidden_unit[i], hidden_unit[i+1]))
        
        self.fc = nn.Sequential(*layers)
        if self.sigmoid:
            self.output_layer = nn.Sigmoid()
        

    def forward(self, x):
        return self.output_layer(self.fc(x)) if self.sigmoid else self.fc(x) 
        
class DIN_aug(DIN):
    def __init__(self, init_para):
        super().__init__(init_para)

    def forward(self, input_data):

        embedding_ls, context_dense_vec, cot_reconstruct_loss = self.embed_orig_and_aug_data_fileds(input_data)

        user_emb, item_emb, rec_his_emb, rec_his_mask = self.embed_input_fields_for_attention_pooling(input_data[0])
       
        browse_atten = self.attn(item_emb.unsqueeze(dim=1),
                            rec_his_emb, rec_his_mask) 
        concat_feature = torch.cat([item_emb, browse_atten.squeeze(dim=1), user_emb, context_dense_vec], dim=-1)
        
        # fully-connected layers
        output = self.fc_layer(concat_feature).squeeze(dim=-1) #batch,1 --> batch 

        return output, cot_reconstruct_loss

    def train_step(self, x, y):
        output, cot_reconstruct_loss = self.predict(x)
        loss = self.loss_func(output, y) + self.model_config['reconstruct_weight']*cot_reconstruct_loss

        return loss
    
    def predict(self, x):
        logits, cot_reconstruct_loss = self.forward(x)
        return logits, cot_reconstruct_loss
    
class DIN_aug_wocot(DIN_aug):
    def __init__(self, init_para):
        super().__init__(init_para)

    def forward(self, input_data):

        embedding_ls, context_dense_vec = self.embed_orig_and_aug_data_wocot_fileds(input_data)

        user_emb, item_emb, rec_his_emb, rec_his_mask = self.embed_input_fields_for_attention_pooling(input_data[0])
       
        browse_atten = self.attn(item_emb.unsqueeze(dim=1),
                            rec_his_emb, rec_his_mask) 
        concat_feature = torch.cat([item_emb, browse_atten.squeeze(dim=1), user_emb, context_dense_vec], dim=-1)
        
        # fully-connected layers
        output = self.fc_layer(concat_feature).squeeze(dim=-1) #batch,1 --> batch 

        return output

    def train_step(self, x, y):
        output = self.predict(x)
        loss = self.loss_func(output, y)

        return loss
    
    def predict(self, x):
        logits = self.forward(x)
        return logits
    
class DIN_aug_mean(DIN):
    def __init__(self, init_para):
        super().__init__(init_para)

    def forward(self, input_data):

        embedding_ls, context_dense_vec = self.embed_orig_and_aug_data_mean_fileds(input_data)

        user_emb, item_emb, rec_his_emb, rec_his_mask = self.embed_input_fields_for_attention_pooling(input_data[0])
       
        browse_atten = self.attn(item_emb.unsqueeze(dim=1),
                            rec_his_emb, rec_his_mask) 
        concat_feature = torch.cat([item_emb, browse_atten.squeeze(dim=1), user_emb, context_dense_vec], dim=-1)
        
        # fully-connected layers
        output = self.fc_layer(concat_feature).squeeze(dim=-1) #batch,1 --> batch 

        return output

    def train_step(self, x, y):
        output = self.predict(x)
        loss = self.loss_func(output, y)

        return loss
    
    def predict(self, x):
        logits = self.forward(x)
        return logits
    
class DIN_aug_kd(DIN):
    def __init__(self, init_para):
        super().__init__(init_para)

    def forward(self, input_data):

        embedding_ls, context_dense_vec, cot_reconstruct_loss = self.embed_orig_and_aug_data_fileds(input_data)

        user_emb, item_emb, rec_his_emb, rec_his_mask = self.embed_input_fields_for_attention_pooling(input_data[0])
       
        browse_atten = self.attn(item_emb.unsqueeze(dim=1),
                            rec_his_emb, rec_his_mask) 
        concat_feature = torch.cat([item_emb, browse_atten.squeeze(dim=1), user_emb, context_dense_vec], dim=-1)
        
        # fully-connected layers
        output = self.fc_layer(concat_feature).squeeze(dim=-1) #batch,1 --> batch 

        return output, cot_reconstruct_loss

    def train_step(self, x, y, llm_score):
        output, cot_reconstruct_loss = self.predict(x)

        distill_loss = self._add_knowledge_distillation_loss(llm_score, output)

        loss = self.loss_func(output, y) + self.model_config['reconstruct_weight']*cot_reconstruct_loss\
                                         + self.model_config['distill_weight']*distill_loss
        return loss
    
    def predict(self, x):
        logits, cot_reconstruct_loss = self.forward(x)
        return logits, cot_reconstruct_loss

class DIN_kd(DIN):
    def __init__(self, init_para):
        super().__init__(init_para)

    def train_step(self, x, y, llm_score):
        output = self.predict(x)

        distill_loss = self._add_knowledge_distillation_loss(llm_score, output)

        loss = self.loss_func(output, y) + self.model_config['distill_weight']*distill_loss
        
        return loss

class DIN_kar(DIN):
    def __init__(self, init_para):
        super().__init__(init_para)

        from .layers import ConvertNet

        self.convert_module = ConvertNet(self.model_config['export_num'], 
                                         self.model_config['specific_export_num'],
                                         self.model_config['convert_arch'],
                                         self.model_config['inp_dim'],
                                         self.model_config['dropout'] )  

    def forward(self, input_data):

        input_data, dense_vec = input_data

        user_emb, item_emb, rec_his_emb, rec_his_mask = self.embed_input_fields_for_attention_pooling(input_data)
       
        dense_vec = self.convert_module(dense_vec)

        browse_atten = self.attn(item_emb.unsqueeze(dim=1),
                            rec_his_emb, rec_his_mask) 
        concat_feature = torch.cat([item_emb, browse_atten.squeeze(dim=1), user_emb, dense_vec], dim=-1)
        
        # fully-connected layers
        output = self.fc_layer(concat_feature).squeeze(dim=-1) #batch,1 --> batch 

        return output

    def train_step(self, x, y):
        loss = self.loss_func( self.predict(x), y)

        return loss
    
    def predict(self, x):

        return self.forward(x)
      