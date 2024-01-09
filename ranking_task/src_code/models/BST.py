import torch.nn as nn
import torch
import torch.nn.functional as F

import torchsnooper

from .base import BaseModel



class BST(BaseModel):
    def __init__(self, init_para):
        super().__init__(init_para)

        self.item_feat_size = sum([feat.embedding_dim for feat in self.input_data_ls if feat.name.startswith('item')])
        self.user_feat_size = sum([feat.embedding_dim for feat in self.input_data_ls if feat.name.startswith('user')])

        self.transformer_encoder = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=self.item_feat_size, 
                                         nhead=self.model_config['n_head'],
                                         dim_feedforward=self.model_config['atten_size'],
                                         dropout=self.model_config['dropout'],
                                         activation=F.leaky_relu,
                                         batch_first=True) 
              for i in range(self.model_config['n_layers'])]
        )

        self.fc_layer = FullyConnectedLayer(
                                            # input_size=self.model_config['max_len']*self.item_feat_size+self.user_feat_size+self.item_feat_size,
                                            input_size=self.user_feat_size+2*self.item_feat_size,
                                            hidden_unit=self.model_config['hid_units'],
                                            batch_norm=False,
                                            sigmoid = True,
                                            activation='relu',
                                            dropout=self.model_config['dropout'],
                                            dice_dim=2)

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


    def forward(self, input_data):

        user_emb, item_emb, rec_his_emb, rec_his_mask = self.embed_input_fields_for_attention_pooling(input_data)
       
        concat_seq_emb = torch.cat([rec_his_emb, item_emb.unsqueeze(dim=1)], dim=1) # bs, history+1, dim
        concat_seq_mask = torch.cat(
            [rec_his_mask, torch.zeros(rec_his_mask.size(0), 1).bool().to(self.device)], dim=1
        )
        
        for layer in self.transformer_encoder:
            concat_seq_emb = layer(concat_seq_emb, src_key_padding_mask=concat_seq_mask)

        # browse_atten = torch.flatten(concat_seq_emb, start_dim=1) # bs, (history+1)*dim
        browse_atten = torch.sum(concat_seq_emb*(~concat_seq_mask.unsqueeze(-1)), dim=1) / (torch.sum(~concat_seq_mask, dim=1, keepdim=True)+1e-12) 
        concat_feature = torch.cat([browse_atten, user_emb, item_emb], dim=-1)
        
        # fully-connected layers
        output = self.fc_layer(concat_feature).squeeze(dim=-1) #batch,1 --> batch 

        return output
        
    def train_step(self, x, y):
        loss = self.loss_func( self.predict(x), y)

        return loss
    
    def predict(self, x):

        return self.forward(x)
      
    
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
        