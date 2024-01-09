
r"""
DCN
################################################
Reference:
    Ruoxi Wang at al. "Deep & Cross Network for Ad Click Predictions." in ADKDD 2017.

Reference code:
    https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/context_aware_recommender/dcn.py
"""

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_

from .DeepFM import MLPLayers
from .base import BaseModel
import torchsnooper


class DCN(BaseModel):
    """Deep & Cross Network replaces the wide part in Wide&Deep with cross network,
    automatically construct limited high-degree cross features, and learns the corresponding weights.

    """

    def __init__(self, init):
        super(DCN, self).__init__(init)

        # load parameters info
        self.mlp_hidden_size = self.model_config["mlp_hidden_size"]
        self.cross_layer_num = self.model_config["cross_layer_num"]
        self.dropout_prob = self.model_config["dropout_prob"]

        self.item_feat_size = sum([feat.embedding_dim for feat in self.input_data_ls if feat.name.startswith('item')])
        self.user_feat_size = sum([feat.embedding_dim for feat in self.input_data_ls if feat.name.startswith('user')])
        # self.total_dim_of_all_fileds = self.user_feat_size + 2 * self.item_feat_size
        if 'aug_dense_feat_dim' in self.model_config:
            self.total_dim_of_all_fileds = self.user_feat_size + 2 * self.item_feat_size + self.model_config['aug_dense_feat_dim']
        else:
            self.total_dim_of_all_fileds = self.user_feat_size + 2 * self.item_feat_size

        if self.name[-3:] == 'kar':
            self.total_dim_of_all_fileds += self.model_config['convert_arch'][-1] * 2

        # define layers and loss
        # init weight and bias of each cross layer
        self.cross_layer_w = nn.ParameterList(
            nn.Parameter(
                torch.randn(self.total_dim_of_all_fileds).to(
                    self.device
                )
            )
            for _ in range(self.cross_layer_num)
        )
        self.cross_layer_b = nn.ParameterList(
            nn.Parameter(
                torch.zeros(self.total_dim_of_all_fileds).to(
                    self.device
                )
            )
            for _ in range(self.cross_layer_num)
        )

        # size of mlp hidden layer
        size_list = [
            self.total_dim_of_all_fileds
        ] + self.mlp_hidden_size
        # size of cross network output
        in_feature_num = (
            self.total_dim_of_all_fileds + self.mlp_hidden_size[-1]
        )

        self.mlp_layers = MLPLayers(size_list, dropout=self.dropout_prob, bn=True)
        self.predict_layer = nn.Linear(in_feature_num, 1)
        self.sigmoid = nn.Sigmoid()

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)


    def cross_network(self, x_0):
        r"""Cross network is composed of cross layers, with each layer having the following formula.

        .. math:: x_{l+1} = x_0 {x_l^T} w_l + b_l + x_l

        :math:`x_l`, :math:`x_{l+1}` are column vectors denoting the outputs from the l -th and
        (l + 1)-th cross layers, respectively.
        :math:`w_l`, :math:`b_l` are the weight and bias parameters of the l -th layer.

        Args:
            x_0(torch.Tensor): Embedding vectors of all features, input of cross network.

        Returns:
            torch.Tensor:output of cross network, [batch_size, num_feature_field * embedding_size]

        """
        x_l = x_0
        for i in range(self.cross_layer_num):
            xl_w = torch.tensordot(x_l, self.cross_layer_w[i], dims=([1], [0]))
            xl_dot = (x_0.transpose(0, 1) * xl_w).transpose(0, 1)
            x_l = xl_dot + self.cross_layer_b[i] + x_l
        return x_l

    def forward(self, interaction):
        embedding_ls = self.embed_input_fields(interaction)

        dcn_all_embeddings = torch.cat(embedding_ls, dim=1) # batch_size, total_dim
        # dcn_all_embeddings = self.concat_embed_input_fields(
        #     interaction
        # )  # [batch_size, num_field, embed_dim]
        # batch_size = dcn_all_embeddings.shape[0]

        # DNN
        deep_output = self.mlp_layers(dcn_all_embeddings)
        # Cross Network
        cross_output = self.cross_network(dcn_all_embeddings)
        stack = torch.cat([cross_output, deep_output], dim=-1)
        output = self.predict_layer(stack)

        return output.squeeze(1)


    
    def train_step(self, x, y):
        loss = self.loss_func( self.predict(x), y)

        return loss
    
    def predict(self, x):

        return self.sigmoid(self.forward(x))
    

class DCN_aug(DCN):
    def __init__(self, init):
        super().__init__(init)

        self.aug_feat_linear = nn.Sequential(
            nn.Linear(self.model_config['aug_dense_feat_dim'], self.model_config['aug_dense_feat_dim']),
            nn.GELU()
        )
        
        
    # @torchsnooper.snoop()
    def forward(self, interaction):
        embedding_ls, context_dense_vec, cot_reconstruct_loss = self.embed_orig_and_aug_data_fileds(interaction)
        

        embedding_ls.append(self.aug_feat_linear(context_dense_vec))

        dcn_all_embeddings = torch.cat(embedding_ls, dim=1) # batch_size, total_dim
        # dcn_all_embeddings = self.concat_embed_input_fields(
        #     interaction
        # )  # [batch_size, num_field, embed_dim]
        # batch_size = dcn_all_embeddings.shape[0]

        # DNN
        deep_output = self.mlp_layers(dcn_all_embeddings)
        # Cross Network
        cross_output = self.cross_network(dcn_all_embeddings)
        stack = torch.cat([cross_output, deep_output], dim=-1)
        output = self.predict_layer(stack)

        return output.squeeze(1), cot_reconstruct_loss


    def train_step(self, x, y):
        output, cot_reconstruct_loss = self.predict(x)
        loss = self.loss_func(output, y) + self.model_config['reconstruct_weight']*cot_reconstruct_loss

        return loss
    
    def predict(self, x):
        logits, cot_reconstruct_loss = self.forward(x)
        return self.sigmoid(logits), cot_reconstruct_loss
    

class DCN_aug_wocot(DCN_aug):
    def __init__(self, init):
        super().__init__(init)
        
        
    # @torchsnooper.snoop()
    def forward(self, interaction):
        embedding_ls, context_dense_vec = self.embed_orig_and_aug_data_wocot_fileds(interaction)
        

        embedding_ls.append(self.aug_feat_linear(context_dense_vec))

        dcn_all_embeddings = torch.cat(embedding_ls, dim=1) # batch_size, total_dim
        # dcn_all_embeddings = self.concat_embed_input_fields(
        #     interaction
        # )  # [batch_size, num_field, embed_dim]
        # batch_size = dcn_all_embeddings.shape[0]

        # DNN
        deep_output = self.mlp_layers(dcn_all_embeddings)
        # Cross Network
        cross_output = self.cross_network(dcn_all_embeddings)
        stack = torch.cat([cross_output, deep_output], dim=-1)
        output = self.predict_layer(stack)

        return output.squeeze(1)


    def train_step(self, x, y):
        output = self.predict(x)
        loss = self.loss_func(output, y)

        return loss
    
    def predict(self, x):
        logits = self.forward(x)
        return self.sigmoid(logits)
    

class DCN_aug_mean(DCN):
    def __init__(self, init):
        super().__init__(init)

        self.aug_feat_linear = nn.Sequential(
            nn.Linear(self.model_config['aug_dense_feat_dim'], self.model_config['aug_dense_feat_dim']),
            nn.GELU()
        )
        
        
    # @torchsnooper.snoop()
    def forward(self, interaction):
        embedding_ls, context_dense_vec = self.embed_orig_and_aug_data_mean_fileds(interaction)
        

        embedding_ls.append(self.aug_feat_linear(context_dense_vec))

        dcn_all_embeddings = torch.cat(embedding_ls, dim=1) # batch_size, total_dim
        # dcn_all_embeddings = self.concat_embed_input_fields(
        #     interaction
        # )  # [batch_size, num_field, embed_dim]
        # batch_size = dcn_all_embeddings.shape[0]

        # DNN
        deep_output = self.mlp_layers(dcn_all_embeddings)
        # Cross Network
        cross_output = self.cross_network(dcn_all_embeddings)
        stack = torch.cat([cross_output, deep_output], dim=-1)
        output = self.predict_layer(stack)

        return output.squeeze(1)


    def train_step(self, x, y):
        output = self.predict(x)
        loss = self.loss_func(output, y)

        return loss
    
    def predict(self, x):
        logits = self.forward(x)
        return self.sigmoid(logits)
    
class DCN_aug_kd(DCN):
    def __init__(self, init):
        super().__init__(init)

        self.aug_feat_linear = nn.Sequential(
            nn.Linear(self.model_config['aug_dense_feat_dim'], self.model_config['aug_dense_feat_dim']),
            nn.GELU()
        )
        
        
    # @torchsnooper.snoop()
    def forward(self, interaction):
        embedding_ls, context_dense_vec, cot_reconstruct_loss = self.embed_orig_and_aug_data_fileds(interaction)
        

        embedding_ls.append(self.aug_feat_linear(context_dense_vec))

        dcn_all_embeddings = torch.cat(embedding_ls, dim=1) # batch_size, total_dim
        # dcn_all_embeddings = self.concat_embed_input_fields(
        #     interaction
        # )  # [batch_size, num_field, embed_dim]
        # batch_size = dcn_all_embeddings.shape[0]

        # DNN
        deep_output = self.mlp_layers(dcn_all_embeddings)
        # Cross Network
        cross_output = self.cross_network(dcn_all_embeddings)
        stack = torch.cat([cross_output, deep_output], dim=-1)
        output = self.predict_layer(stack)

        return output.squeeze(1), cot_reconstruct_loss


    def train_step(self, x, y, llm_score):
        output, cot_reconstruct_loss = self.predict(x)

        distill_loss = self._add_knowledge_distillation_loss(llm_score, output)

        loss = self.loss_func(output, y) + self.model_config['reconstruct_weight']*cot_reconstruct_loss\
                                         + self.model_config['distill_weight']*distill_loss

        return loss
    
    def predict(self, x):
        logits, cot_reconstruct_loss = self.forward(x)
        return self.sigmoid(logits), cot_reconstruct_loss
    
class DCN_kd(DCN):
    def __init__(self, init):
        super().__init__(init)

    def train_step(self, x, y, llm_score):
        output = self.predict(x)

        distill_loss = self._add_knowledge_distillation_loss(llm_score, output)

        loss = self.loss_func(output, y) + self.model_config['distill_weight']*distill_loss

        return loss
    
class DCN_kar(DCN):
    def __init__(self, init):
        super().__init__(init)

        from .layers import ConvertNet

        self.convert_module = ConvertNet(self.model_config['export_num'], 
                                         self.model_config['specific_export_num'],
                                         self.model_config['convert_arch'],
                                         self.model_config['inp_dim'],
                                         self.model_config['dropout'] )        
        
    # @torchsnooper.snoop()
    def forward(self, interaction):

        x, dense_vec = interaction

        embedding_ls = self.embed_input_fields(x)
        
        dense_vec = self.convert_module(dense_vec)

        embedding_ls.append(dense_vec)

        dcn_all_embeddings = torch.cat(embedding_ls, dim=1) # batch_size, total_dim
        # dcn_all_embeddings = self.concat_embed_input_fields(
        #     interaction
        # )  # [batch_size, num_field, embed_dim]
        # batch_size = dcn_all_embeddings.shape[0]

        # DNN
        deep_output = self.mlp_layers(dcn_all_embeddings)
        # Cross Network
        cross_output = self.cross_network(dcn_all_embeddings)
        stack = torch.cat([cross_output, deep_output], dim=-1)
        output = self.predict_layer(stack)

        return output.squeeze(1)


    def train_step(self, x, y):
        output = self.predict(x)
        loss = self.loss_func(output, y) 

        return loss
    
    def predict(self, x):
        logits = self.forward(x)
        return self.sigmoid(logits)

