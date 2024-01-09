
r"""
xDeepFM
################################################
Reference:
    Jianxun Lian at al. "xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems."
    in SIGKDD 2018.

Reference code:
    - https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/context_aware_recommender/xdeepfm.py
"""

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_

from .DeepFM import MLPLayers, lr
from .base import BaseModel



class xDeepFM(BaseModel):
    """xDeepFM combines a CIN (Compressed Interaction Network) with a classical DNN.
    The model is able to learn certain bounded-degree feature interactions explicitly;
    Besides, it can also learn arbitrary low- and high-order feature interactions implicitly.
    """

    def __init__(self, init):
        super(xDeepFM, self).__init__(init)

        # load parameters info
        self.mlp_hidden_size = self.model_config["mlp_hidden_size"]
        # self.reg_weight = self.model_config["reg_weight"]
        self.dropout_prob = self.model_config["dropout_prob"]
        self.direct = self.model_config["direct"]
        self.cin_layer_size = temp_cin_size = list(self.model_config["cin_layer_size"])

        # Check whether the size of the CIN layer is legal.
        if not self.direct:
            self.cin_layer_size = list(map(lambda x: int(x // 2 * 2), temp_cin_size))
            if self.cin_layer_size[:-1] != temp_cin_size[:-1]:
                self.logger.warning(
                    "Layer size of CIN should be even except for the last layer when direct is True."
                    "It is changed to {}".format(self.cin_layer_size)
                )

        self.num_feature_field = self.model_config['num_feature_field']
        # Create a convolutional layer for each CIN layer
        self.conv1d_list = nn.ModuleList()
        self.field_nums = [self.num_feature_field]
        for i, layer_size in enumerate(self.cin_layer_size):
            conv1d = nn.Conv1d(self.field_nums[-1] * self.field_nums[0], layer_size, 1)
            self.conv1d_list.append(conv1d)
            if self.direct:
                self.field_nums.append(layer_size)
            else:
                self.field_nums.append(layer_size // 2)

        self.lr = lr(self.model_config['lr_dim'], self.input_data_ls)
        self.lr.Linear_bias = self.lr.Linear_bias.to(self.device)

        self.item_feat_size = sum([feat.embedding_dim for feat in self.input_data_ls if feat.name.startswith('item')])
        self.user_feat_size = sum([feat.embedding_dim for feat in self.input_data_ls if feat.name.startswith('user')])
        # self.total_dim_of_all_fileds = self.user_feat_size + 2 * self.item_feat_size

        if 'aug_dense_feat_dim' in self.model_config:
            self.total_dim_of_all_fileds = self.user_feat_size + 2 * self.item_feat_size + self.model_config['aug_dense_feat_dim']
        else:
            self.total_dim_of_all_fileds = self.user_feat_size + 2 * self.item_feat_size


        if self.name[-3:] == 'kar':
            self.total_dim_of_all_fileds += self.model_config['convert_arch'][-1] * 2


        # Create MLP layer
        size_list = (
            [self.total_dim_of_all_fileds] + self.mlp_hidden_size + [1]
        )
        self.mlp_layers = MLPLayers(size_list, dropout=self.dropout_prob)

        # Get the output size of CIN
        if self.direct:
            self.final_len = sum(self.cin_layer_size)
        else:
            self.final_len = (
                sum(self.cin_layer_size[:-1]) // 2 + self.cin_layer_size[-1]
            )

        self.cin_linear = nn.Linear(self.final_len, 1)
        self.sigmoid = nn.Sigmoid()
        # self.loss = nn.BCEWithLogitsLoss()
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding) or isinstance(module, nn.Conv1d):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)




    def compressed_interaction_network(self, input_features, activation="ReLU"):
        r"""For k-th CIN layer, the output :math:`X_k` is calculated via

        .. math::
            x_{h,*}^{k} = \sum_{i=1}^{H_k-1} \sum_{j=1}^{m}W_{i,j}^{k,h}(X_{i,*}^{k-1} \circ x_{j,*}^0)

        :math:`H_k` donates the number of feature vectors in the k-th layer,
        :math:`1 \le h \le H_k`.
        :math:`\circ` donates the Hadamard product.

        And Then, We apply sum pooling on each feature map of the hidden layer.
        Finally, All pooling vectors from hidden layers are concatenated.

        Args:
            input_features(torch.Tensor): [batch_size, field_num, embed_dim]. Embedding vectors of all features.
            activation(str): name of activation function.

        Returns:
            torch.Tensor: [batch_size, num_feature_field * embedding_size]. output of CIN layer.
        """
        batch_size, _, embedding_size = input_features.shape
        hidden_nn_layers = [input_features]
        final_result = []
        for i, layer_size in enumerate(self.cin_layer_size):
            z_i = torch.einsum(
                "bhd,bmd->bhmd", hidden_nn_layers[-1], hidden_nn_layers[0]
            )
            z_i = z_i.view(
                batch_size, self.field_nums[0] * self.field_nums[i], embedding_size
            )
            z_i = self.conv1d_list[i](z_i)

            # Pass the CIN intermediate result through the activation function.
            if activation.lower() == "identity":
                output = z_i
            else:
                activate_func = activation_layer(activation)
                if activate_func is None:
                    output = z_i
                else:
                    output = activate_func(z_i)

            # Get the output of the hidden layer.
            if self.direct:
                direct_connect = output
                next_hidden = output
            else:
                if i != len(self.cin_layer_size) - 1:
                    next_hidden, direct_connect = torch.split(
                        output, 2 * [layer_size // 2], 1
                    )
                else:
                    direct_connect = output
                    next_hidden = 0

            final_result.append(direct_connect)
            hidden_nn_layers.append(next_hidden)
        result = torch.cat(final_result, dim=1)
        result = torch.sum(result, -1)
        return result

    def forward(self, interaction):
        embedding_ls = self.embed_input_fields(interaction)
        xdeepfm_input = torch.stack(embedding_ls, dim=1)

        # Get the output of CIN.
        # xdeepfm_input = self.concat_embed_input_fields(
        #     interaction
        # )  # [batch_size, num_field, embed_dim]
        cin_output = self.compressed_interaction_network(xdeepfm_input)
        cin_output = self.cin_linear(cin_output)

        # Get the output of MLP layer.
        batch_size = xdeepfm_input.shape[0]
        dnn_output = self.mlp_layers(xdeepfm_input.view(batch_size, -1))

        # Get predicted score.
        y_p = self.lr(interaction) + cin_output + dnn_output

        return y_p.squeeze(1)

    def train_step(self, x, y):
        loss = self.loss_func( self.predict(x), y)

        return loss
    
    def predict(self, x):

        return self.sigmoid(self.forward(x))
    
    
def activation_layer(activation_name="relu", emb_dim=None):
    """Construct activation layers

    Args:
        activation_name: str, name of activation function
        emb_dim: int, used for Dice activation

    Return:
        activation: activation layer
    """
    if activation_name is None:
        activation = None
    elif isinstance(activation_name, str):
        if activation_name.lower() == "sigmoid":
            activation = nn.Sigmoid()
        elif activation_name.lower() == "tanh":
            activation = nn.Tanh()
        elif activation_name.lower() == "relu":
            activation = nn.ReLU()
        elif activation_name.lower() == "leakyrelu":
            activation = nn.LeakyReLU()
        # elif activation_name.lower() == "dice":
        #     activation = Dice(emb_dim)
        elif activation_name.lower() == "none":
            activation = None
    elif issubclass(activation_name, nn.Module):
        activation = activation_name()
    else:
        raise NotImplementedError(
            "activation function {} is not implemented".format(activation_name)
        )

    return activation


class xDeepFM_aug(xDeepFM):
    def __init__(self, init):
        super().__init__(init)

        self.lr_dense_vec_weight = nn.Linear(self.model_config['cot_mapping_mlp_hidden_size'][-1], 1, bias=False)


    def forward(self, interaction):

        embedding_ls, context_dense_vec, cot_reconstruct_loss = self.embed_orig_and_aug_data_fileds(interaction)
        embedding_ls.append(context_dense_vec)


        xdeepfm_input = torch.stack(embedding_ls, dim=1)

        # Get the output of CIN.
        # xdeepfm_input = self.concat_embed_input_fields(
        #     interaction
        # )  # [batch_size, num_field, embed_dim]
        cin_output = self.compressed_interaction_network(xdeepfm_input)
        cin_output = self.cin_linear(cin_output)

        # Get the output of MLP layer.
        batch_size = xdeepfm_input.shape[0]
        dnn_output = self.mlp_layers(xdeepfm_input.view(batch_size, -1))

        # Get predicted score.
        y_p = self.lr(interaction[0]) + self.lr_dense_vec_weight(context_dense_vec)\
              + cin_output + dnn_output

        return y_p.squeeze(1), cot_reconstruct_loss

    def train_step(self, x, y):
        output, cot_reconstruct_loss = self.predict(x)
        loss = self.loss_func(output, y) + self.model_config['reconstruct_weight']*cot_reconstruct_loss

        return loss
    
    def predict(self, x):
        logits, cot_reconstruct_loss = self.forward(x)
        return self.sigmoid(logits), cot_reconstruct_loss
    

class xDeepFM_aug_wocot(xDeepFM):
    def __init__(self, init):
        super().__init__(init)

        self.lr_dense_vec_weight = nn.Linear(self.model_config['cot_mapping_mlp_hidden_size'][-1], 1, bias=False)


    def forward(self, interaction):

        embedding_ls, context_dense_vec = self.embed_orig_and_aug_data_wocot_fileds(interaction)
        embedding_ls.append(context_dense_vec)


        xdeepfm_input = torch.stack(embedding_ls, dim=1)

        # Get the output of CIN.
        # xdeepfm_input = self.concat_embed_input_fields(
        #     interaction
        # )  # [batch_size, num_field, embed_dim]
        cin_output = self.compressed_interaction_network(xdeepfm_input)
        cin_output = self.cin_linear(cin_output)

        # Get the output of MLP layer.
        batch_size = xdeepfm_input.shape[0]
        dnn_output = self.mlp_layers(xdeepfm_input.view(batch_size, -1))

        # Get predicted score.
        y_p = self.lr(interaction[0]) + self.lr_dense_vec_weight(context_dense_vec)\
              + cin_output + dnn_output

        return y_p.squeeze(1)

    def train_step(self, x, y):
        output = self.predict(x)
        loss = self.loss_func(output, y)

        return loss
    
    def predict(self, x):
        logits = self.forward(x)
        return self.sigmoid(logits)
    

class xDeepFM_aug_mean(xDeepFM):
    def __init__(self, init):
        super().__init__(init)

        self.lr_dense_vec_weight = nn.Linear(self.model_config['cot_mapping_mlp_hidden_size'][-1], 1, bias=False)


    def forward(self, interaction):

        embedding_ls, context_dense_vec = self.embed_orig_and_aug_data_mean_fileds(interaction)
        embedding_ls.append(context_dense_vec)


        xdeepfm_input = torch.stack(embedding_ls, dim=1)

        # Get the output of CIN.
        # xdeepfm_input = self.concat_embed_input_fields(
        #     interaction
        # )  # [batch_size, num_field, embed_dim]
        cin_output = self.compressed_interaction_network(xdeepfm_input)
        cin_output = self.cin_linear(cin_output)

        # Get the output of MLP layer.
        batch_size = xdeepfm_input.shape[0]
        dnn_output = self.mlp_layers(xdeepfm_input.view(batch_size, -1))

        # Get predicted score.
        y_p = self.lr(interaction[0]) + self.lr_dense_vec_weight(context_dense_vec)\
              + cin_output + dnn_output

        return y_p.squeeze(1)

    def train_step(self, x, y):
        output = self.predict(x)
        loss = self.loss_func(output, y)

        return loss
    
    def predict(self, x):
        logits = self.forward(x)
        return self.sigmoid(logits)
    

class xDeepFM_aug_kd(xDeepFM):
    def __init__(self, init):
        super().__init__(init)

        self.lr_dense_vec_weight = nn.Linear(self.model_config['cot_mapping_mlp_hidden_size'][-1], 1, bias=False)


    def forward(self, interaction):

        embedding_ls, context_dense_vec, cot_reconstruct_loss = self.embed_orig_and_aug_data_fileds(interaction)
        embedding_ls.append(context_dense_vec)


        xdeepfm_input = torch.stack(embedding_ls, dim=1)

        # Get the output of CIN.
        # xdeepfm_input = self.concat_embed_input_fields(
        #     interaction
        # )  # [batch_size, num_field, embed_dim]
        cin_output = self.compressed_interaction_network(xdeepfm_input)
        cin_output = self.cin_linear(cin_output)

        # Get the output of MLP layer.
        batch_size = xdeepfm_input.shape[0]
        dnn_output = self.mlp_layers(xdeepfm_input.view(batch_size, -1))

        # Get predicted score.
        y_p = self.lr(interaction[0]) + self.lr_dense_vec_weight(context_dense_vec)\
              + cin_output + dnn_output

        return y_p.squeeze(1), cot_reconstruct_loss

    def train_step(self, x, y, llm_score):
        output, cot_reconstruct_loss = self.predict(x)

        distill_loss = self._add_knowledge_distillation_loss(llm_score, output)

        loss = self.loss_func(output, y) + self.model_config['reconstruct_weight']*cot_reconstruct_loss\
                                         + self.model_config['distill_weight']*distill_loss
        return loss
    
    def predict(self, x):
        logits, cot_reconstruct_loss = self.forward(x)
        return self.sigmoid(logits), cot_reconstruct_loss    

class xDeepFM_kd(xDeepFM):
    def __init__(self, init):
        super().__init__(init)

    def train_step(self, x, y, llm_score):
        output = self.predict(x)

        distill_loss = self._add_knowledge_distillation_loss(llm_score, output)

        loss = self.loss_func(output, y) + self.model_config['distill_weight']*distill_loss
        
        return loss


class xDeepFM_kar(xDeepFM):
    def __init__(self, init):
        super().__init__(init)

        self.lr_dense_vec_weight = nn.Linear(self.model_config['convert_arch'][-1] * 2, 1, bias=False)

        from .layers import ConvertNet

        self.convert_module = ConvertNet(self.model_config['export_num'], 
                                         self.model_config['specific_export_num'],
                                         self.model_config['convert_arch'],
                                         self.model_config['inp_dim'],
                                         self.model_config['dropout'] ) 
        

    def forward(self, interaction):
        x, dense_vec = interaction

        embedding_ls = self.embed_input_fields(x)
        
        dense_vec = self.convert_module(dense_vec)

        embedding_ls.append(dense_vec)

        xdeepfm_input = torch.stack(embedding_ls, dim=1)

        # Get the output of CIN.
        # xdeepfm_input = self.concat_embed_input_fields(
        #     interaction
        # )  # [batch_size, num_field, embed_dim]
        cin_output = self.compressed_interaction_network(xdeepfm_input)
        cin_output = self.cin_linear(cin_output)

        # Get the output of MLP layer.
        batch_size = xdeepfm_input.shape[0]
        dnn_output = self.mlp_layers(xdeepfm_input.view(batch_size, -1))

        # Get predicted score.
        y_p = self.lr(x) + self.lr_dense_vec_weight(dense_vec) + cin_output + dnn_output

        return y_p.squeeze(1)
