import torch
import torch.nn as nn

from .base import BaseModel

import torchsnooper

'''
referred to https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/context_aware_recommender/deepfm.py
'''

class DeepFM(BaseModel):
    def __init__(self, init_para) -> None:
        super().__init__(init_para)

        self.register_embedding_tables()

        # load parameters info
        self.mlp_hidden_size = self.model_config["mlp_hidden_size"]
        self.dropout_prob = self.model_config["dropout_prob"]

        # define layers and loss
        self.fm = BaseFactorizationMachine(reduce_sum=True)
        size_list = [
            sum([ emb_tab_dict.weight.size(-1) for _, emb_tab_dict in self.emb_tab_dict.items()])\
            + sum([ emb_tab_dict.weight.size(-1) for name, emb_tab_dict in self.emb_tab_dict.items() if name.startswith('item')])
        ] + self.mlp_hidden_size

        self.lr = lr(self.model_config['lr_dim'], self.input_data_ls)
        self.lr.Linear_bias = self.lr.Linear_bias.to(self.device)

        self.mlp_layers = MLPLayers(size_list, self.dropout_prob)
        self.deep_predict_layer = nn.Linear(
            self.mlp_hidden_size[-1], 1
        )  # Linear product to the final score
        self.sigmoid = nn.Sigmoid()

    # @torchsnooper.snoop()
    def forward(self, x):
        embedding_ls = self.embed_input_fields(x)

        deepfm_all_embeddings = torch.cat(embedding_ls, dim=1) # batch_size, total_dim
        # batch_size = deepfm_all_embeddings.shape[0]
        y_fm = self.lr(x) + self.fm(torch.stack(embedding_ls, dim=1))

        y_deep = self.deep_predict_layer(
            self.mlp_layers(deepfm_all_embeddings)
        )
        y = y_fm + y_deep
        return y.squeeze(-1)
    
    def train_step(self, x, y):
        loss = self.loss_func( self.predict(x), y)

        return loss
    
    def predict(self, x):

        return self.sigmoid(self.forward(x))

class lr(nn.Module):
    def __init__(self, feat_dim, input_data_ls):
        super().__init__()

        self.input_data_ls = input_data_ls
        self.emb_tab_dict = nn.ModuleDict()
        for feat in self.input_data_ls:
            if feat.name.startswith('history'):
                continue
            self.emb_tab_dict[f'{feat.name}_emb_tab'] = \
                    nn.Embedding(num_embeddings=feat.vocabulary_size, 
                                 embedding_dim=feat_dim, padding_idx=0)
            
        self.Linear_bias = nn.parameter.Parameter(data=torch.randn(1), requires_grad=True).reshape(1, 1)
    
    def embed_input_fields(self, input_data):

        emb_data_ls = []

        def mean_pooling(tensor_data, id_data, dim):
            mask = id_data != 0
            tensor_data = torch.sum(tensor_data*mask.unsqueeze(-1), dim=dim)
            tensor_data = tensor_data / (torch.sum(mask, dim=dim, keepdim=True)+1e-9) # 1e-9 to avoid dividing zero

            return tensor_data

        for data_filed, data in zip(self.input_data_ls, input_data):
            if data_filed.name.startswith('history'):
                continue
                history_data = self.emb_tab_dict[f'{data_filed.name[len("history")+1:]}_emb_tab'](data)
                if len(data.size()) == 2:
                    history_data = mean_pooling(history_data, data, 1)
                elif len(data.size()) == 3:
                    history_data = mean_pooling(history_data.reshape(history_data.size(0), -1, history_data.size(-1)), 
                                                    data.reshape(data.size(0), -1), 1)
                else:
                    raise ValueError(f'wrong dimension of input data {data_filed.name} with size {data.size()}')
                emb_data_ls.append(history_data)
            elif data_filed.name in ['item_cate_id']:
                history_data = self.emb_tab_dict[f'{data_filed.name}_emb_tab'](data)
                if len(data.size()) == 2:
                    history_data = mean_pooling(history_data, data, 1)
                emb_data_ls.append(history_data)
            else:
                emb_data_ls.append(
                    self.emb_tab_dict[f'{data_filed.name}_emb_tab'](data)
                ) 

        return emb_data_ls
        
    def forward(self, x):

        embedding_ls = self.embed_input_fields(x)

        linear_logit = torch.sum(torch.cat(embedding_ls, dim=1), dim=1, keepdim=True) + self.Linear_bias
        
        return linear_logit



class BaseFactorizationMachine(nn.Module):
    r"""Calculate FM result over the embeddings

    Args:
        reduce_sum: bool, whether to sum the result, default is True.

    Input:
        input_x: tensor, A 3D tensor with shape:``(batch_size,field_size,embed_dim)``.

    Output
        output: tensor, A 3D tensor with shape: ``(batch_size,1)`` or ``(batch_size, embed_dim)``.
    """

    def __init__(self, reduce_sum=True):
        super(BaseFactorizationMachine, self).__init__()
        self.reduce_sum = reduce_sum

    def forward(self, input_x):
        square_of_sum = torch.sum(input_x, dim=1) ** 2
        sum_of_square = torch.sum(input_x**2, dim=1)
        output = square_of_sum - sum_of_square
        if self.reduce_sum:
            output = torch.sum(output, dim=1, keepdim=True)
        output = 0.5 * output
        return output
    
class MLPLayers(nn.Module):
    r"""MLPLayers

    Args:
        - layers(list): a list contains the size of each layer in mlp layers
        - dropout(float): probability of an element to be zeroed. Default: 0
        - activation(str): activation function after each layer in mlp layers. Default: 'relu'.
                           candidates: 'sigmoid', 'tanh', 'relu', 'leekyrelu', 'none'

    Shape:

        - Input: (:math:`N`, \*, :math:`H_{in}`) where \* means any number of additional dimensions
          :math:`H_{in}` must equal to the first value in `layers`
        - Output: (:math:`N`, \*, :math:`H_{out}`) where :math:`H_{out}` equals to the last value in `layers`

    Examples::

        >>> m = MLPLayers([64, 32, 16], 0.2, 'relu')
        >>> input = torch.randn(128, 64)
        >>> output = m(input)
        >>> print(output.size())
        >>> torch.Size([128, 16])
    """

    def __init__(
        self,
        layers,
        dropout=0.0,
        activation="relu",
        bn=False,
        init_method=None,
        last_activation=True,
    ):
        super(MLPLayers, self).__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        self.use_bn = bn
        self.init_method = init_method

        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(
            zip(self.layers[:-1], self.layers[1:])
        ):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))
            if self.use_bn:
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))
            activation_func = nn.ReLU()
            if activation_func is not None:
                mlp_modules.append(activation_func)
        if self.activation is not None and not last_activation:
            mlp_modules.pop()
        self.mlp_layers = nn.Sequential(*mlp_modules)
    #     if self.init_method is not None:
    #         self.apply(self.init_weights)

    # def init_weights(self, module):
    #     # We just initialize the module with normal distribution as the paper said
    #     if isinstance(module, nn.Linear):
    #         if self.init_method == "norm":
    #             normal_(module.weight.data, 0, 0.01)
    #         if module.bias is not None:
    #             module.bias.data.fill_(0.0)

    def forward(self, input_feature):
        return self.mlp_layers(input_feature)


class DeepFM_aug(DeepFM):
    def __init__(self, init_para) -> None:
        super().__init__(init_para)

        del self.mlp_layers

        size_list = [
            self.model_config['decoder_emb_dim']\
            + sum([ emb_tab_dict.weight.size(-1) for _, emb_tab_dict in self.emb_tab_dict.items()])\
            + sum([ emb_tab_dict.weight.size(-1) for name, emb_tab_dict in self.emb_tab_dict.items() if name.startswith('item')])
        ] + self.mlp_hidden_size

        self.mlp_layers = MLPLayers(size_list, self.dropout_prob)

        self.lr_dense_vec_weight = nn.Linear(self.model_config['cot_mapping_mlp_hidden_size'][-1], 1, bias=False)

    # @torchsnooper.snoop()
    def forward(self, x):

        embedding_ls, context_dense_vec, cot_reconstruct_loss = self.embed_orig_and_aug_data_fileds(x)

        deepfm_all_embeddings = torch.cat(embedding_ls, dim=1) # batch_size, total_dim
        # batch_size = deepfm_all_embeddings.shape[0]
        y_fm = self.lr(x[0]) + self.lr_dense_vec_weight(context_dense_vec) + self.fm(torch.stack(embedding_ls, dim=1))

        y_deep = self.deep_predict_layer(
            self.mlp_layers(
               torch.cat([deepfm_all_embeddings, context_dense_vec], dim=1)
            )
        )
        y = y_fm + y_deep
        return y.squeeze(-1), cot_reconstruct_loss
    
    def train_step(self, x, y):
        output, cot_reconstruct_loss = self.predict(x)
        loss = self.loss_func(output, y) + self.model_config['reconstruct_weight']*cot_reconstruct_loss

        return loss
    
    def predict(self, x):
        logits, cot_reconstruct_loss = self.forward(x)
        return self.sigmoid(logits), cot_reconstruct_loss


class DeepFM_aug_wocot(DeepFM_aug):
    def __init__(self, init_para) -> None:
        super().__init__(init_para)

    def forward(self, x):

        embedding_ls, context_dense_vec = self.embed_orig_and_aug_data_wocot_fileds(x)

        deepfm_all_embeddings = torch.cat(embedding_ls, dim=1) # batch_size, total_dim
        # batch_size = deepfm_all_embeddings.shape[0]
        y_fm = self.lr(x[0]) + self.lr_dense_vec_weight(context_dense_vec) + self.fm(torch.stack(embedding_ls, dim=1))

        y_deep = self.deep_predict_layer(
            self.mlp_layers(
               torch.cat([deepfm_all_embeddings, context_dense_vec], dim=1)
            )
        )
        y = y_fm + y_deep
        return y.squeeze(-1)
    
    def train_step(self, x, y):
        output = self.predict(x)
        loss = self.loss_func(output, y)

        return loss
    
    def predict(self, x):
        logits = self.forward(x)
        return self.sigmoid(logits)
    

class DeepFM_aug_mean(DeepFM_aug):
    def __init__(self, init_para) -> None:
        super().__init__(init_para)

    def forward(self, x):

        embedding_ls, context_dense_vec = self.embed_orig_and_aug_data_mean_fileds(x)

        deepfm_all_embeddings = torch.cat(embedding_ls, dim=1) # batch_size, total_dim
        # batch_size = deepfm_all_embeddings.shape[0]
        y_fm = self.lr(x[0]) + self.lr_dense_vec_weight(context_dense_vec) + self.fm(torch.stack(embedding_ls, dim=1))

        y_deep = self.deep_predict_layer(
            self.mlp_layers(
               torch.cat([deepfm_all_embeddings, context_dense_vec], dim=1)
            )
        )
        y = y_fm + y_deep
        return y.squeeze(-1)
    
    def train_step(self, x, y):
        output = self.predict(x)
        loss = self.loss_func(output, y)

        return loss
    
    def predict(self, x):
        logits = self.forward(x)
        return self.sigmoid(logits)
    

class DeepFM_aug_kd(DeepFM):
    def __init__(self, init_para) -> None:
        super().__init__(init_para)

        del self.mlp_layers

        size_list = [
            self.model_config['decoder_emb_dim']\
            + sum([ emb_tab_dict.weight.size(-1) for _, emb_tab_dict in self.emb_tab_dict.items()])\
            + sum([ emb_tab_dict.weight.size(-1) for name, emb_tab_dict in self.emb_tab_dict.items() if name.startswith('item')])
        ] + self.mlp_hidden_size

        self.mlp_layers = MLPLayers(size_list, self.dropout_prob)

        self.lr_dense_vec_weight = nn.Linear(self.model_config['cot_mapping_mlp_hidden_size'][-1], 1, bias=False)

    # @torchsnooper.snoop()
    def forward(self, x):

        embedding_ls, context_dense_vec, cot_reconstruct_loss = self.embed_orig_and_aug_data_fileds(x)

        deepfm_all_embeddings = torch.cat(embedding_ls, dim=1) # batch_size, total_dim
        # batch_size = deepfm_all_embeddings.shape[0]
        y_fm = self.lr(x[0]) + self.lr_dense_vec_weight(context_dense_vec) + self.fm(torch.stack(embedding_ls, dim=1))

        y_deep = self.deep_predict_layer(
            self.mlp_layers(
               torch.cat([deepfm_all_embeddings, context_dense_vec], dim=1)
            )
        )
        y = y_fm + y_deep
        return y.squeeze(-1), cot_reconstruct_loss
    
    def train_step(self, x, y, llm_score):
        output, cot_reconstruct_loss = self.predict(x)
        
        distill_loss = self._add_knowledge_distillation_loss(llm_score, output)

        loss = self.loss_func(output, y) + self.model_config['reconstruct_weight']*cot_reconstruct_loss\
                                         + self.model_config['distill_weight']*distill_loss
        return loss
    
    def predict(self, x):
        logits, cot_reconstruct_loss = self.forward(x)
        return self.sigmoid(logits), cot_reconstruct_loss
    
class DeepFM_kd(DeepFM):
    def __init__(self, init_para) -> None:
        super().__init__(init_para)

    def train_step(self, x, y, llm_score):
        output = self.predict(x)
        
        distill_loss = self._add_knowledge_distillation_loss(llm_score, output)

        loss = self.loss_func(output, y) + self.model_config['distill_weight']*distill_loss
        
        return loss
    
class DeepFM_kar(DeepFM):
    def __init__(self, init_para) -> None:
        super().__init__(init_para)

        del self.mlp_layers

        size_list = [
            self.model_config['convert_arch'][-1] * 2\
            + sum([ emb_tab_dict.weight.size(-1) for _, emb_tab_dict in self.emb_tab_dict.items()])\
            + sum([ emb_tab_dict.weight.size(-1) for name, emb_tab_dict in self.emb_tab_dict.items() if name.startswith('item')])
        ] + self.mlp_hidden_size

        self.mlp_layers = MLPLayers(size_list, self.dropout_prob)

        self.lr_dense_vec_weight = nn.Linear(self.model_config['convert_arch'][-1] * 2, 1, bias=False)

        from .layers import ConvertNet

        self.convert_module = ConvertNet(self.model_config['export_num'], 
                                         self.model_config['specific_export_num'],
                                         self.model_config['convert_arch'],
                                         self.model_config['inp_dim'],
                                         self.model_config['dropout'] )   

    def forward(self, x):

        x, dense_vec = x

        embedding_ls = self.embed_input_fields(x)
        
        dense_vec = self.convert_module(dense_vec)

        deepfm_all_embeddings = torch.cat(embedding_ls, dim=1) # batch_size, total_dim
        # batch_size = deepfm_all_embeddings.shape[0]
        y_fm = self.lr(x) + self.lr_dense_vec_weight(dense_vec) + self.fm(torch.stack(embedding_ls, dim=1))

        y_deep = self.deep_predict_layer(
            self.mlp_layers(
               torch.cat([deepfm_all_embeddings, dense_vec], dim=1)
            )
        )
        y = y_fm + y_deep
        return y.squeeze(-1)

    def train_step(self, x, y):
        output = self.predict(x)
        loss = self.loss_func(output, y) 

        return loss
    
    def predict(self, x):
        logits = self.forward(x)
        return self.sigmoid(logits)


