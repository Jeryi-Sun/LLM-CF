import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, constant_

from .DeepFM import MLPLayers
from .base import BaseModel

  
class AutoInt(BaseModel):
    """AutoInt is a novel CTR prediction model based on self-attention mechanism,
    which can automatically learn high-order feature interactions in an explicit fashion.

    """

    def __init__(self, init_para):
        super(AutoInt, self).__init__(init_para)

        # load parameters info
        self.attention_size = self.model_config["attention_size"]
        self.dropout_probs = self.model_config["dropout_probs"]
        self.n_layers = self.model_config["n_layers"]
        self.num_heads = self.model_config["num_heads"]
        self.mlp_hidden_size = self.model_config["mlp_hidden_size"]
        self.has_residual = self.model_config["has_residual"]
        self.embedding_size = self.model_config['embedding_size']
        self.num_feature_field = self.model_config['num_feat_field']

        # define layers and loss
        self.att_embedding = nn.Linear(self.embedding_size, self.attention_size)
        self.embed_output_dim = self.num_feature_field * self.embedding_size
        self.atten_output_dim = self.num_feature_field * self.attention_size
        size_list = [self.embed_output_dim] + self.mlp_hidden_size
        self.mlp_layers = MLPLayers(size_list, dropout=self.dropout_probs)
        # multi-head self-attention network
        self.self_attns = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    self.attention_size, self.num_heads, dropout=self.dropout_probs
                )
                for _ in range(self.n_layers)
            ]
        )
        self.attn_fc = torch.nn.Linear(self.atten_output_dim, 1)
        self.deep_predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)
        if self.has_residual:
            self.v_res_embedding = torch.nn.Linear(
                self.embedding_size, self.attention_size
            )

        self.dropout_layer = nn.Dropout(p=self.dropout_probs)
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

    def autoint_layer(self, infeature):
        """Get the attention-based feature interaction score

        Args:
            infeature (torch.FloatTensor): input feature embedding tensor. shape of[batch_size,field_size,embed_dim].

        Returns:
            torch.FloatTensor: Result of score. shape of [batch_size,1] .
        """

        att_infeature = self.att_embedding(infeature)
        cross_term = att_infeature.transpose(0, 1)
        for self_attn in self.self_attns:
            cross_term, _ = self_attn(cross_term, cross_term, cross_term)
        cross_term = cross_term.transpose(0, 1)
        # Residual connection
        if self.has_residual:
            v_res = self.v_res_embedding(infeature)
            cross_term += v_res
        # Interacting layer
        cross_term = F.relu(cross_term).contiguous().view(-1, self.atten_output_dim)
        batch_size = infeature.shape[0]
        att_output = self.attn_fc(cross_term) + self.deep_predict_layer(
            self.mlp_layers(infeature.view(batch_size, -1))
        )
        return att_output

    def forward(self, interaction):
        autoint_all_embeddings = torch.stack( 
            self.embed_input_fields(interaction), dim=1
        )  # [batch_size, num_field, embed_dim]
        output = self.autoint_layer(
            autoint_all_embeddings
        )
        return output.squeeze(1)


    def train_step(self, x, y):
        loss = self.loss_func( self.predict(x), y)

        return loss
    
    def predict(self, x):

        return self.sigmoid(self.forward(x))

class AutoInt_aug(AutoInt):
    def __init__(self, init_para):

        init_para['model_config']['num_feat_field'] += 1

        super().__init__(init_para)

    def forward(self, x):
        embedding_ls, context_dense_vec, cot_reconstruct_loss = self.embed_orig_and_aug_data_fileds(x)

        embedding_ls.append(context_dense_vec) # add new dense feature

        autoint_all_embeddings = torch.stack( 
            embedding_ls, dim=1
        )  # [batch_size, num_field+1, embed_dim]
        output = self.autoint_layer(
            autoint_all_embeddings
        )
        return output.squeeze(1), cot_reconstruct_loss


    def train_step(self, x, y):
        output, cot_reconstruct_loss = self.predict(x)
        loss = self.loss_func(output, y) + self.model_config['reconstruct_weight']*cot_reconstruct_loss

        return loss
    
    def predict(self, x):
        logits, cot_reconstruct_loss = self.forward(x)
        return self.sigmoid(logits), cot_reconstruct_loss


class AutoInt_aug_wocot(AutoInt):
    def __init__(self, init_para):

        init_para['model_config']['num_feat_field'] += 1

        super().__init__(init_para)

    def forward(self, x):
        embedding_ls, context_dense_vec = self.embed_orig_and_aug_data_wocot_fileds(x)

        embedding_ls.append(context_dense_vec) # add new dense feature

        autoint_all_embeddings = torch.stack( 
            embedding_ls, dim=1
        )  # [batch_size, num_field+1, embed_dim]
        output = self.autoint_layer(
            autoint_all_embeddings
        )
        return output.squeeze(1)


    def train_step(self, x, y):
        output = self.predict(x)
        loss = self.loss_func(output, y)

        return loss
    
    def predict(self, x):
        logits = self.forward(x)
        return self.sigmoid(logits)

class AutoInt_aug_mean(AutoInt):
    def __init__(self, init_para):

        init_para['model_config']['num_feat_field'] += 1

        super().__init__(init_para)

    def forward(self, x):
        embedding_ls, context_dense_vec = self.embed_orig_and_aug_data_mean_fileds(x)

        embedding_ls.append(context_dense_vec) # add new dense feature

        autoint_all_embeddings = torch.stack( 
            embedding_ls, dim=1
        )  # [batch_size, num_field+1, embed_dim]
        output = self.autoint_layer(
            autoint_all_embeddings
        )
        return output.squeeze(1)


    def train_step(self, x, y):
        output = self.predict(x)
        loss = self.loss_func(output, y)

        return loss
    
    def predict(self, x):
        logits = self.forward(x)
        return self.sigmoid(logits)
    
class AutoInt_aug_kd(AutoInt):

    def __init__(self, init_para):

        init_para['model_config']['num_feat_field'] += 1

        super().__init__(init_para)

    def forward(self, x):
        embedding_ls, context_dense_vec, cot_reconstruct_loss = self.embed_orig_and_aug_data_fileds(x)

        embedding_ls.append(context_dense_vec) # add new dense feature

        autoint_all_embeddings = torch.stack( 
            embedding_ls, dim=1
        )  # [batch_size, num_field+1, embed_dim]
        output = self.autoint_layer(
            autoint_all_embeddings
        )
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


class AutoInt_kd(AutoInt):
    def __init__(self, init_para):


        super().__init__(init_para)

    def train_step(self, x, y, llm_score):
        output = self.predict(x)
        
        distill_loss = self._add_knowledge_distillation_loss(llm_score, output)

        loss = self.loss_func(output, y) + self.model_config['distill_weight']*distill_loss
        
        return loss

class AutoInt_kar(AutoInt):
    def __init__(self, init_para):
        init_para['model_config']['num_feat_field'] += 1

        super().__init__(init_para)

        from .layers import ConvertNet

        self.convert_module = ConvertNet(self.model_config['export_num'], 
                                         self.model_config['specific_export_num'],
                                         self.model_config['convert_arch'],
                                         self.model_config['inp_dim'],
                                         self.model_config['dropout'] )      

    def forward(self, interaction):

        interaction, dense_vec = interaction

        dense_vec = self.convert_module(dense_vec)

        embedding_ls = self.embed_input_fields(interaction)

        embedding_ls.append(dense_vec) # add new dense feature

        autoint_all_embeddings = torch.stack( 
            embedding_ls, dim=1
        )  # [batch_size, num_field+1, embed_dim]
        output = self.autoint_layer(
            autoint_all_embeddings
        )
        return output.squeeze(1)


    def train_step(self, x, y):
        output = self.predict(x)
        loss = self.loss_func(output, y) 

        return loss
    
    def predict(self, x):
        logits = self.forward(x)
        return self.sigmoid(logits)
