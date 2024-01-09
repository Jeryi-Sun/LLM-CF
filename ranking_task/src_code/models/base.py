import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

import os, logging
from tqdm import tqdm

import torchsnooper
from .transformer_decoder import ModelArgs, Transformer

class GPTDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, batch_first=True):
        super(GPTDecoderLayer, self).__init__()
        self.batch_first = batch_first
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=batch_first)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, src):
        if not self.batch_first:
            src = src.transpose(0, 1)  # Convert to (sequence_length, batch_size, feature_number)

        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(nn.functional.relu(self.linear1(src2))))
        src = src + self.dropout2(src2)

        if not self.batch_first:
            src = src.transpose(0, 1)  # Convert back to (batch_size, sequence_length, feature_number)

        return src

class BaseModel(nn.Module):
    def __init__(self, init_para) -> None:
        super().__init__()

        self.wksp = init_para['workspace']
        self.name = init_para['name']
        self.device = init_para['device']
        self.patience = init_para['patience'] # patience for early stop
        self.loss_func = init_para['loss_function']
        self.eval_metrics = init_para['eval_metric_ls']
        self.metric_names = init_para['metric_name_ls']
        self.model_config = init_para['model_config']
        self.input_data_ls = init_para['input_data']
        self.register_embedding_tables()
        self.set_device()


        '''if use augmentation data'''
        if 'aug' in self.name:
            self.aug_input_data_ls = init_para['input_aug_data']
            self.num_aug_samples = init_para['num_aug']
            from .DeepFM import MLPLayers
            self.bge_emb_mapping = MLPLayers(
                self.model_config['cot_mapping_mlp_hidden_size'],
                last_activation=False
            )
            self.cot_emb_reconstruct_mapping = MLPLayers(
                self.model_config['cot_mapping_mlp_hidden_size'][::-1],
                last_activation=False
            )
            params = ModelArgs()
            params.dim = self.model_config['decoder_emb_dim']
            params.n_heads = self.model_config['decoder_nhead']
            params.dim_feedforward = 2*self.model_config['decoder_emb_dim']
            self.context_transformer = Transformer(params=params)
            self.history_mlp = MLPLayers(
                self.model_config['ICL_history_mlp'],
                last_activation=False
            )


    def register_embedding_tables(self):
        self.emb_tab_dict = nn.ModuleDict()
        for feat in self.input_data_ls:
            if feat.name.startswith('history'):
                continue
            self.emb_tab_dict[f'{feat.name}_emb_tab'] = \
                    nn.Embedding(num_embeddings=feat.vocabulary_size, embedding_dim=feat.embedding_dim, padding_idx=0)
            # print(f'{feat.name}_emb_tab')
    
    def embed_input_fields(self, input_data, input_data_name_ls=None, use_item_textual_data=False):
        emb_data_ls = []

        def mean_pooling(tensor_data, id_data, dim):
            mask = id_data != 0
            tensor_data = torch.sum(tensor_data*mask.unsqueeze(-1), dim=dim)
            tensor_data = tensor_data / (torch.sum(mask, dim=dim, keepdim=True)+1e-9) # 1e-9 to avoid dividing zero

            return tensor_data
        
        input_data_name_ls = self.input_data_ls if input_data_name_ls is None else input_data_name_ls

        for data_filed, data in zip(input_data_name_ls, input_data):
            if data_filed.name.startswith('history'):
                history_data = self.emb_tab_dict[f'{data_filed.name[len("history")+1:]}_emb_tab'](data)
                # mean pooling for history behaviors
                if len(data.size()) == 2:
                    history_data = mean_pooling(history_data, data, 1)
                elif len(data.size()) == 3:
                    history_data = mean_pooling(history_data.reshape(history_data.size(0), -1, history_data.size(-1)), 
                                                    data.reshape(data.size(0), -1), 1)
                else:
                    raise ValueError(f'wrong dimension of input data {data_filed.name} with size {data.size()}')
                emb_data_ls.append(history_data)
                if data_filed.name == 'history_item_id' and use_item_textual_data:
                    history_data = self.emb_tab_dict[f'item_text_id_emb_tab'](data)
                    history_data = mean_pooling(history_data, data, 1)
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
                if data_filed.name == 'item_id' and use_item_textual_data:
                    emb_data_ls.append(
                        self.emb_tab_dict[f'item_text_id_emb_tab'](data)
                    ) 

        return emb_data_ls

            
    def embed_input_fields_for_attention_pooling(self, input_data):

        def mean_pooling(tensor_data, id_data, dim):
            mask = id_data != 0
            tensor_data = torch.sum(tensor_data*mask.unsqueeze(-1), dim=dim)
            tensor_data = tensor_data / (torch.sum(mask, dim=dim, keepdim=True)+1e-9) # 1e-9 to avoid dividing zero

            return tensor_data

        history_emb_ls, item_emb_ls = [], []
        history_mask = None
        for data_filed, data in zip(self.input_data_ls, input_data):
            if data_filed.name.startswith('history'):
                if data_filed.name == 'history_item_id':
                    history_mask = data == 0.0
                history_data = self.emb_tab_dict[f'{data_filed.name[len("history")+1:]}_emb_tab'](data)
                if len(data.size()) == 2:
                    pass
                elif len(data.size()) == 3:
                    history_data = mean_pooling(history_data, data, 2)
                else:
                    raise ValueError(f'wrong dimension of input data {data_filed.name} with size {data.size()}')
                history_emb_ls.append(history_data)
                
            elif data_filed.name.startswith('item'): 
                
                if data_filed.name in ['item_cate_id']:
                    history_data = self.emb_tab_dict[f'{data_filed.name}_emb_tab'](data)
                    if len(data.size()) == 2:
                        history_data = mean_pooling(history_data, data, 1)
                else:
                    history_data = self.emb_tab_dict[f'{data_filed.name}_emb_tab'](data)
                        
                item_emb_ls.append(history_data)
            else:
                user_emb = self.emb_tab_dict[f'{data_filed.name}_emb_tab'](data)
                
        return user_emb, torch.cat(item_emb_ls, dim=-1), torch.cat(history_emb_ls, dim=-1), history_mask


    def set_emb_tab_for_aug_data(self, cot_emb: torch.tensor, item_text_emb_dict: dict):
        
        '''BGE embeddings of item textual feature'''
        item_text_emb = torch.zeros((len(item_text_emb_dict)+1, 768)) # zero for padding
        for item_id, text_emb in item_text_emb_dict.items():
            item_text_emb[int(item_id)] = torch.tensor(text_emb, dtype=torch.float32)
        
        self.emb_tab_dict['item_text_id_emb_tab'] = nn.Embedding.from_pretrained(
            embeddings=item_text_emb, freeze=True
        ).to(self.device)

        '''BGE embeddings of COT(chain of thoughts)'''
        cot_emb = torch.nn.functional.normalize(cot_emb, p=2, dim=1)

        self.emb_tab_dict['cot_id_emb_tab'] = nn.Embedding.from_pretrained(
            embeddings=cot_emb, freeze=True
        ).to(self.device)

        self.emb_tab_dict['label_id_emb_tab'] = nn.Embedding(
            2, self.model_config['aug_label_id_dim']
        ).to(self.device)
      
    def embed_orig_and_aug_data_fileds(self, x):
        '''embed both orignal and augmentation data input'''
        orig_data_input, aug_data_input = x

        orig_data_emb = self.embed_input_fields(orig_data_input)
        aug_data_emb = [
            self.embed_input_fields(aug_data, self.aug_input_data_ls, True) 
            for index, aug_data in enumerate(aug_data_input) if index < self.num_aug_samples
        ]
        '''
        each augmented data sample in aug_data_emb:
            * user_id_emb
            * item_id_emb
            * item_textual_emb : if use textual embedding.  768 dim BGE embedding
            * item_brand_id_emb
            * item_category_id_emb
            * history_item_id_emb
            * history_item_texutual_id_emb: if use textual embedding.  768 dim BGE embedding
            * history_item_brand_id_emb
            * history_item_category_id_emb
            * cot_emb: 768 dim BGE embedding
            * label_emb
        '''

        orig_data_w_text = self.embed_input_fields(orig_data_input, use_item_textual_data=True)
        orig_data_w_text[2] = self.bge_emb_mapping(orig_data_w_text[2])
        orig_data_w_text[6] = self.bge_emb_mapping(orig_data_w_text[6])

        # 这边加一个 MLP 去映射到我们的大小上
        cot_data_emb_raw = [data[-2] for data in aug_data_emb]
        for data in aug_data_emb:
            data[-2] = self.bge_emb_mapping(data[-2]) # mapping cot emb
            data[2] = self.bge_emb_mapping(data[2]) # mapping item_textual emb
            data[6] = self.bge_emb_mapping(data[6]) # mapping history_item textual emb

        final_aug_data_emb = [ [self.history_mlp(torch.cat(data[:-2], dim=-1))] + data[-2:] for data in aug_data_emb]

        

        ICL_data_emb = torch.cat(
            [
               torch.stack(data, dim=1) for data in final_aug_data_emb
            ], dim=1
        )
        # if dont use textual data
        # ICL_data_emb = torch.cat(
        #     [ICL_data_emb, self.history_mlp(torch.cat(orig_data_emb, dim=1)).unsqueeze(1)],
        #     dim=1)
        # else if use textual data
        ICL_data_emb = torch.cat(
            [ICL_data_emb, self.history_mlp(torch.cat(orig_data_w_text, dim=1)).unsqueeze(1)],
            dim=1)
             
        ICL_hidden_states = self.context_transformer(ICL_data_emb)

        # 增加对于 cot 的学习能力
        positions, pos = [], 0
        for _ in range(self.num_aug_samples):
            positions.append(pos)
            pos += 3
        cot_outputs = self.cot_emb_reconstruct_mapping(ICL_hidden_states[:, positions, :])
        cot_inputs = torch.stack(cot_data_emb_raw, dim=1)

        # Calculate cosine similarity
        cosine_sim = nn.CosineSimilarity(dim=-1)
        similarity = cosine_sim(cot_outputs, cot_inputs)

        # Calculate loss as 1 - mean cosine similarity
        cot_reconstruct_loss = 1 - similarity.mean()

        ICL_feat = ICL_hidden_states[:, -1, :]

        return orig_data_emb, ICL_feat, cot_reconstruct_loss

    def embed_orig_and_aug_data_wocot_fileds(self, x):
        '''embed both orignal and augmentation data input'''
        orig_data_input, aug_data_input = x

        orig_data_emb = self.embed_input_fields(orig_data_input)
        aug_data_emb = [
            self.embed_input_fields(aug_data, self.aug_input_data_ls, True) 
            for index, aug_data in enumerate(aug_data_input) if index < self.num_aug_samples
        ]
        '''
        each augmented data sample in aug_data_emb:
            * user_id_emb
            * item_id_emb
            * item_textual_emb : if use textual embedding.  768 dim BGE embedding
            * item_brand_id_emb
            * item_category_id_emb
            * history_item_id_emb
            * history_item_texutual_id_emb: if use textual embedding.  768 dim BGE embedding
            * history_item_brand_id_emb
            * history_item_category_id_emb
            * cot_emb: 768 dim BGE embedding
            * label_emb
        '''

        orig_data_w_text = self.embed_input_fields(orig_data_input, use_item_textual_data=True)
        orig_data_w_text[2] = self.bge_emb_mapping(orig_data_w_text[2])
        orig_data_w_text[6] = self.bge_emb_mapping(orig_data_w_text[6])

        # 这边加一个 MLP 去映射到我们的大小上
        for data in aug_data_emb:
            data[-2] = self.bge_emb_mapping(data[-2]) # mapping cot emb
            data[2] = self.bge_emb_mapping(data[2]) # mapping item_textual emb
            data[6] = self.bge_emb_mapping(data[6]) # mapping history_item textual emb
        aug_data_emb_wocot = []
        for data in aug_data_emb:
            aug_data_emb_wocot.append(data[:-2]+data[-1:])
        final_aug_data_emb = [ [self.history_mlp(torch.cat(data[:-1], dim=-1))] + data[-1:] for data in aug_data_emb_wocot]

        

        ICL_data_emb = torch.cat(
            [
               torch.stack(data, dim=1) for data in final_aug_data_emb
            ], dim=1
        )
        # if dont use textual data
        # ICL_data_emb = torch.cat(
        #     [ICL_data_emb, self.history_mlp(torch.cat(orig_data_emb, dim=1)).unsqueeze(1)],
        #     dim=1)
        # else if use textual data
        ICL_data_emb = torch.cat(
            [ICL_data_emb, self.history_mlp(torch.cat(orig_data_w_text, dim=1)).unsqueeze(1)],
            dim=1)
             
        ICL_hidden_states = self.context_transformer(ICL_data_emb)

        ICL_feat = ICL_hidden_states[:, -1, :]

        return orig_data_emb, ICL_feat


    def embed_orig_and_aug_data_mean_fileds(self, x):
        '''embed both orignal and augmentation data input'''
        orig_data_input, aug_data_input = x

        orig_data_emb = self.embed_input_fields(orig_data_input)
        aug_data_emb = [
            self.embed_input_fields(aug_data, self.aug_input_data_ls, True) 
            for index, aug_data in enumerate(aug_data_input) if index < self.num_aug_samples
        ]
        '''
        each augmented data sample in aug_data_emb:
            * user_id_emb
            * item_id_emb
            * item_textual_emb : if use textual embedding.  768 dim BGE embedding
            * item_brand_id_emb
            * item_category_id_emb
            * history_item_id_emb
            * history_item_texutual_id_emb: if use textual embedding.  768 dim BGE embedding
            * history_item_brand_id_emb
            * history_item_category_id_emb
            * cot_emb: 768 dim BGE embedding
            * label_emb
        '''

        orig_data_w_text = self.embed_input_fields(orig_data_input, use_item_textual_data=True)
        orig_data_w_text[2] = self.bge_emb_mapping(orig_data_w_text[2])
        orig_data_w_text[6] = self.bge_emb_mapping(orig_data_w_text[6])

        # 这边加一个 MLP 去映射到我们的大小上
        cot_data_emb_raw = [data[-2] for data in aug_data_emb]
        for data in aug_data_emb:
            data[-2] = self.bge_emb_mapping(data[-2]) # mapping cot emb
            data[2] = self.bge_emb_mapping(data[2]) # mapping item_textual emb
            data[6] = self.bge_emb_mapping(data[6]) # mapping history_item textual emb

        final_aug_data_emb = [ [self.history_mlp(torch.cat(data[:-2], dim=-1))] + data[-2:] for data in aug_data_emb]

        

        ICL_data_emb = torch.cat(
            [
               torch.stack(data, dim=1) for data in final_aug_data_emb
            ], dim=1
        )
        # if dont use textual data
        # ICL_data_emb = torch.cat(
        #     [ICL_data_emb, self.history_mlp(torch.cat(orig_data_emb, dim=1)).unsqueeze(1)],
        #     dim=1)
        # else if use textual data
        ICL_data_emb = torch.cat(
            [ICL_data_emb, self.history_mlp(torch.cat(orig_data_w_text, dim=1)).unsqueeze(1)],
            dim=1)
             
        ICL_hidden_states = ICL_data_emb


        # Calculate loss as 1 - mean cosine similarity

        ICL_feat = torch.mean(ICL_hidden_states, dim=1)

        return orig_data_emb, ICL_feat




    def set_device(self):
        if self.device == 'cpu':
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(self.device)

    def save_ckpt(self):

        model_path = os.path.join(self.wksp, 'best.pth')
        torch.save(self.state_dict(), model_path)

    def load_ckpt(self, assigned_path=None):

        ckpt_path = self.wksp
        model_path = None
        if assigned_path is not None:
            '''specific assigned path'''
            model_path = assigned_path
        else:
            '''default path'''   
            model_path = os.path.join(ckpt_path, 'best.pth')
        self.load_state_dict(torch.load(model_path, map_location=self.device))


    def fit(self, epochs, training_dataloader, validation_dataloader, optimizer):

        self.optimizer = optimizer

        best_metric, num_stop_increasing_epochs = 0, 0
        
        for epoch in range(epochs):

            if self.name[-3:] == 'kar':
                self.kar_train_one_epoch(epoch, training_dataloader)
                watch_metric_value = self.kar_test(epoch, validation_dataloader)
            else:
                self.train_one_epoch(epoch, training_dataloader)
                watch_metric_value = self.test(epoch, validation_dataloader)
            if watch_metric_value > best_metric:
                self.save_ckpt()
                logging.info('save ckpt at epoch {}'.format(epoch))
                print('save ckpt at epoch {}'.format(epoch))
                best_metric = watch_metric_value
                num_stop_increasing_epochs = 0
            else:
                num_stop_increasing_epochs += 1
                if num_stop_increasing_epochs >= self.patience:
                    logging.info('early stop at epoch {}'.format(epoch))
                    print('early stop at epoch {}'.format(epoch))
                    break

    def _add_knowledge_distillation_loss(self, teacher_scores, student_scores):
        '''
        teacher_scores: [B, 2], logits(label 1, label 0)
        student_scores: [B]. logits of label 1
        '''

        teacher_scores = F.softmax(teacher_scores / self.model_config['teacher_temp'], dim=-1)
        teacher_scores = teacher_scores[:, 0]

                       # binary cross entropy loss with soft labels
        distill_loss = self.loss_func(student_scores, teacher_scores) 

        # std_scores_0 = 1 - student_scores
        # student_scores = torch.stack([student_scores, std_scores_0], dim=1)

        # # student_scores = F.log_softmax(student_scores / self.model_config['student_temp'], dim=-1)
        # student_scores = student_scores.log()
        # teacher_scores = F.softmax(teacher_scores / self.model_config['teacher_temp'], dim=-1)
        # distill_loss = F.kl_div(student_scores, teacher_scores, reduction='batchmean')

        return distill_loss


    def train_one_epoch(self, epoch, training_dataloader):
        
        self.train()
        tqdm_ = tqdm(iterable=training_dataloader, mininterval=1, ncols=100)

        epoch_loss = 0.0

        for step, sample in enumerate(tqdm_):
            
            self.optimizer.zero_grad()

            x, y, aug_x = sample['x'], sample['y'], sample['aug_x']
            x = tuple(data.to(self.device) for data in x)
            aug_x = [tuple(data.to(self.device) for data in i_aug_x) for i_aug_x in aug_x]
            y = y.to(self.device)

            do_distill = 'distill_weight' in self.model_config
            llm_score = sample['llm_score'].to(self.device) if do_distill else None

            if 'aug' in self.name:
                '''for ctr models with augmented data'''
                if do_distill:
                    loss = self.train_step((x, aug_x), y, llm_score)
                else:
                    loss = self.train_step((x, aug_x), y)
            else:
                '''for ctr models'''
                if do_distill:
                    loss = self.train_step(x, y, llm_score)
                else:
                    loss = self.train_step(x, y)

            if torch.isnan(loss):
                raise ValueError('loss is NaN!')

            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            
            if step % (training_dataloader.__len__() // 50) == 0 and step != 0:
                tqdm_.set_description(
                        "epoch {:d} , step {:d} , loss: {:.4f}".format(epoch, step+1, epoch_loss / (step+1)))
                
        logging.info('epoch {}:  loss = {}'.format(epoch, epoch_loss / (step+1)))

    def kar_train_one_epoch(self, epoch, training_dataloader):
        
        self.train()
        tqdm_ = tqdm(iterable=training_dataloader, mininterval=1, ncols=100)

        epoch_loss = 0.0

        for step, sample in enumerate(tqdm_):
            
            self.optimizer.zero_grad()

            x, y, kar_dense_vec = sample['x'], sample['y'], sample['kar_data']
            x = tuple(data.to(self.device) for data in x)
            kar_dense_vec = list(data.to(self.device) for data in kar_dense_vec)
            y = y.to(self.device)

            do_distill = 'distill_weight' in self.model_config
            llm_score = sample['llm_score'].to(self.device) if do_distill else None

            loss = self.train_step((x, kar_dense_vec), y)

            if torch.isnan(loss):
                raise ValueError('loss is NaN!')

            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            
            if step % (training_dataloader.__len__() // 50) == 0 and step != 0:
                tqdm_.set_description(
                        "epoch {:d} , step {:d} , loss: {:.4f}".format(epoch, step+1, epoch_loss / (step+1)))
                
        logging.info('epoch {}:  loss = {}'.format(epoch, epoch_loss / (step+1)))


    def train_step(self, x, y):

        raise NotImplementedError

    @torch.no_grad()
    def test(self, epoch, test_dataloader):
        self.eval()

        pred_logits, labels = [], []

        for batch_data in tqdm(iterable=test_dataloader, mininterval=1, ncols=100):
            
            x, y, aug_x = batch_data['x'], batch_data['y'], batch_data['aug_x']
            x = tuple(data.to(self.device) for data in x)
            aug_x = [tuple(data.to(self.device) for data in i_aug_x) for i_aug_x in aug_x]
            
            step_label = y
            if 'aug' in self.name:
                '''for ctr models with augmented data'''
                step_pred_logits = self.predict((x, aug_x)) #B
                if isinstance(step_pred_logits, tuple):
                    step_pred_logits = step_pred_logits[0]
            else:
                '''for ctr models'''
                step_pred_logits = self.predict(x) #B

            pred_logits.extend(step_pred_logits.detach().cpu().tolist())
            labels.extend(step_label.detach().cpu().tolist())


        metrics = [eval(labels, pred_logits) for eval in self.eval_metrics]

        logging.info(f'results at epoch {epoch}')
        print(f'results at epoch {epoch}')
        for metric, name in zip(metrics, self.metric_names):
            logging.info(f"     {name} : {metric}")
            print(f"      {name} : {metric}")

        named_metrics = {name : metric for metric, name in zip(metrics, self.metric_names)}

        return named_metrics['auc']
            
    @torch.no_grad()
    def kar_test(self, epoch, test_dataloader):
        self.eval()

        pred_logits, labels = [], []

        for batch_data in tqdm(iterable=test_dataloader, mininterval=1, ncols=100):
            
            x, y, kar_dense_vec = batch_data['x'], batch_data['y'], batch_data['kar_data']
            x = tuple(data.to(self.device) for data in x)
            kar_dense_vec = list(data.to(self.device) for data in kar_dense_vec)
            
            step_label = y
            '''for ctr models'''
            step_pred_logits = self.predict((x, kar_dense_vec)) #B

            pred_logits.extend(step_pred_logits.detach().cpu().tolist())
            labels.extend(step_label.detach().cpu().tolist())


        metrics = [eval(labels, pred_logits) for eval in self.eval_metrics]

        logging.info(f'results at epoch {epoch}')
        print(f'results at epoch {epoch}')
        for metric, name in zip(metrics, self.metric_names):
            logging.info(f"     {name} : {metric}")
            print(f"      {name} : {metric}")

        named_metrics = {name : metric for metric, name in zip(metrics, self.metric_names)}

        return named_metrics['auc']
    