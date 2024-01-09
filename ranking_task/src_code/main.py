import argparse

import datetime, logging, yaml, random, os, pickle
from collections import namedtuple

import torch
from torch.utils.data import dataloader, dataset
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

import models

def setup_seed(seed):
    '''setting random seeds'''
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed) 
setup_seed(2023)

parser = argparse.ArgumentParser()

parser.add_argument('--name', type=str, help='experiment name', default='default')
parser.add_argument('--workspace', type=str, default='./workspace')

parser.add_argument('--dataset', type=str, default='beauty')
parser.add_argument('--use_cpu', dest='use_gpu', action='store_false')
parser.set_defaults(use_gpu=True)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--num_workers', type=int, default=16, help='num of workers used for multi-processing data loading')
parser.add_argument('--model', type=str, help='which model to use', default='DeepFM')
parser.add_argument('--batch_size', type=int, help='training batch_size', default=128)
parser.add_argument('--test_batch_size', type=int, help='testing batch_size', default=256)
parser.add_argument('--new_config', type=str, help='update model config', default='')

args = parser.parse_args()

# update hyper-paremeters w.r.t dataset
# dataset_paras = {
#     'beauty': {'batch_size': 256, 'test_batch_size': 1024, 'epochs': 50},
# }
# if args.batch_size == 0 : args.batch_size = dataset_paras[args.dataset]['batch_size'] 
# if args.test_batch_size == 0: args.test_batch_size = dataset_paras[args.dataset]['test_batch_size'] 
# if args.epochs == 0: args.epochs = dataset_paras[args.dataset]['epochs'] 

new_config = {} if args.new_config == '' else eval(args.new_config)
cur_hyper_paras = [(k ,v) for k,v in new_config.items()]
new_name = f'{args.name}'
for k,v in cur_hyper_paras:
    new_name += f'_{k}_{v}'
args.name = new_name


## update model config
dataset_ = args.dataset

model_config = f'config/{dataset_}/{args.model}.yaml'
model_config = yaml.load(
    open(model_config, 'r'), Loader=yaml.FullLoader
)
for key, value in new_config.items():
    '''check new config'''
    if key not in model_config:
        raise NameError(f'wrong new config key: {key}')
model_config.update(new_config)

## create workspace
def init_workspace_and_logging():
    workspace_par_dir = args.workspace
    if not os.path.exists(workspace_par_dir):
        os.mkdir(workspace_par_dir)
    wksp_path = os.path.join(workspace_par_dir,
                    f'{args.name}_{str(datetime.datetime.now().month)}_{str(datetime.datetime.now().day)}_{str(datetime.datetime.now().hour)}'
                             )
    if not os.path.exists(wksp_path):
        os.mkdir(wksp_path)

    # set log file path
    log_file_name = os.path.join(wksp_path, f'{args.name}.log')
    logging.basicConfig(format='%(asctime)s - %(message)s',
                level=logging.INFO, filename=log_file_name, filemode='w')

    logging.info('Configs:')
    for flag, value in model_config.items():
        logging.info('{}: {}'.format(flag, value))

    return wksp_path

wksp_path = init_workspace_and_logging()

## load dataset
def load_dataset_and_post_process():
    
    data_par_path = f'../data/{dataset_}/'

    def load_json(file_name):
            import json
            with open(file_name, 'r') as f:
                record = json.loads(f.read())
            return record
        
    datamaps = load_json(os.path.join(data_par_path, 'new_datamaps.json'))
    item2attributes = load_json(os.path.join(data_par_path, 'new_item_attributes.json'))

    if not os.path.exists(os.path.join(data_par_path, 'dataset', 'training_w_attr.tsv')):
        print('start loading data')
        training_inter = pd.read_csv(os.path.join(data_par_path, 'dataset',
                                                'training.tsv'),
                                        sep = '\t')
        valid_inter = pd.read_csv(os.path.join(data_par_path, 'dataset',
                                            'validation.tsv'),
                                    sep = '\t')
        test_inter = pd.read_csv(os.path.join(data_par_path, 'dataset',
                                            'test.tsv'),
                                    sep = '\t')
        
        def pad_and_truncate(seq: list):
            seq = eval(seq)

            ret = seq[-model_config['max_len']:]
            if len(ret) < model_config['max_len']:
                ret.extend( (model_config['max_len'] - len(ret)) * ['0'] )
            return ret
        
        # truncate user history
        training_inter['history'] = training_inter['history'].apply(
            pad_and_truncate
        )
        valid_inter['history'] = valid_inter['history'].apply(
            pad_and_truncate
        )
        test_inter['history'] = test_inter['history'].apply(
            pad_and_truncate
        )

        

        def post_process_item_attributes(inter):
            uid, history, item, label = inter['user_id'].values, \
                inter['history'].values, inter['item'].values, inter['label'].values
            
            uid, item, label = uid.astype(np.int32), item.astype(np.int32), label.astype(np.float32)
            item_id = item
            item_brand_id = np.array([
                datamaps['brand2id'][item2attributes[str(iid)]['brand'][0]]
                    if iid!=0 and item2attributes[str(iid)]['brand'].__len__()>0 
                    else 0 
                for iid in item
            ])

            cate_num = [values['category'].__len__() for _, values in item2attributes.items()]
            cate_num_max = max(cate_num)
            item_cate_id = []
            for iid in item_id:
                if iid == 0:
                    item_cate_id.append([0] * cate_num_max)
                else:
                    temp_cate = []
                    for cate in item2attributes[str(iid)]['category']:
                        temp_cate.append(datamaps['cate2id'][cate])
                    if len(temp_cate) < cate_num_max:
                        temp_cate += [0] * (cate_num_max - len(temp_cate))
                    item_cate_id.append(temp_cate)
            item_cate_id = np.array(item_cate_id)


            history_item_id = np.array([np.array(seq).astype(np.int32) for seq in history])
            history_brand_id = []
            for seq in history_item_id:
                temp = []
                for iid in seq:
                    if iid != 0 and item2attributes[str(iid)]['brand'].__len__()>0:
                        temp.append(
                            datamaps['brand2id'][item2attributes[str(iid)]['brand'][0]]
                        )
                    else:
                        temp.append(0)
                history_brand_id.append(temp)
            history_brand_id = np.array(history_brand_id)

            
            history_cate_id = []
            for seq in history_item_id:
                temp = []
                for iid in seq:
                    if iid == 0:
                        temp.append([0] * cate_num_max)
                    else:
                        temp_cate = []
                        for cate in item2attributes[str(iid)]['category']:
                            temp_cate.append(datamaps['cate2id'][cate])
                        if len(temp_cate) < cate_num_max:
                            temp_cate += [0] * (cate_num_max - len(temp_cate))
                        temp.append(temp_cate)
                history_cate_id.append(temp)
                
            history_cate_id = np.array(history_cate_id)

            tensor_dataset = dataset.TensorDataset(
                torch.LongTensor(uid), 
                torch.LongTensor(item_id), torch.LongTensor(item_brand_id), torch.LongTensor(item_cate_id),
                torch.LongTensor(history_item_id), torch.LongTensor(history_brand_id), torch.LongTensor(history_cate_id),
                torch.tensor(label, dtype=torch.float32)
            )

            return tensor_dataset
            
        training_dataset = post_process_item_attributes(training_inter)
        valid_dataset = post_process_item_attributes(valid_inter)
        test_dataset = post_process_item_attributes(test_inter)

        with open(os.path.join(data_par_path, 'dataset', 'training_w_attr.tsv'), 'wb') as f:
            pickle.dump(training_dataset, f)
        with open(os.path.join(data_par_path, 'dataset', 'valid_w_attr.tsv'), 'wb') as f:
            pickle.dump(valid_dataset, f)
        with open(os.path.join(data_par_path, 'dataset', 'test_w_attr.tsv'), 'wb') as f:
            pickle.dump(test_dataset, f)
        print('finish processing data')
        
    else:
        with open(os.path.join(data_par_path, 'dataset', 'training_w_attr.tsv'), 'rb') as f:
            training_dataset = pickle.load(f)
        with open(os.path.join(data_par_path, 'dataset', 'valid_w_attr.tsv'), 'rb') as f:
            valid_dataset = pickle.load(f)
        with open(os.path.join(data_par_path, 'dataset', 'test_w_attr.tsv'), 'rb') as f:
            test_dataset = pickle.load(f)
        

    SparseFeat = namedtuple('SparseFeat', ['name', 'vocabulary_size', 'embedding_dim'])
    VarLenSparseFeat = namedtuple('VarLenSparseFeat',
                                  ['name','sparsefeat', 'maxlen'])
    
    input_data = [
        SparseFeat(name='user_id', vocabulary_size=datamaps['user2id'].__len__()+1,
                   embedding_dim=model_config['user_id_dim']
                   ),
        SparseFeat(name='item_id', vocabulary_size=datamaps['item2id'].__len__()+1,
                   embedding_dim=model_config['item_id_dim']
                   ),
        SparseFeat(name='item_brand_id', vocabulary_size=datamaps['brand2id'].__len__()+1,
                   embedding_dim=model_config['item_brand_id_dim']
                   ),
        SparseFeat(name='item_cate_id', vocabulary_size=datamaps['cate2id'].__len__()+1,
                   embedding_dim=model_config['item_cate_id_dim']
                   ),
        VarLenSparseFeat(sparsefeat='item_id', maxlen=model_config['max_len'],
                         name='history_item_id'),
        VarLenSparseFeat(sparsefeat='item_brand_id', maxlen=model_config['max_len'],
                         name='history_item_brand_id'),
        VarLenSparseFeat(sparsefeat='item_cate_id', maxlen=model_config['max_len'],
                         name='history_item_cate_id'),
    ]

    GLOBAL_WORKER_ID = None
    GLOBAL_SEED = 1
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def worker_init_fn(worker_id):
        global GLOBAL_WORKER_ID
        GLOBAL_WORKER_ID = worker_id
        set_seed(GLOBAL_SEED + worker_id)

    def get_dataloader(data_set, bs, shuffle=False):
        return dataloader.DataLoader(  data_set, batch_size = bs,
                        pin_memory = True, 
                        worker_init_fn = worker_init_fn, 
                        num_workers = args.num_workers,
                        prefetch_factor = bs // args.num_workers + 1 if args.num_workers!=0 else 2,
                        shuffle=shuffle
                    )
    
    training_dataloader = get_dataloader(training_dataset, args.batch_size, shuffle=True)
    valid_dataloader = get_dataloader(valid_dataset, args.test_batch_size)
    test_dataloader = get_dataloader(test_dataset, args.test_batch_size)

    print('finish data loading!')

    return training_dataloader, valid_dataloader, test_dataloader, input_data

training_dataloader, valid_dataloader, test_dataloader, input_data = load_dataset_and_post_process()


model_names = ['DeepFM', 'DIN', 'AutoInt', 'DCN', 'DCNv2', 'xDeepFM']

if args.model not in model_names:
    raise NameError(f"wrong model name: {args.model}")

init_para = {
    'workspace': wksp_path, 'device': args.gpu_id if args.use_gpu else 'cpu',
    'patience': 2, 'loss_function': torch.nn.BCELoss(), 'eval_metric_ls': [roc_auc_score, log_loss], 
    'metric_name_ls': [ 'auc', 'log_loss',], 'model_config': model_config,
    'input_data': input_data
}

model = getattr(models, f'{args.model}')(init_para)
model = model.to(model.device)

optim = torch.optim.Adam(model.parameters(), lr=model_config['lr'], weight_decay=model_config['weight_decay'])

model.fit(
    epochs = args.epochs, training_dataloader=training_dataloader, 
    validation_dataloader = valid_dataloader, optimizer = optim
)

model.load_ckpt()
model.test(
    epoch = 'test', test_dataloader = test_dataloader
)

