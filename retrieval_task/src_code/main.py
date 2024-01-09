import argparse

import datetime, logging, yaml, random, os, pickle
from collections import namedtuple

import torch
from torch.utils.data import dataloader, dataset
import numpy as np
import pandas as pd

import models

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
parser.add_argument('--num_neg', type=int, help='number of negative samples for training', default=128)
parser.add_argument('--batch_size', type=int, help='training batch_size', default=256)
parser.add_argument('--test_batch_size', type=int, help='testing batch_size', default=256)
parser.add_argument('--new_config', type=str, help='update model config', default='')
parser.add_argument('--random_seed', type=int, default=2023)

args = parser.parse_args()

def setup_seed(seed):
    '''setting random seeds'''
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed) 
setup_seed(args.random_seed)

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

class retrieval_dataset(dataset.Dataset):
    def __init__(self, interaction_data:pd.DataFrame):
        super().__init__()

        self.user_id = interaction_data['user_id'].values.astype(np.int32)
        self.history = interaction_data['history'].values
        self.item_id = interaction_data['item'].values.astype(np.int32)
        # self.label = interaction_data['label'].values.astype(np.float32)

    def __len__(self):
        return self.user_id.__len__()

    def __getitem__(self, index):
        user_id = torch.tensor(self.user_id[index]).long()
        item_id = torch.tensor(self.item_id[index]).long()
        # label = torch.Tensor(self.label[index])
        history = torch.tensor(self.history[index]).long()

        return user_id, history, item_id
    
class my_dataset(dataset.Dataset):
    def __init__(self, dataset, aug_datasets=None) -> None:
        super().__init__()
        self.dataset = dataset
        self.aug_datasets = aug_datasets

    def __len__(self):
        return self.dataset.__len__()
    
    def __getitem__(self, index):
        
        orig_data = self.dataset.__getitem__(index)

        if self.aug_datasets is not None:
            aug_data = [
                aug.__getitem__(index) for aug in self.aug_datasets
            ]

            return {
                'x': orig_data[:-1], 'aug_x': aug_data, 'y': orig_data[-1]
            }
        else:
            return {
                'x': orig_data[:-1], 'y': orig_data[-1]
            }

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
    
    data_par_path = f'../data/retrieval_data/{dataset_}/'

    def load_json(file_name):
        import json
        with open(file_name, 'r') as f:
            record = json.loads(f.read())
        return record
        
    datamaps = load_json(os.path.join(data_par_path, 'new_datamaps.json'))
    # item2attributes = load_json(os.path.join(data_par_path, 'new_item_attributes.json'))

    training_inter = pd.read_csv(os.path.join(data_par_path, 'dataset', 'training.tsv'), sep='\t')
    valid_inter = pd.read_csv(os.path.join(data_par_path, 'dataset', 'validation.tsv'), sep='\t')
    test_inter = pd.read_csv(os.path.join(data_par_path, 'dataset', 'test.tsv'), sep='\t')

    def pad_and_truncate(seq: list):
        seq = eval(seq)

        ret = seq[-model_config['max_len']:]
        if len(ret) < model_config['max_len']:
            ret.extend( (model_config['max_len'] - len(ret)) * ['0'] )

        ret = [int(i) for i in ret]
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

    # Filter out negative sampled interactions. 
    # They are used for CTR models, not the retrieval models.
    training_inter = training_inter[training_inter['label']==1.0]
    valid_inter = valid_inter[valid_inter['label']==1.0]
    test_inter = test_inter[test_inter['label']==1.0]

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
        VarLenSparseFeat(sparsefeat='item_id', maxlen=model_config['max_len'],
                         name='history_item_id'),
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
    
    training_dataset_orig = retrieval_dataset(training_inter)
    valid_dataset_orig = retrieval_dataset(valid_inter)
    test_dataset_orig = retrieval_dataset(test_inter)

    training_dataset = my_dataset(training_dataset_orig)
    valid_dataset = my_dataset(valid_dataset_orig)
    test_dataset = my_dataset(test_dataset_orig)
    
    training_dataloader = get_dataloader(training_dataset, args.batch_size, shuffle=True)
    valid_dataloader = get_dataloader(valid_dataset, args.test_batch_size)
    test_dataloader = get_dataloader(test_dataset, args.test_batch_size)

    print('finish data loading!')

    return training_dataloader, valid_dataloader, test_dataloader, input_data

training_dataloader, valid_dataloader, test_dataloader, input_data = load_dataset_and_post_process()


model_names = ['SASREC', 'YoutubeDNN', 'GRU4REC', 'SRGNN']

if args.model not in model_names:
    raise NameError(f"wrong model name: {args.model}")

init_para = {
    'workspace': wksp_path, 'device': args.gpu_id if args.use_gpu else 'cpu',
    'patience': 2, 'loss_function': torch.nn.BCELoss(),
    'model_config': model_config, 'name': args.model,
    'num_neg': args.num_neg,  
    'input_data': input_data,
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

