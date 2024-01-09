import argparse

import datetime, logging, yaml, random, os, pickle
from collections import namedtuple

import torch
from torch.utils.data import dataloader, dataset
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

import models


parser = argparse.ArgumentParser()
'''
run experiments with augmented data using this file
'''

parser.add_argument('--name', type=str, help='experiment name', default='default')
parser.add_argument('--workspace', type=str, default='./workspace')

parser.add_argument('--dataset', type=str, default='beauty')
parser.add_argument('--use_cpu', dest='use_gpu', action='store_false')
parser.set_defaults(use_gpu=True)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--num_aug', type=int, default=4)
parser.add_argument('--num_workers', type=int, default=16, help='num of workers used for multi-processing data loading')
parser.add_argument('--model', type=str, help='which model to use', default='DeepFM')
parser.add_argument('--batch_size', type=int, help='training batch_size', default=128)
parser.add_argument('--test_batch_size', type=int, help='testing batch_size', default=256)
parser.add_argument('--new_config', type=str, help='update model config', default='')
parser.add_argument('--random_seed', type=int, default=1)
parser.add_argument('--cot_name', type=str, default="default")
parser.add_argument('--prefix_dir', type=str, default="augment_data")
parser.add_argument('--postfix_dir', type=str, default="/")




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
prefix_dir = args.prefix_dir
postfix_dir = args.postfix_dir

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


class my_dataset(dataset.Dataset):
    def __init__(self, dataset, aug_datasets):
        super().__init__()
        self.dataset = dataset
        self.aug_datasets = aug_datasets

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):
        original_data = self.dataset.__getitem__(index)
        aug_data = [aug_dataset.__getitem__(index)
                    for aug_dataset in self.aug_datasets]

        return {'x': original_data[:-1], 'y':original_data[-1], 
                        'aug_x':aug_data}

## load dataset
def load_dataset_and_post_process():
    
    data_par_path = f'../data/{prefix_dir}/{dataset_}/{postfix_dir}'

    def load_json(file_name):
            import json
            with open(file_name, 'r') as f:
                record = json.loads(f.read())
            return record
        
    datamaps = load_json(os.path.join(f'../data/augment_data/{dataset_}/', 'new_datamaps.json'))
    item2attributes = load_json(os.path.join(f'../data/augment_data/{dataset_}/', 'new_item_attributes.json'))

    if not os.path.exists(os.path.join(data_par_path,  'training_w_attr.tsv')):
        print('start loading data')
        training_inter = pd.read_csv(os.path.join(data_par_path, 
                                                'train_icl.tsv'),
                                        sep = '\t')
        valid_inter = pd.read_csv(os.path.join(data_par_path, 
                                            'valid_icl.tsv'),
                                    sep = '\t')
        test_inter = pd.read_csv(os.path.join(data_par_path,
                                            'test_icl.tsv'),
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

        augmentation_data = pickle.load(
            open(f'../data/augment_data/{dataset_}/train_cot_user_pos.pkl', 'rb')
        )

        def read_augment_data_samples(inter_data):

            data_aug_inter = [ [] for i in range(args.num_aug)]
            for line in inter_data['ICL'].values:
                for aug_cnt, arg_id in enumerate(eval(line)):
                    user_id, history, item, label = augmentation_data[arg_id]['user_id'],\
                                augmentation_data[arg_id]['history'], augmentation_data[arg_id]['item'],\
                                augmentation_data[arg_id]['label']
                    cot_emb_id = arg_id
                    history = pad_and_truncate(history)
                    data_aug_inter[aug_cnt].append(
                        (user_id, history, item, cot_emb_id, label)
                    )

            data_aug_inter_pd = [
                pd.DataFrame(
                    data=data_aug_inter[i], columns=['user_id', 'history', 'item', 'cot_id', 'label']
                )
                for i in range(args.num_aug)
            ]
            
            return data_aug_inter_pd
        
        training_aug_inter = read_augment_data_samples(training_inter)
        valid_aug_inter = read_augment_data_samples(valid_inter)
        test_aug_inter = read_augment_data_samples(test_inter)

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

            if 'cot_id' in inter:
                cot_id = inter['cot_id'].values
                cot_id = cot_id.astype(np.int32)
                tensor_dataset = dataset.TensorDataset(
                    torch.LongTensor(uid), 
                    torch.LongTensor(item_id), torch.LongTensor(item_brand_id), torch.LongTensor(item_cate_id),
                    torch.LongTensor(history_item_id), torch.LongTensor(history_brand_id), torch.LongTensor(history_cate_id),
                    torch.LongTensor(cot_id),
                    torch.LongTensor(label)
                )
            else:

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

        print('finish processing original data')

        training_aug_datasets = [
            post_process_item_attributes(inter)
            for inter in training_aug_inter
        ]
        valid_aug_datasets = [
            post_process_item_attributes(inter)
            for inter in valid_aug_inter
        ]
        test_aug_datasets = [
            post_process_item_attributes(inter)
            for inter in test_aug_inter
        ]

        print('finish processing augmentation data')

        

        training_dataset = my_dataset(training_dataset, training_aug_datasets)
        valid_dataset = my_dataset(valid_dataset, valid_aug_datasets)
        test_dataset = my_dataset(test_dataset, test_aug_datasets)

        with open(os.path.join(data_par_path, 'training_w_attr.tsv'), 'wb') as f:
            pickle.dump(training_dataset, f)
        with open(os.path.join(data_par_path, 'valid_w_attr.tsv'), 'wb') as f:
            pickle.dump(valid_dataset, f)
        with open(os.path.join(data_par_path, 'test_w_attr.tsv'), 'wb') as f:
            pickle.dump(test_dataset, f)
        print('finish processing data')
        
    else:
        with open(os.path.join(data_par_path, 'training_w_attr.tsv'), 'rb') as f:
            training_dataset = pickle.load(f)
        with open(os.path.join(data_par_path, 'valid_w_attr.tsv'), 'rb') as f:
            valid_dataset = pickle.load(f)
        with open(os.path.join(data_par_path, 'test_w_attr.tsv'), 'rb') as f:
            test_dataset = pickle.load(f)
        

    SparseFeat = namedtuple('SparseFeat', ['name', 'vocabulary_size', 'embedding_dim'])
    VarLenSparseFeat = namedtuple('VarLenSparseFeat',
                                  ['name','sparsefeat', 'maxlen'])
    DenseFeat = namedtuple('DenseFeat', ['name', 'embedding_dim'])
    
    input_data_name_ls = [
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

    input_aug_data_name_ls = [
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
        DenseFeat(name='cot_id', embedding_dim=768),
        SparseFeat(name='label_id', vocabulary_size=2,
                   embedding_dim=model_config['aug_label_id_dim'] if 'aug_label_id_dim' in model_config else 32
                   ),
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

    return training_dataloader, valid_dataloader, test_dataloader, input_data_name_ls, input_aug_data_name_ls

training_dataloader, valid_dataloader, test_dataloader, input_data_name_ls,\
      input_aug_data_name_ls = load_dataset_and_post_process()


model_names = ['DeepFM','DeepFM_aug','DeepFM_aug_mean', 'DeepFM_aug_wocot', 'DIN', 'DIN_aug', 'DIN_aug_wocot', 'DIN_aug_mean', 'AutoInt', 'AutoInt_aug', 'AutoInt_aug_mean',\
                'AutoInt_aug_wocot', 'DCN', 'DCN_aug', 'DCN_aug_mean', 'DCN_aug_wocot', 'DCNv2', 'DCNv2_aug','DCNv2_aug_mean', 'DCNv2_aug_wocot', 'xDeepFM', 'xDeepFM_aug', 'xDeepFM_aug_mean',\
                      'xDeepFM_aug_wocot']

if args.model not in model_names:
    raise NameError(f"wrong model name: {args.model}")

init_para = {
    'workspace': wksp_path, 'device': args.gpu_id if args.use_gpu else 'cpu',
    'patience': 2, 'loss_function': torch.nn.BCELoss(), 'eval_metric_ls': [roc_auc_score, log_loss], 
    'metric_name_ls': [ 'auc', 'log_loss',], 'model_config': model_config,
    'input_data': input_data_name_ls, 'input_aug_data': input_aug_data_name_ls,
    'num_aug': args.num_aug, 'name': args.model
}

model = getattr(models, f'{args.model}')(init_para)
model = model.to(model.device)

if args.cot_name=='base':
    print("load base model cot !")
    cot_emb = pickle.load(
        open(f'../data/augment_data/{dataset_}/cot_emb_base.pkl', 'rb')
    )
elif args.cot_name=='old':
    print("load base model old !")
    cot_emb = pickle.load(
        open(f'../data/augment_data/{dataset_}/cot_emb_old.pkl', 'rb')
    )
else:
    cot_emb = pickle.load(
        open(f'../data/augment_data/{dataset_}/cot_emb.pkl', 'rb')
    )
item_text_emb = pickle.load(
    open(f'../data/augment_data/{dataset_}/items2text_emb.pkl', 'rb')
)
if 'aug' in args.model:
    model.set_emb_tab_for_aug_data(torch.tensor(np.array(cot_emb)), item_text_emb)

optim = torch.optim.Adam(model.parameters(), lr=model_config['lr'], weight_decay=model_config['weight_decay'])

model.fit(
    epochs = args.epochs, training_dataloader=training_dataloader, 
    validation_dataloader = valid_dataloader, optimizer = optim
)

model.load_ckpt()
model.test(
    epoch = 'test', test_dataloader = test_dataloader
)

