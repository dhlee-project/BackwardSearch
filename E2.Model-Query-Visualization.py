import os
import sys
import shutil
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import pickle
import glob
import pandas as pd
import time
import copy
import argparse
import torch
import torch.nn as nn
from src.models import ConditionalIRModel, load_encoder
from src.datautils import pickle_loader
from src.evalutils import extract_feature, cir_mapping, visualize
from src.utils import InverseMapping

parser = argparse.ArgumentParser(description='CIR Model')
parser.add_argument('--mname', type=str, default='CIR_Model')
parser.add_argument('--dataset_name', type=str, default='wikiart')
parser.add_argument('--category_type', type=str, default='stylegenre')  # birds, category, stylegenre
parser.add_argument('--num_condition', type=int, default=1)
parser.add_argument('--query_key', type=int, default=10)
parser.add_argument('--encoder_model', type=str,
                    default='convnext_base')  # # resnet50, convnext_base, wideresnet50, vit_b, wideresnet50
parser.add_argument('--encoding_size', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_epoch', type=int, default=30)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--weight_decay', type=float, default=0.000001)
parser.add_argument('--lr_scheduler_step', type=float, default=10)
parser.add_argument('--lr_scheduler_gamma', type=float, default=0.1)
parser.add_argument("--enc_update", type=bool, default=True)
parser.add_argument("--inverse_lr", type=float, default=0.2)
parser.add_argument("--inverse_lambda", type=float, default=1.1)
parser.add_argument("--inverse_max_itr", type=float, default=100)
parser.add_argument("--inverse_earlystopping", type=bool, default=True)
parser.add_argument("--student_reg", type=float, default=0.0)
parser.add_argument("--search_topk", type=int, default=10)
parser.add_argument("--omit_tagless_img", type=bool, default=True)
parser.add_argument("--evaluation_distill", type=bool, default=False)
parser.add_argument("--search_random", type=bool, default=False)
parser.add_argument("--save_result", type=bool, default=False)
parser.add_argument("--n_workers", type=float, default=16)
parser.add_argument("--verbose_step", type=float, default=100)
parser.add_argument("--mode", default='client')
parser.add_argument("--port", default=52162)
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args.device = device

os.makedirs('./result/', exist_ok=True)

if args.dataset_name == 'wikiart' and args.num_condition == 2:
    task_name = 'wikiart_sg'
else:
    task_name = args.dataset_name

attr_info_path = f'./data/{args.dataset_name}/attributes/classes.label2num.pkl'
attr_info = pickle_loader(attr_info_path)
if args.dataset_name == 'wikiart':
    attr_info.pop('artist')

if 'wikiart_sg' in task_name:
    dataset_name = 'wikiart'
    intr_attrs = ['style', 'genre']
elif 'wikiart' in task_name:
    dataset_name = 'wikiart'
    intr_attrs = ['style']
elif 'APY' in task_name:
    dataset_name = 'APY'
    intr_attrs = ['category']
elif 'CUB' in task_name:
    dataset_name = 'CUB'
    intr_attrs = ['birds']
dtype = 'test'

# query load
intr_attr = '-'.join(intr_attrs)
with open(f'./data/{args.dataset_name}/attributes/query_{intr_attr}_{dataset_name}_{dtype}_random.pkl', 'rb') as f:
    queryset = pickle.load(f)

# tag load
tag_info_path = f'../DATA/cir_data/{dataset_name}/attributes/filename.tag.pkl'
tag_info = pickle_loader(tag_info_path)

print("Load Model ...")
base_path = './model'
file_name = f'CIR_Model-dataset{args.dataset_name}-cate{args.category_type}-basemodel{args.encoder_model}' \
            f'-encodingsize{args.encoding_size}-enc_update{args.enc_update}-num_epoch30-final'
file_path = os.path.join(base_path, file_name)
data_train_db = np.load(os.path.join(file_path, f'inference_{args.dataset_name}_train_embedding_epoch30.npy'),
                  allow_pickle=True).item()
data_valid_db = np.load(os.path.join(file_path, f'inference_{args.dataset_name}_test_embedding_epoch30.npy'),
                  allow_pickle=True).item()

name2dbkey = {}
for _key in data_valid_db:
    name2dbkey[data_valid_db[_key]['name']] = _key

print(f'Query : {len(queryset)}')
print(f'Train DB images : {len(data_train_db)}')
print(f'Valid DB images : {len(data_valid_db)}')


model_encoder, enc_output_size = load_encoder(args.encoder_model,
                                              pretrained=True,
                                              enc_update=args.enc_update)
model_encoder.to(device)
model = ConditionalIRModel(basemodel=model_encoder,
                           basemodel_output_size=enc_output_size,
                           encoding_size=args.encoding_size,
                           attr_info=attr_info,
                           args=args)

path = os.path.join('./model', file_name, 'model_30.pth')
model.load_state_dict(torch.load(path)['model_state_dict'])
model.eval()

invmapping_args = {'inv_max_itr': args.inverse_max_itr,
                   'inv_lr': args.inverse_lr,
                   'inv_lambda': args.inverse_lambda,
                   'batch_size': 1,
                   'early_stopping': args.inverse_earlystopping,
                   'device': args.device,
                   "criterion": {'ce': nn.CrossEntropyLoss(),
                                 'mse': nn.MSELoss(),
                                 'be': nn.BCEWithLogitsLoss()},
                   }

InvMapping = InverseMapping(model, invmapping_args)

key = args.query_key # intr_key_list[i]
_fname = os.path.basename(queryset[key]['candidate'])
db_key = name2dbkey[_fname]

label_dict = []
for _key in queryset[key]['condition']:
    _intr_attr = _key
    _intr_attr_val = queryset[key]['condition'][_key]
    intr_cls_id = attr_info[_intr_attr]['label2num'][_intr_attr_val]
    label_dict.append([[_intr_attr], torch.tensor(intr_cls_id), 1])
data_query_updated = cir_mapping(data_valid_db, db_key, label_dict, InvMapping)

_query_f = extract_feature(data_query_updated, query_inv=True)
_db_f = extract_feature(data_valid_db, query_inv=False)
data_q_sim = cosine_similarity(_query_f[:, 1:], _db_f[:, 1:])
visualize(key, queryset, data_query_updated, data_valid_db, data_q_sim, intr_attrs, attr_info, tag_info, args)

