import os
import cv2
import glob
import json
import pickle
import random
import numpy as np
import pandas as pd
import json
import cv2
import torch
from torchvision.transforms import Resize, CenterCrop
import matplotlib.pyplot as plt
from src.datautils import pickle_loader, QueryDataset
from src.models import load_encoder
from src.datautils import data_transforms
from sklearn.metrics.pairwise import cosine_similarity

def TagChecker(tag, NumOfExistTag):
    return len(tag) == 0 or tag == 'nan' or len(tag.split(',')) < NumOfExistTag


random.seed(1234)
np.random.seed(777)
'''
Wikiart dataset 
'''
import itertools

dtype = 'test'
NumGetQuery = 1000
NumOfExistTag = 1

for date_type in ['wikiart_sg', 'CUB', 'APY', 'wikiart']:
# for date_type in  ['CUB', 'APY', 'wikiart']:
    random.seed(1234)
    np.random.seed(777)

    if date_type == 'wikiart_sg':
        dataset_name = 'wikiart'
        intr_attrs = ['style', 'genre']
    elif date_type == 'wikiart':
        dataset_name = 'wikiart'
        intr_attrs = ['style']
    elif date_type == 'CUB':
        dataset_name = 'CUB'
        intr_attrs = ['birds']
    elif date_type == 'APY':
        dataset_name = 'APY'
        intr_attrs = ['category']

    dataset_train = pd.read_csv(f'./data/{dataset_name}/attributes/preprocessed_{dataset_name}_train.csv')
    dataset_test = pd.read_csv(f'./data/{dataset_name}/attributes/preprocessed_{dataset_name}_test.csv')
    attr_info_path = f'./data/{dataset_name}/attributes/classes.label2num.pkl'
    attr_info = pickle_loader(attr_info_path)

    conditions_list_dict = {}
    for intr_attr in intr_attrs:
        conditions_list_dict[intr_attr] = list(attr_info[intr_attr]['label2num'].items())
    tag_info_path = f'./data/{dataset_name}/attributes/filename.tag.pkl'
    tag_info = pickle_loader(tag_info_path)


    if dtype == 'test':
        _dataset = dataset_test
    if dtype == 'train':
        _dataset = dataset_train

    ## balanced sampling
    if dataset_name == 'APY':
        len_category = len(attr_info['category']['num2label'])
        idx_list = list(_dataset.index.values)
        cate, cnt = np.unique(_dataset.category.values, return_counts=True)
        cnt_weight = (1 / cnt) / len_category
        cnt_dict = {}
        for i in cate:
            cnt_dict[i] = cnt_weight[i]
        sample_weight_list = [cnt_dict[i] for i in _dataset.category.values]
        np.random.choice(idx_list, 1, p=sample_weight_list)
    elif dataset_name == 'CUB':
        len_category = len(attr_info['birds']['num2label'])
        idx_list = list(_dataset.index.values)
        cate, cnt = np.unique(_dataset.birds.values, return_counts=True)
        cnt_weight = (1 / cnt) / len_category
        cnt_dict = {}
        for i in cate:
            cnt_dict[i] = cnt_weight[i]
        sample_weight_list = [cnt_dict[i] for i in _dataset.birds.values]
        np.random.choice(idx_list, 1, p=sample_weight_list)
    elif dataset_name == 'wikiart' and len(intr_attrs) == 1:
        idx_list = list(_dataset.index.values)
        cate, cnt = np.unique(_dataset['style'].values, return_counts=True)
        cnt_weight = (1 / cnt) / len(cate)
        cnt_dict = {}
        for i in cate:
            cnt_dict[i] = cnt_weight[i]
        sample_weight_list = [cnt_dict[i] for i in _dataset['style'].values]
    elif dataset_name == 'wikiart' and len(intr_attrs) == 2:
        len_category = len(attr_info['style']['num2label'])
        idx_list = list(_dataset.index.values)

        _sg_val = [str(i[0]) + '-' + str(i[1]) for i in _dataset[['style', 'genre']].values]
        cate, cnt = np.unique(_sg_val, return_counts=True)
        cnt_weight = (1 / cnt) / len(cate)
        cnt_dict = {}
        for idx, i in enumerate(cate):
            cnt_dict[i] = cnt_weight[idx]
        sample_weight_list = [cnt_dict[i] for i in _sg_val]


    kk = 0
    queryset = {}
    _uniq_list = []
    while kk < NumGetQuery:
        # idx = random.choice(idx_list)
        idx = np.random.choice(idx_list, 1, p=sample_weight_list)[0]

        if dataset_name == 'wikiart':
            path, artist, genre, style = _dataset.loc[idx, :].values
            candidate_path = path
            candidate_filename = '/'.join(candidate_path.split('/')[-2:])
            _key = candidate_filename
            t_g, t_s, t_m, tt_tag = tag_info[_key]
            t_tag = str(tt_tag) + ',' + str(t_m) + ',' + str(t_g) + ',' + str(t_s)
            t_tag = t_tag.replace('nan', '').replace(' ', '').replace(',,', ',').lower()
            candidate_tags = str(t_tag)
        else:
            path, tag = _dataset.loc[idx, :].values
            candidate_path = path
            candidate_base_path = candidate_path.split('/')[-2:]
            if dataset_name == 'CUB':
                _category = candidate_base_path[0]
            elif dataset_name == 'APY':
                _category = candidate_base_path[1].split('_')[-2]
            candidate_filename = '/'.join(candidate_base_path)
            _key = candidate_filename
            candidate_tags = str(tag_info[_key][1]) + ',' + _category

        tagfalse = TagChecker(candidate_tags, NumOfExistTag)
        if tagfalse:
            continue

        # candidate_emb = extracted_features[_key]
        _sub_dataset2 = _dataset.drop(idx).copy()
        select_conditions_dict = {}
        for intr_attr in intr_attrs:
            _list = conditions_list_dict[intr_attr]
            np.random.shuffle(_list)
            select_conditions_dict[intr_attr] = _list

        conditions_list2 = []
        for i in range(1):
            c_name_dict = {}
            c_idx_dict = {}
            for intr_attr in intr_attrs:
                c, ii = random.choice(select_conditions_dict[intr_attr])
                c_name_dict[intr_attr] = c
                c_idx_dict[intr_attr] = ii
            conditions_list2.append([c_name_dict, c_idx_dict])

        step = 0
        for c_name, c_id in conditions_list2:
            # break
            _sub_c_dat = _sub_dataset2.copy()
            for intr_attr in intr_attrs:
                _sub_c_dat = _sub_c_dat[_sub_c_dat[intr_attr].values == c_id[intr_attr]]
            if len(_sub_c_dat) == 0:
                continue
            _sub_c_dat = _sub_c_dat.reset_index(drop=True)
            sim_list = []
            fname_list = []
            tags_list = []
            target_emb_list = []
            for j in range(len(_sub_c_dat)):
                if dataset_name == 'wikiart':
                    t_path, _, _, _ = _sub_c_dat.iloc[j, :].values
                    t_fname = '/'.join(t_path.split('/')[-2:])
                    t_g, t_s, t_m, tt_tag = tag_info[t_fname]
                    t_tag = str(tt_tag) + ',' + str(t_m) + ',' + str(t_g) + ',' + str(t_s)
                    t_tag = t_tag.replace('nan', '').replace(' ', '').replace(',,', ',').lower()
                    target_tags = str(t_tag)
                else:
                    t_path, tag = _sub_c_dat.loc[j, :].values
                    target_path = t_path

                    target_base_path = target_path.split('/')[-2:]
                    if dataset_name == 'CUB':
                        _category = target_base_path[0]
                    elif dataset_name == 'APY':
                        _category = target_base_path[1].split('_')[-2]
                    t_fname = '/'.join(target_path.split('/')[-2:])
                    _key = t_fname
                    target_tags = str(tag_info[_key][1]) + ',' + _category
                    target_tags = target_tags.lstrip(',').rstrip(',')

                A = candidate_tags.split(',')
                B = target_tags.split(',')
                sim = (len(np.intersect1d(A, B))) / (len(np.union1d(A, B)))

                # target_emb = extracted_features[t_fname]
                tagfalse = TagChecker(target_tags, NumOfExistTag)
                if tagfalse:
                    sim = 0

                sim_list.append(sim)
                fname_list.append(t_fname)
                tags_list.append(target_tags)
                # target_emb_list.append(target_emb[0])

            _sub_c_dat.insert(1, 'tags', tags_list)
            _sub_c_dat.insert(1, 'fname', fname_list)
            _sub_c_dat.insert(1, 'sim', sim_list)
            sim_list = np.array(sim_list)

            if np.sum(np.array(sim_list) < 0.1) < 1:
                print('skip : no relation')
                continue

            rank_k = 10
            _cur_k = 0
            if len(sim_list) < rank_k:
                continue

            step += 1
            _query = {}
            _query['candidate'] = candidate_filename
            _query['candidate_tags'] = candidate_tags
            _query['condition'] = c_name

            _query_uniq = {}
            _query_uniq['candidate'] = candidate_filename
            _query_uniq['condition'] = c_name

            _query_str = str(_query_uniq)
            if _query_str not in _uniq_list:
                queryset[kk] = _query
                _uniq_list.append(_query_str)
                kk += 1

            stop_bool = random.choice([False])
            if stop_bool:
                break

        print(kk)

    intr_attrs = '-'.join(intr_attrs)
    with open(f'./data/{dataset_name}/attributes/query_{intr_attrs}_{dataset_name}_{dtype}_random.pkl', 'wb') as f:
        pickle.dump(queryset, f)