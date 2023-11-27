import os
import copy
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import matplotlib.pyplot as plt
import textwrap

def extract_feature(data, query_inv):
    var_feature = 'feature'
    if query_inv:
        var_feature = 'inv_feature'
    idx = list(data.keys())[0]
    d_f = np.zeros((len(data), len(np.array(data[idx][var_feature]).reshape(-1).tolist()) + 1))
    for idx, key in enumerate(data):
        d_f[idx, :] = np.array([key] + np.array(data[key][var_feature]).reshape(-1).tolist())
    print(f'feature shape : {d_f.shape}')

    return d_f

def evaluation_recall_run(dataloader=None, database=None, model=None, student=None, InvMapping=None,
                          invmapping_args=None,
                          attr_info=None, dtype=None, args=None, log=None):
    attr_list = list(attr_info.keys())
    data_query_updated = {};
    key = 0
    for itr, batch in enumerate(dataloader):
        model.to(args.device)
        query_paths = batch['query']
        condition_dict = batch['condition']

        _conditions = {}
        for i in range(len(condition_dict)):
            _conditions[i] = {}
            _conditions[i]['attr'] = condition_dict[i][0]
            _conditions[i]['label'] = condition_dict[i][1].tolist()

        input_features = []

        for i in range(len(query_paths)):
            features = database[query_paths[i]]['feature']
            input_features.append(features)

        updated_features = InvMapping.batch_compute(input_features.copy(), condition_dict)

        input_features = np.array(input_features).squeeze()
        input_features = torch.FloatTensor(input_features).squeeze().to(args.device)
        updated_features = np.array(updated_features).squeeze()
        updated_features = torch.FloatTensor(updated_features).to(args.device)

        targets = {}
        for i in range(len(condition_dict)):
            _attr = list(set(_conditions[i]['attr']))[0]
            targets[_attr] = torch.tensor(_conditions[i]['label']).to(args.device)

        if student:
            teacher_output_cls = {}
            for i, _attr in enumerate(invmapping_args['intr_attr']):
                teacher_output_cls[_attr] = model.attr_clssifiers[_attr](updated_features)
                cond = np.zeros((len(input_features), len(attr_info[_attr]['label2num'])))
                for ii in range(len(cond)):
                    cond[ii, int(targets[_attr][ii])] = 1
                if i == 0:
                    condition = cond
                else:
                    condition = np.concatenate((condition, cond), axis=1)

            condition = torch.FloatTensor(condition).to(args.device)
            cond_input_feature = torch.concat((input_features.to(torch.float).to(args.device),
                                               condition), dim=1)
            student_output_feature = student(cond_input_feature)

            student_output_feature = student_output_feature.detach().cpu().numpy()

        # save updated features
        for i in range(len(updated_features)):
            data_query_updated[key] = {}
            data_query_updated[key]['filename'] = query_paths[i]

            for _attr in attr_list:
                for ii in _conditions:
                    if _attr in list(_conditions[ii]['attr']):
                        data_query_updated[key][f'{_attr}_condition-id'] = _conditions[ii]['label'][i]
                        data_query_updated[key][f'{_attr}_condition'] = attr_info[_attr]['num2label'][
                            _conditions[ii]['label'][i]]

            data_query_updated[key]['feature_inv'] = updated_features[i].tolist()
            if student:
                data_query_updated[key]['feature_student_inv'] = student_output_feature[i].tolist()
            data_query_updated[key]['feature_original'] = input_features[i].tolist()
            key += 1
    log.info(f'info: completed inversemapping proc. {dtype}')
    return data_query_updated


def compute_topk_tag_sim(query_tag, filename_topn_path, tag_info, top_k, args):
    sim_list = []

    if args.dataset_name == 'wikiart':
        query_tag = ','.join([str(ii).replace('nan', '') for ii in query_tag if str(ii) != 'nan'])
        query_tag = query_tag.replace('nan', '').replace(' ', '').replace(',,', ',').lower().lstrip(',').rstrip(',')
        query_tag = query_tag.split(',')
    else:
        query_tag = query_tag[1].split(',')
    for _k in range(top_k):
        if args.dataset_name == 'wikiart':
            _tag = tag_info[filename_topn_path[_k]]
            _tag = ','.join([str(ii).replace('nan', '') for ii in _tag if str(ii) != 'nan'])
            _tag = _tag.replace('nan', '').replace(' ', '').replace(',,', ',').lower().lstrip(',').rstrip(',')
        elif args.dataset_name == 'APY':
            _search_path = filename_topn_path[_k]
            _cate = _search_path.split('_')[-2]
            t_path, _tag = tag_info[_search_path]
            _tag = _tag + ',' + _cate
            _tag = _tag.lstrip(',').rstrip(',')
        elif args.dataset_name == 'CUB':
            _search_path = filename_topn_path[_k]
            _cate = _search_path.split('/')[0]
            t_path, _tag = tag_info[_search_path]
            _tag = _tag + ',' + _cate
            _tag = _tag.lstrip(',').rstrip(',')
        else:
            assert (f'not support dataset : {args.dataset_name}')

        B = _tag.split(',')
        A = query_tag
        sim = (len(np.intersect1d(A, B))) / (len(np.union1d(A, B)))
        sim_list.append(sim)
    return sim_list


def compute_eval_metric(query_data, dtype, updated_query_dict, sub_inference_data, tag_info, top_k=10, num_query=1000,
                        infer_method_type='feature_inv', args=None):
    features_database = np.zeros((len(sub_inference_data), args.encoding_size))
    labels_index = {}
    labels_index['name'] = {}
    labels_index['path'] = {}
    for i, key in enumerate(sub_inference_data):
        features_database[i, :] = sub_inference_data[key]['feature'].copy()
        labels_index['name'][i] = sub_inference_data[key]['name']
        labels_index['path'][i] = sub_inference_data[key]['path']

    ap_list = []
    sim_list = []
    retrieval_list = []
    query_list = []
    for idx in tqdm(range(num_query)):
        query_path = query_data[dtype][idx]['candidate']

        query_conditon = query_data[dtype][idx]['condition']
        query_conditon_str = ','.join(list(query_conditon.values()))

        query = updated_query_dict[dtype][idx][infer_method_type]  # feature_inv, feature_original, feature_inv
        query = np.array(query).reshape(1, -1)
        if args.search_random:
            _idx = np.arange(len(features_database))
            np.random.shuffle(_idx)
            data_sim_argsort = _idx[:top_k]
        else:
            data_q_sim = cosine_similarity(query, features_database)
            data_sim_argsort = np.argpartition(-data_q_sim, top_k, axis=1)[:, :top_k]
            sort_idx = np.argsort(data_q_sim.reshape(-1)[data_sim_argsort])[0, ::-1]
            data_sim_argsort = data_sim_argsort[:, sort_idx]
        _filename_topn_name = np.vectorize(labels_index['name'].get)(data_sim_argsort)
        _filename_topn_path = np.vectorize(labels_index['path'].get)(data_sim_argsort).squeeze()

        query_cond = np.array(list(query_data[dtype][idx]['condition'].values()))
        if args.dataset_name == 'wikiart' or args.dataset_name == 'CUB':
            search_cond = np.array([i.split('/')[-2] for i in _filename_topn_path])
            filename_topn_path = np.array(['/'.join(i.split('/')[-2:]) for i in _filename_topn_path])
        else:
            search_cond = np.array([i.split('/')[-1].split('_')[-2] for i in _filename_topn_path])
            filename_topn_path = np.array(['/'.join(i.split('/')[-2:]) for i in _filename_topn_path])

        # ap
        if args.dataset_name == 'wikiart' and len(query_conditon) == 2:
            _correct = 0
            for _i in range(len(filename_topn_path)):
                _style = sub_inference_data[filename_topn_path[_i]]['style']
                _genre = sub_inference_data[filename_topn_path[_i]]['genre']
                search_cond = _style + ',' + _genre
                if query_conditon_str == search_cond:
                    _correct += 1
            ap = _correct / top_k
            ap_list.append(ap)
        else:
            ap = np.sum(query_cond == search_cond) / top_k
            ap_list.append(ap)

        query_tag = tag_info[query_path]
        if args.dataset_name == 'APY':
            _category = query_path.split('_')[-2]
            query_tag[1] = query_tag[1] + ',' + _category
        elif args.dataset_name == 'CUB':
            _category = query_path.split('/')[0]
            query_tag[1] = query_tag[1] + ',' + _category
            pass
        elif args.dataset_name == 'wikiart':
            pass
        else:
            pass

        _query_sim_ = compute_topk_tag_sim(query_tag, filename_topn_path, tag_info, top_k, args)
        sim_list.append(np.mean(_query_sim_))
        retrieval_list.append(filename_topn_path)
        query_list.append(query_path)

    mAP = round(np.sum(ap_list) / num_query, 3)
    recall = None
    mSIM = round(np.sum(sim_list) / num_query, 3)
    mSIM_std = round(np.std(sim_list), 3)
    print(f'mAP@{top_k}:{mAP}, mSIM@{top_k}:{mSIM}, std_SIM@{top_k}:{mSIM_std}')
    return [mAP, recall, mSIM, sim_list, query_list, retrieval_list]


def cir_mapping(data_query, key, label_dict, InvMapping):
    idx = 0
    data_query_updated = {}
    data_query_updated[idx] = copy.deepcopy(data_query[key])
    features = data_query[key]['feature']
    # inv mapping
    # InvMapping.batch_compute(input_features.copy(), condition_dict)
    updated_features =  InvMapping.batch_compute([features], label_dict)
    # save updated features
    data_query_updated[idx]['inv_feature'] = updated_features[0].tolist()
    return data_query_updated


def visualize(key, queryset, data_query, data_db, data_q_sim, intr_attrs, attr_info, tag_info, args):
    idx = 0
    # query_name = queryset[key]['candidate']
    if args.dataset_name == 'wikiart' and len(intr_attrs) == 1:
        # init_cate = query_name.split('/')[0]
        _genre = data_query[idx]['genre']
        _style = data_query[idx]['style']
        init_cate = _style + ',' + _genre
    elif args.dataset_name == 'wikiart' and len(intr_attrs) == 2:
        _genre = data_query[idx]['genre']
        _style = data_query[idx]['style']
        init_cate = _style + ',' + _genre
    elif args.dataset_name == 'APY':
        init_cate = data_query[idx]['category']
    else:
        init_cate = data_query[idx]['birds']

    cond_cate = ','.join(list(queryset[key]['condition'].values()))
    queryset_tags = queryset[key]['candidate_tags']

    attr_list = list(attr_info.keys())
    plt.figure(figsize=(30, 10))
    dummy_i = 1
    intr_sim = data_q_sim[idx, :]
    sort_val = np.argsort(1 - intr_sim)[1:]

    for _iter in range(6):
        rank = _iter
        if _iter == 0:
            _name = os.path.splitext(data_query[idx]['name'])[0]
            _path = data_query[idx]['path']

            label_dict = {}
            for _attr in attr_list:
                label_dict[_attr] = data_query[idx][_attr]

            _style = _path.split('/')[-2]
            plt.subplot(2, 6, rank + dummy_i)
            img = plt.imread(_path)
            plt.imshow(img)

            label_str = ''

            for _ii, _init_cate in enumerate(init_cate.split(',')):
                if _ii == 0:
                    label_str += f'Class : {_init_cate}'
                else:
                    label_str += f', {_init_cate}'
            # label_str += f'Class : {retrieve_cate} \n'
            label_str += f';'
            label_str += f'Condition : {cond_cate}; '
            label_str += f'Tag : {queryset_tags}; '

            label_str = textwrap.fill(label_str, width=20,
                                      break_long_words=True)
            plt.xlabel(label_str,
                       fontsize=20)
            plt.xticks([], [])
            plt.yticks([], [])

        if _iter == 6:
            dummy_i = + dummy_i + 1

        if _iter != 0:
            _sim = round(intr_sim[sort_val[rank - 1]], 3)
            _name = os.path.splitext(data_db[sort_val[rank - 1]]['name'])[0]
            _path = data_db[sort_val[rank - 1]]['path']

            label_dict = {}
            for _attr in attr_list:
                label_dict[_attr] = data_db[sort_val[rank - 1]][_attr]
            retrieve_cate = ','.join(list(label_dict.values()))

            if args.dataset_name == 'wikiart':
                _name_key = '/'.join(_path.split('/')[-2:])
                _tag = tag_info[_name_key]
                _tag = ','.join([str(ii).replace('nan', '') for ii in _tag if str(ii) != 'nan'])
                _tag = _tag.replace('nan', '').replace(' ', '').replace(',,', ',').lower().lstrip(',').rstrip(',')
                _sorted_tag = _tag
            else:
                _name_key = '/'.join(_path.split('/')[-2:])
                _sorted_tag = tag_info[_name_key][1] + ',' + retrieve_cate
            _sorted_tag = _sorted_tag.lstrip(',').rstrip(',')
            queryset_tags = queryset_tags.lstrip(',').rstrip(',')
            A = queryset_tags.split(',')
            B = _sorted_tag.split(',')
            sim = (len(np.intersect1d(A, B))) / (len(np.union1d(A, B)))

            plt.subplot(2, 6, rank + dummy_i)
            img = plt.imread(_path)
            plt.imshow(img)

            label_str = ''
            for _ii, _retrieved_cate in enumerate(retrieve_cate.split(',')):
                if _ii == 0:
                    label_str += f'Class : {_retrieved_cate}'
                else:
                    label_str += f', {_retrieved_cate}'
            label_str += f';'
            label_str += f'Tag sim. : {sim:.2}; '
            label_str += f'Tag : {_sorted_tag}; '
            label_str = textwrap.fill(label_str, width=20,
                                      break_long_words=True)
            plt.xlabel(label_str,
                       fontsize=20, wrap=True)
            plt.xticks([], [])
            plt.yticks([], [])
        if _iter == 0:
            plt.title('Query', size=20)
        else:
            title_str = ''
            plt.title(f'Top {rank}', size=20)

    plt.savefig(f'./result/{_name}.jpg')





