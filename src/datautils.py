import os
import pickle
import random

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image


def data_path_loader(data_path):
    output = [];
    header = []
    with open(data_path, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            if idx == 0:
                line = line.replace('\n', '')
                line = line.split(',')
                header = line
                continue
            line = line.replace('\n', '')
            line = line.split(',')
            output.append(line)
    return header, output


def pickle_loader(data_path):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data


class RetrieavalDataset(Dataset):
    def __init__(self, data, header, transform=None):
        self.data = data  # [[path, attr1, attr2 ], ...]
        self.header = header[1:]  # ['path', 'attr1_name', 'attr2_name' ...]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx]
        image_path = line[0]
        attrs = line[1:]
        image = read_image(image_path)

        c, h, w = image.shape
        if c == 1:
            image = image.repeat(3, 1, 1)

        label_dict = {}
        for i, columns in enumerate(self.header):
            label_dict[columns] = attrs[i]

        if self.transform:
            image = self.transform(image)

        return {'image': image, 'label': label_dict, 'image_path': image_path}


def data_transforms(dtype='train'):
    if dtype == 'train':
        art_transforms = transforms.Compose([
            transforms.Resize(224, antialias=True),
            transforms.RandomCrop((224, 224)),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    elif dtype == 'test' or dtype == 'valid':
        art_transforms = transforms.Compose([
            transforms.Resize(224, antialias=True),
            transforms.CenterCrop((224, 224)),
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    elif dtype == 'inference':
        art_transforms = transforms.Compose([
            transforms.Resize(224, antialias=True),
            transforms.CenterCrop((224, 224)),
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    elif dtype == 'train_square':
        art_transforms = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            # transforms.CenterCrop((224, 224)),
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    elif dtype == 'test_square':
        art_transforms = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            # transforms.CenterCrop((224, 224)),
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    else:
        print('Type error!!, Available dtype List : train or inference')
    return art_transforms


class QueryDataset(Dataset):
    def __init__(self, dataset, attr_info, intr_attr, args):
        self.dataset = dataset
        self.attr_info = attr_info
        self.attr_list = list(attr_info.keys())
        self.args = args
        self.selected_attr = intr_attr[0]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        query_path = data['candidate']
        condition = data['condition']
        condition_list = []
        for _cond_key in condition:
            condition_list.append([_cond_key,
                                   self.attr_info[_cond_key]['label2num'][condition[_cond_key]],
                                   1])

        return {'query': query_path, 'condition': condition_list}


def omit_nontag_data(inference_data_all, tag_info, dtype_list, args):
    if args.omit_tagless_img == True:
        print(f'ommited tagless image.')
        inference_data = {}
        for dtype in dtype_list:
            inference_data[dtype] = {}
            k = 0
            for idx, i in enumerate(inference_data_all[dtype]):
                _path, filename = os.path.split(inference_data_all[dtype][i]['path'])
                _category2 = os.path.split(_path)[-1]
                _key = os.path.join(_category2, filename)
                _fname = os.path.join(_category2, filename)
                if args.dataset_name == 'wikiart':
                    t_g, t_s, t_m, tt_tag = tag_info[_key]
                    t_tag = str(tt_tag) + ',' + str(t_m) + ',' + str(t_g) + ',' + str(t_s)
                    _tag = t_tag.replace('nan', '').replace(' ', '').replace(',,', ',').lower()
                elif args.dataset_name == 'APY':
                    _category = filename.split('_')[-2]
                    _tag = str(tag_info[_key][1]) + ',' + _category
                elif args.dataset_name == 'CUB':
                    _tag = str(tag_info[_key][1]) + ',' + _category2
                else:
                    print('error there are no such a dataset')

                if len(_tag) == 0 or _tag == 'nan' or len(_tag.split(',')) < 1:
                    continue
                else:
                    inference_data[dtype][_fname] = inference_data_all[dtype][i]
                    inference_data[dtype][_fname]['tag'] = _tag
                    k += 1
            print(f'remained tagless image. {dtype} DB images : {len(inference_data_all[dtype])}')
    else:
        inference_data = inference_data_all
    return inference_data


class DistilDataset(Dataset):
    def __init__(self, dataset, attr_info, intr_attr, args):
        self.dataset = dataset
        self.args = args
        self.dataset_index = list(dataset.keys())
        self.attr_info = attr_info
        self.attr_list = list(attr_info.keys())
        self.selected_attr = intr_attr

        # multiple conditions
        self.attr_classes_list = []
        for i in range(len(self.selected_attr)):
            self.attr_classes_list.append(list(attr_info[self.selected_attr[i]]['label2num'].keys()))

    def __len__(self):
        return len(self.dataset_index)

    def __getitem__(self, idx):
        key = self.dataset_index[idx]
        data = self.dataset[key]
        features = np.array(data['feature'])
        filename = data['path']
        tag = data['tag']

        target = {}
        for _attr in self.attr_list:
            target[_attr] = data[_attr]  # 추후 id 제거 필요
        # 랜덤 조건
        conditional_target = []
        for attr_classes in self.attr_classes_list:
            conditional_target.append(random.choice(attr_classes))

        condition_list = []
        condition_zip = zip(self.selected_attr, conditional_target)
        for _selected_attr, _cond_target in condition_zip:
            condition_list.append([_selected_attr,
                                   self.attr_info[_selected_attr]['label2num'][_cond_target],
                                   1])

        return {'feature': features, 'target': target, 'tag': tag, 'condition': condition_list,
                'filename': filename}
