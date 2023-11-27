import os
import shutil
import pickle
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt

from src.utils import read_data

#####################################
############ Cub dataset ############
#####################################


cub_attributes = read_data('./data/CUB/attributes/attributes.txt') # <attribute_id> <attribute_name>
cub_attributes = np.array(cub_attributes)
num2label = {}
label2num = {}
for i, label in cub_attributes:
    num2label[i] = label
    label2num[label] = i

cub_colname_list = list(np.array(cub_attributes)[:,1])
# class_header = {'label2num':label2num,
#                 'num2label':num2label}
# with open('./data/CUB/attributes/classes.label2num.pkl', 'wb') as f:
#     pickle.dump(class_header, f)

# convert_dict={}
# for i, label in cub_attributes:
#     convert_dict[label] = {'num2label':{}, 'label2num':{}}
#     convert_dict[label]['num2label'][1] = 'True'
#     convert_dict[label]['num2label'][1] = 'False'
#     convert_dict[label]['label2num']['False'] = 0
#     convert_dict[label]['label2num']['True'] = 1

cub_img_dat = read_data('./data/CUB/attributes/images.txt') # <image_id> <image_file_name>
cub_img_dat = [[i[0], './data/CUB/images/'+i[1]] for i in cub_img_dat]
cub_img_dat = pd.DataFrame(cub_img_dat)
cub_img_dat.columns = ['id', 'path']

cub_labels = read_data('./data/CUB/attributes/image_attribute_labels.txt') # <image_id> <attribute_id> <is_present> <certainty_id> <worker_id>
cub_labels = pd.DataFrame(cub_labels)
cub_labels.columns = ['id', 'attribute_id', 'is_present', 'certainty_id', 'worker_id', 'etc_0', 'etc_1']
cub_labels = cub_labels[['id', 'attribute_id', 'is_present']]

n_id = np.unique(cub_labels.id.values.astype(int)).shape[0]
n_var = cub_attributes.shape[0]
dat = np.zeros((n_id, n_var+2))
dat = pd.DataFrame(dat)
dat.index = np.unique(cub_labels.id.values.astype(int))
# dat.columns = ['id', 'path'] + list(convert_dict.keys())
dat.columns = ['id', 'path'] + cub_colname_list

data_dict = {}
for i in range(len(cub_labels)):
    _id, _attr_id, _is_present = cub_labels.iloc[i,:]
    dat.loc[int(_id), num2label[_attr_id]] = _is_present

    if i % 1000 == 0:
        print(f'{i}, {len(cub_labels)}')

dat.to_csv('./data/CUB/attributes/tmp_preprocessed_crosstab_dat.csv', index=False)
dat = pd.read_csv('./data/CUB/attributes/tmp_preprocessed_crosstab_dat.csv')
dat['id'] = dat.index.values.astype(str)

for i in range(len(cub_img_dat)):
    _id, _path = cub_img_dat.iloc[i,:]
    dat.loc[int(_id),'path'] = _path
    if i % 1000 == 0:
        print(f'indices are preproceing... {i}, {len(cub_img_dat)} ')

dat = dat.drop(labels='id', axis=1)
dat = dat.dropna(axis=0)
dat = dat.drop(labels=0, axis=0)

_species = [os.path.split(os.path.split(i)[0])[1] for i in dat.path.values]
dat['birds'] = _species
dat = dat.reset_index(inplace=False, drop=True)
np.unique(np.array(_species), return_counts=True)

attr_list = list(dat.columns[1:-1])
dat2 = dat[['path','birds'] + attr_list]
# dat2.to_csv('./data/CUB/attributes/preprocessed_CUB_with_tag.csv', index=False)

#################################################
## 많이 나타나는 변수 제거
#################################################
# plt.boxplot(dat2.iloc[:,2:].astype(int).sum().values/len(dat2))
# plt.show()
# np.quantile(dat2.iloc[:,2:].astype(int).sum().values/len(dat2), 0.8)
# np.mean(dat2.iloc[:,2:].astype(int).sum().values/len(dat2))
# np.std(dat2.iloc[:,2:].astype(int).sum().values/len(dat2))

# np.quantile(dat2.iloc[:,2:].astype(int).sum().values/len(dat2), 0.8)
# 0.18493383101459113
selected_var = list(dat2.iloc[:,2:].astype(int).sum().values/len(dat2)<0.1849)
dat2 = dat2.loc[:,[True, True]+selected_var]
attr_list = np.array(attr_list)[selected_var]


tag_dict = {}
for i in range(len(dat2)):
    lines = dat2.iloc[i,:]
    _key = lines.values[0]
    _class = lines.values[1]
    _attr = lines.values[2:]
    _attr = ','.join(np.array(attr_list)[_attr.astype(bool)].tolist())

    _path, filename = os.path.split(_key)
    _style = os.path.split(_path)[-1]
    _key = os.path.join(_style, filename)
    tag_dict[_key] = [_class, _attr]

with open('./data/CUB/attributes/filename.tag.pkl', 'wb') as f:
    pickle.dump(tag_dict, f)


dat = dat[['path','birds']] # +list(dat.columns[1:-1])]

convert_dict={}
convert_dict['birds'] = {'num2label':{}, 'label2num':{}}
cub_class = np.unique(dat.birds.values)
for i, label in enumerate(cub_class):
    convert_dict['birds']['num2label'][i] = label
    convert_dict['birds']['label2num'][label] = i

with open('./data/CUB/attributes/classes.label2num.pkl', 'wb') as f:
    pickle.dump(convert_dict, f)

dat['birds'] = [convert_dict['birds']['label2num'][i] for i in dat['birds'].values]


split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
for train_index, test_index in split.split(dat, dat["birds"]):
    dat_data_train = dat.loc[train_index]
    dat_data_test = dat.loc[test_index]

dat_data_train.to_csv('./data/CUB/attributes/preprocessed_CUB_train.csv', index=False)
dat_data_test.to_csv('./data/CUB/attributes/preprocessed_CUB_test.csv', index=False)

