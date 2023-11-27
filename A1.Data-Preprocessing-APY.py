import os
import shutil
import pickle
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import json
from src.utils import read_data


def preprocessing_apy_img(_dataset, name2path, save_img):
    file_path_list = []
    for i in range(len(_dataset)):
        image_name = _dataset.iloc[i,:]['path']
        category = _dataset.iloc[i,:]['category']
        image_coord = _dataset.loc[i,:][['x1', 'x2', 'x3', 'x4']]
        xmin, ymin, xmax, ymax = image_coord.astype(int)
        impath = os.path.join('./data/APY/images_raw', name2path[image_name])
        name, ext = os.path.splitext(image_name)
        new_filename = f'{name}_{category}_{i}{ext}'
        file_path = f'./data/APY/images_crop/{new_filename}'
        file_path_list.append(file_path)
        if save_img ==True:
            img = plt.imread(impath)[ymin:ymax, xmin:xmax, :]
            plt.imsave(file_path, img)

        if i % 100 == 0:
            print(f'{i}th images were preprocessing')
    print('save_all')
    return file_path_list

apy_pascal_train = read_data('./data/APY/attributes/apascal_train.txt')
apy_pascal_test = read_data('./data/APY/attributes/apascal_test.txt')
# apy_yahoo_test = read_data('./data/APY/attributes/ayahoo_test.txt')
apy_attr = read_data('./data/APY/attributes/attribute_names.txt', sep=False)
data = pd.read_csv('./data/APY/images_raw/image_label.csv', index_col=0)

img_path_list = data.image_path.values
name2path = {}
for i in range(len(img_path_list)):
    filepath = img_path_list[i]
    filename = os.path.split(filepath)[-1]
    name2path[filename] = filepath

# images preprocessing
apy_colums_selected = ['path', 'category'] + apy_attr
apy_colums = ['path', 'category', 'x1', 'x2', 'x3', 'x4'] + apy_attr
apy_pascal_train = pd.DataFrame(apy_pascal_train, columns=apy_colums)
apy_pascal_test = pd.DataFrame(apy_pascal_test, columns=apy_colums)

save_img = True
os.makedirs('./data/APY/images_crop', exist_ok=True)
file_path_list = preprocessing_apy_img(apy_pascal_train, name2path, save_img)
file_path_list_test = preprocessing_apy_img(apy_pascal_test, name2path, save_img)

apy_pascal_train['path'] = file_path_list
apy_pascal_test['path'] = file_path_list_test
apy_pascal_train = apy_pascal_train[apy_colums_selected]
apy_pascal_test = apy_pascal_test[apy_colums_selected]

# apy_yahoo_test = pd.DataFrame(apy_yahoo_test, columns=apy_colums)
# apy_yahoo_test = apy_yahoo_test[apy_colums_selected]

np.unique(apy_pascal_test['category'].values)
np.unique(apy_pascal_train['category'].values)
# np.unique(apy_yahoo_test['category'].values)
# label 2 num
apy_pascal_class = np.unique(apy_pascal_train['category'].values)

convert_dict={}
convert_dict['category'] = {'num2label': {}, 'label2num': {}}
apy_class = np.unique(apy_pascal_class)
for i, label in enumerate(apy_class):
    convert_dict['category']['num2label'][i] = label
    convert_dict['category']['label2num'][label] = i

with open('./data/APY/attributes/classes.label2num.pkl', 'wb') as f:
    pickle.dump(convert_dict, f)

apy_pascal_train['category'] = [convert_dict['category']['label2num'][i] for i in apy_pascal_train['category'].values]
apy_pascal_test['category'] = [convert_dict['category']['label2num'][i] for i in apy_pascal_test['category'].values]

dat2 = pd.concat((apy_pascal_train,apy_pascal_test), axis=0)
attr_list = list(dat2.columns[2:])

tag_dict = {}
for i in range(len(dat2)):
    # break
    lines = dat2.iloc[i,:]
    _key = lines.values[0]
    _class = lines.values[1]
    _attr = lines.values[2:].astype(int)
    _attr = ','.join(np.array(attr_list)[_attr.astype(bool)].tolist())
    # break
    _path, filename = os.path.split(_key)
    _style = os.path.split(_path)[-1]
    _key = os.path.join(_style, filename)
    tag_dict[_key] = [_class, _attr]

with open('./data/APY/attributes/filename.tag.pkl', 'wb') as f:
    pickle.dump(tag_dict, f)

apy_pascal_train = apy_pascal_train[['path', 'category']]
apy_pascal_test = apy_pascal_test[['path', 'category']]

apy_pascal_train.to_csv('./data/APY/attributes/preprocessed_APY_train.csv', index=False)
apy_pascal_test.to_csv('./data/APY/attributes/preprocessed_APY_test.csv', index=False)
