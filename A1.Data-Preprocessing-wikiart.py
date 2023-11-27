import os
import shutil
import pickle
import pandas as pd
import numpy as np
import pickle
import json

import glob
import cv2
import torch
from torchvision.transforms import Resize, CenterCrop
import matplotlib.pyplot as plt
from src.utils import read_data

preprocessing_resize_image_complete = True
if preprocessing_resize_image_complete == False:
    images_path = glob.glob('data/wikiart/images_raw/**/*')
    os.makedirs('data/wikiart/images_256', exist_ok=True)
    for i, path in enumerate(images_path):
        dir_path, filename = os.path.split(path)
        dir_path_new = dir_path.replace('images_raw', 'images_256')
        os.makedirs(dir_path_new, exist_ok=True)
        path_new = os.path.join(dir_path_new, filename)

        img = cv2.imread(path)
        img_t = Resize(256)(torch.Tensor(img.transpose(2,0,1)))
        # img_t = CenterCrop(256)(img_t)
        img_t = img_t.numpy().transpose(1,2,0).astype(np.uint8)
        cv2.imwrite(path_new, img_t, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        if i%100 == 0:
            print(i)

'''
preprocessing wikiart dataset
classes.label2num : linking dictionary of label and index 
'''

# Define Class - id
with open('data/wikiart/attributes/class_data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

tag_data = pd.read_csv('data/wikiart/attributes/wikiart_attributions.csv', low_memory=False)
len(tag_data)

img_path_list = []
for i in range(len(data)):
    img_path_list.append(data[i][0])
len(img_path_list)

tag_dict = {}
for i in tag_data[['path', 'Genre', 'Style', 'Media', 'tag']].values:
    tag_dict[i[0]] = [i[1], i[2], i[3], i[4]]

with open('./data/wikiart/attributes/filename.tag.pkl', 'wb') as f:
    pickle.dump(tag_dict, f)

# with open('./data/wikiart/attributes/filename.tag.pkl', 'rb') as f:
#     data = pickle.load(f)

# media = []
# tags = []
# for _key in data.keys():
#     media+=str(data[_key][2]).replace(', ', ',').split(',')
#     tags+=str(data[_key][3]).replace(', ', ',').replace('"', '').replace("'", '').split(',')
#
# bb = []
# cc = []
# for i in aa:
#     i = str(i)
#     if not i == 'nan':
#         bb.append(len(i.split(',')))
#         cc+= i.split(',')
#     if i == 'nan':
#         bb.append(0)
#         cc+= i.split(',')
# np.unique(cc).shape
# np.median(bb)
# np.mean(bb)
# np.sum(np.array(bb)>1)

## style list
style = []
artist = []
for i in range(len(data)):
    aa = data[i]
    _style = aa[0].split('/')[0]
    _filename = aa[0].split('/')[1]
    _artist = _filename.split('_')[0]
    _style_id = aa[1][2]
    _artist_id = aa[1][0]
    style.append([_style, _style_id])
    artist.append([_artist, _artist_id])
sdat = pd.DataFrame(style, columns=['style', 'style_id'])
sdat = sdat.drop_duplicates()
sdat = sdat.values

for i in range(len(data)):
    aa = data[i]
    _style = aa[0].split('/')[0]
    _filename = aa[0].split('/')[1]
    _genre_id = aa[1][1]
    if _genre_id == 129:
        print(_filename)

    if i == 1000:
        break

## genre list
gdat = [['abstract', 129],
        ['cityscape', 130],
        ['genre painting', 131],
        ['illustration', 132],
        ['landscape', 133],
        ['nude painting', 134],
        ['portrait', 135],
        ['religious painting', 136],
        ['sketch and study', 137],
        ['still life', 138],
        ['other genre', 139]]

gdat = np.array(gdat)
adat = pd.DataFrame(artist, columns=['artist', 'artist_id'])
adat.drop_duplicates()
adat = adat[adat['artist_id'].values > 0].drop_duplicates()
adat.reset_index(drop=True, inplace=True)
adat.loc[128] = ['other artist', 0]
adat = adat.values
atrr_dat = np.concatenate((sdat, gdat, adat), axis=0)

label2num = {}
num2label = {}
for k, v in atrr_dat:
    label2num[k] = int(v)
    num2label[int(v)] = k

class_header = {'label2num':label2num,
                'num2label':num2label}

with open('./data/wikiart/attributes/classes.label2num_raw.pkl', 'wb') as f:
    pickle.dump(class_header, f)









wikiart = pd.read_csv('./data/wikiart/attributes/wclasses.csv')
with open('./data/wikiart/attributes/classes.label2num_raw.pkl', 'rb') as f:
    data = pickle.load(f)


wikiart['artist'] = [data['num2label'][i] for i in wikiart['artist'].values]
wikiart['genre'] = [data['num2label'][i] for i in wikiart['genre'].values]
wikiart['style'] = [data['num2label'][i] for i in wikiart['style'].values]

uniq_art_list = np.unique(wikiart['artist'].values)
uniq_genre_list = np.unique(wikiart['genre'].values)
uniq_style_list = np.unique(wikiart['style'].values)

a_num2label = {}
a_label2num = {}
for i, label in enumerate(uniq_art_list):
    i = int(i)
    a_num2label[i] = label
    a_label2num[label] = i
g_num2label = {}
g_label2num = {}
for i, label in enumerate(uniq_genre_list):
    i = int(i)
    g_num2label[i] = label
    g_label2num[label] = i
s_num2label = {}
s_label2num = {}
for i, label in enumerate(uniq_style_list):
    i = int(i)
    s_num2label[i] = label
    s_label2num[label] = i

label2num_dict = {'artist' : {'label2num':a_label2num,
                              'num2label':a_num2label},
                  'genre' : {'label2num':g_label2num,
                              'num2label':g_num2label},
                  'style' : {'label2num':s_label2num,
                              'num2label':s_num2label}
                  }

wikiart['artist'] = [label2num_dict['artist']['label2num'][i] for i in wikiart['artist'].values]
wikiart['genre'] = [label2num_dict['genre']['label2num'][i] for i in wikiart['genre'].values]
wikiart['style'] = [label2num_dict['style']['label2num'][i] for i in wikiart['style'].values]

wikiart.columns = ['path', 'artist', 'genre', 'style']
wikiart['path'] = [os.path.join('./data/wikiart/images', i) for i in wikiart['path'].values]

np.random.seed(777)
d_size = len(wikiart)
idx = np.arange(d_size)
np.random.shuffle(idx)
thr = int(d_size*0.8)
tr_idx = idx[:thr]
tt_idx = idx[thr:]

wikiart_train = wikiart.iloc[tr_idx, :]
wikiart_test =  wikiart.iloc[tt_idx, :]
wikiart_train.to_csv('./data/wikiart/attributes/preprocessed_wikiart_train.csv', index=False)
wikiart_test.to_csv('./data/wikiart/attributes/preprocessed_wikiart_test.csv', index=False)

with open('./data/wikiart/attributes/classes.label2num.pkl', 'wb') as f:
    pickle.dump(label2num_dict, f)