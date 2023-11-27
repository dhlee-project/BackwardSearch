import argparse
import os
import shutil

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

from src.datautils import RetrieavalDataset, data_transforms
from src.datautils import pickle_loader, data_path_loader
from src.models import load_encoder, ConditionalIRModel
from src.utils import evaluation, logger, load_model

parser = argparse.ArgumentParser(description='CIR Model')
parser.add_argument('--mname', type=str, default='CIR_Model')
parser.add_argument('--dataset_name', type=str, default='CUB')
parser.add_argument('--category_type', type=str, default='birds')  # birds, category, stylegenre
parser.add_argument('--encoder_model', type=str, default='convnext_base')  # convnext_base# resnet50
parser.add_argument('--encoding_size', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_epoch', type=int, default=30)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--weight_decay', type=float, default=0.000001)
parser.add_argument('--lr_scheduler_step', type=float, default=10)
parser.add_argument('--lr_scheduler_gamma', type=float, default=0.1)
parser.add_argument("--enc_update", type=bool, default=True)
parser.add_argument("--n_workers", type=float, default=16)
parser.add_argument("--verbose_step", type=float, default=100)
parser.add_argument("--mode", default='client')
parser.add_argument("--port", default=52162)
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args.device = device

if args.dataset_name == 'wikiart':
    dtype_list = ['wikiart_train', 'wikiart_test']
    transform_type = 'train'
if args.dataset_name == 'CUB':
    dtype_list = ['CUB_train', 'CUB_test']
    transform_type = 'train'
if args.dataset_name == 'APY':
    dtype_list = ['APY_train', 'APY_test']
    transform_type = 'train_square'

# args.modelname='test'
args.modelname = f'{args.mname}-dataset{args.dataset_name}-cate{args.category_type}-basemodel{args.encoder_model}-encodingsize{args.encoding_size}' \
                 f'-enc_update{args.enc_update}-num_epoch{args.num_epoch}-final'

# mkdir
if not os.path.exists(f'./model/{args.modelname}/weight'):
    os.makedirs(f'./model/{args.modelname}/weight', exist_ok=True)

if os.path.exists(f'./model/{args.modelname}/src'):
    shutil.rmtree(f'./model/{args.modelname}/src')
shutil.copytree(f'./src', f'./model/{args.modelname}/src')

log = logger(loggername='CIR Logger', outputpath=args.modelname)
log.info(f'info: Model : {args.modelname}')
log.info(f'info: Model : {args}')
log.info('info: Loading datasets')
log.info('info: Load Dataset')

attr_info_path = f'./data/{args.dataset_name}/attributes/classes.label2num.pkl'
attr_info = pickle_loader(attr_info_path)

if args.dataset_name == 'wikiart' and args.category_type == 'stylegenre':
    attr_info.pop('artist')
elif args.dataset_name == 'wikiart' and args.category_type == 'style':
    attr_info.pop('artist')
    attr_info.pop('genre')
if args.dataset_name == 'fashion-iq':
    attr_info.pop('shirt')
    attr_info.pop('toptee')
# if args.dataset_name == 'APY':
#     attr_info = attr_info['category']

RetrieavalDatasets = {}
for dtype in dtype_list:
    if 'train' in dtype:
        _phase = 'train'
        transform_type = 'train'
    else:
        _phase = 'test'
        transform_type = 'test'
    if args.dataset_name == 'APY':
        transform_type = transform_type + '_square'

    data_path = f'./data/{args.dataset_name}/attributes/preprocessed_{dtype}.csv'
    header, data = data_path_loader(data_path)
    torch_dataset = RetrieavalDataset(data,
                                      header,
                                      data_transforms(transform_type))

    torch_dataloader = DataLoader(torch_dataset,
                                  batch_size=args.batch_size,  #
                                  shuffle=False,
                                  num_workers=args.n_workers)
    RetrieavalDatasets[dtype] = torch_dataloader

log.info('info: Load Model')
model_encoder, enc_output_size = load_encoder(args.encoder_model,  # # vit_b
                                              pretrained=True,
                                              enc_update=args.enc_update)
model_encoder.to(args.device)
model = ConditionalIRModel(basemodel=model_encoder,
                           basemodel_output_size=enc_output_size,
                           encoding_size=args.encoding_size,
                           attr_info=attr_info,
                           args=args)  # args.embedding_size

# define optim
criterion = {'mse': nn.MSELoss(),
             'ce': nn.CrossEntropyLoss(),
             'bce': nn.BCEWithLogitsLoss()}

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

scheduler = optim.lr_scheduler.StepLR(optimizer,
                                      step_size=args.lr_scheduler_step,
                                      gamma=args.lr_scheduler_gamma)

model_prefix = 'model'
model, optimizer, scheduler = load_model(model, optimizer, scheduler, model_prefix, args)
model.to(args.device)
model.eval()

# evaluation
log.info('info: Run Inference')

for dtype in dtype_list:
    with torch.no_grad():
        log.info(f'info: Inference - {dtype}')
        evaluation(RetrieavalDatasets[dtype], model, attr_info, dtype, args, log)
