import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.datautils import pickle_loader, omit_nontag_data, DistilDataset
from src.models import load_encoder, ConditionalIRModel, Student
from src.utils import InverseMapping, epoch_distil_run
from src.utils import logger, load_model

parser = argparse.ArgumentParser(description='CIR Model')
parser.add_argument('--mname', type=str, default='CIR_Model')
parser.add_argument('--dataset_name', type=str, default='CUB')
parser.add_argument('--category_type', type=str, default='birds') # birds, category, stylegenre
parser.add_argument('--encoder_model', type=str, default='convnext_base') #  # resnet50
parser.add_argument('--num_condition', type=int, default=1)
parser.add_argument('--encoding_size', type=int, default=256) # 2048
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_epoch', type=int, default=30)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0.000001)
parser.add_argument('--lr_scheduler_step', type=float, default=10)
parser.add_argument('--lr_scheduler_gamma', type=float, default=0.1)
parser.add_argument("--enc_update", type=bool, default=True)
parser.add_argument("--inverse_lr", type=float, default=0.2)
parser.add_argument("--inverse_lambda", type=float, default=1.1)
parser.add_argument("--inverse_max_itr", type=float, default=100)
parser.add_argument("--student_reg", type=float, default=0.0)
parser.add_argument("--student_num_epoch", type=float, default=50)
parser.add_argument("--omit_tagless_img", type=bool, default=True)
parser.add_argument("--n_workers", type=float, default=16)
parser.add_argument("--verbose_step", type=float, default=100)
parser.add_argument("--mode", default='client')
parser.add_argument("--port", default=52162)
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args.device = device
args.omit_tagless_img = True

invmapping_args = {'inv_max_itr': args.inverse_max_itr,
                   'inv_lr': args.inverse_lr,
                   'inv_lambda': args.inverse_lambda,
                   'batch_size': 512,
                   'early_stopping': True,
                   'device': args.device,
                   "criterion": {'ce': nn.CrossEntropyLoss(),
                                 'mse': nn.MSELoss(),
                                 'be': nn.BCEWithLogitsLoss()},
                   }

student_args = {
                 'num_epochs': args.student_num_epoch,
                 'best_loss': float('inf'),
                 'criterion' : invmapping_args['criterion']
                 }

attr_info_path = f'./data/{args.dataset_name}/attributes/classes.label2num.pkl'
attr_info = pickle_loader(attr_info_path)

if args.dataset_name == 'wikiart' and args.category_type == 'stylegenre':
    attr_info.pop('artist')
elif args.dataset_name == 'wikiart' and args.category_type == 'style':
    attr_info.pop('artist')
    attr_info.pop('genre')

if args.dataset_name != 'wikiart' and args.num_condition != 1:
    assert('model only support 2 condition on wikiart')

if args.dataset_name == 'wikiart' and args.category_type == 'stylegenre' \
        and args.num_condition == 1:
    dtype_list = ['wikiart_train']
    intr_attr = ['style']
    infer_dtype_list = ['wikiart_test']
elif args.dataset_name == 'wikiart' and args.category_type == 'stylegenre' \
        and args.num_condition == 2:
    dtype_list = ['wikiart_train']
    intr_attr = ['style', 'genre']
    infer_dtype_list = ['wikiart_test']
elif args.dataset_name == 'CUB' and args.category_type == 'birds':
    dtype_list = ['CUB_train']
    intr_attr = ['birds']
    infer_dtype_list = ['CUB_test']
elif args.dataset_name == 'APY' and args.category_type == 'category':
    dtype_list = ['APY_train']
    intr_attr = ['category']
    infer_dtype_list = ['APY_test']
else:
    assert('error. please check dataset_name, category_type argument')

invmapping_args['intr_attr'] = intr_attr
invmapping_args['selected_var'] = intr_attr
student_args['intr_attr'] = invmapping_args['intr_attr']
# inverse mapping arguments


args.modelname = f'{args.mname}-dataset{args.dataset_name}-cate{args.category_type}-basemodel{args.encoder_model}-encodingsize{args.encoding_size}'\
            f'-enc_update{args.enc_update}-num_epoch{args.num_epoch}-final'

if args.num_condition == 1:
    if not os.path.exists(f'./model/{args.modelname}/knowledge_distill_model_lambda{args.inverse_lambda}_reg{args.student_reg}'):
        os.makedirs(f'./model/{args.modelname}/knowledge_distill_model_lambda{args.inverse_lambda}_reg{args.student_reg}', exist_ok=True)
elif args.num_condition == 2 and args.dataset_name == 'wikiart':
    if not os.path.exists(f'./model/{args.modelname}/knowledge_distill_model_lambda{args.inverse_lambda}_reg{args.student_reg}_cond2'):
        os.makedirs(f'./model/{args.modelname}/knowledge_distill_model_lambda{args.inverse_lambda}_reg{args.student_reg}_cond2', exist_ok=True)
else:
    assert('no model')

log = logger(loggername='CIR Logger', outputpath=args.modelname, phase='distil')
log.info(f'info: Model : {args.modelname}')
log.info(f'info: Model : {args}')
log.info('info: Loading datasets')
log.info('info: Load Dataset')


tag_info_path = f'./data/{args.dataset_name}/attributes/filename.tag.pkl'
tag_info = pickle_loader(tag_info_path)

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

# criterion = {'ce': nn.CrossEntropyLoss(), 'mse': nn.MSELoss(), 'be': nn.BCELoss}
model, optimizer, scheduler = load_model(model, optimizer=None, scheduler=None, args=args)
model.to(device)
for param in model.parameters():
    param.requires_grad = False
model.eval()

args.file_path = os.path.join('./model', args.modelname)
inference_data_all = {}
for dtype in dtype_list:
    inference_data_all[dtype] = np.load(os.path.join(args.file_path,
                                                 f'inference_{dtype}_embedding_epoch{args.num_epoch}.npy'),
                                    allow_pickle=True).item()
    print(f'{dtype} DB images : {len(inference_data_all[dtype])}')

inference_data = omit_nontag_data(inference_data_all, tag_info, dtype_list, args)
intr_attr = invmapping_args['intr_attr']

attr_classes_list = []
for _attr in intr_attr:
    attr_classes_list += list(attr_info[_attr]['label2num'].keys())
student = Student(args.encoding_size, len(attr_classes_list)).to(device)
student.train()

student_args['optimizer'] = optim.Adam(student.parameters(), lr=1e-3)
student_args['optimizer'].zero_grad()

InvMapping = InverseMapping(model, invmapping_args)

DistillDatasets = {}
for dtype in dtype_list:
    torch_dataset = DistilDataset(inference_data[dtype], attr_info, intr_attr, args)

    torch_dataloader = DataLoader(torch_dataset,
                                  batch_size=invmapping_args['batch_size'],  #
                                  shuffle=True,
                                  num_workers=8)
    DistillDatasets[dtype] = torch_dataloader

loss_history = {'train': [], 'val': []}
metric_history = {'train': [], 'val': []}
for epoch in range(1, student_args['num_epochs']):
    for dtype in dtype_list:
        if 'train' in dtype:
            args.current_phase = 'train'
            student.train()
            tr_loss = epoch_distil_run(DistillDatasets[dtype],
                                       model, InvMapping, student,
                                       attr_info, student_args, epoch, args)
        else:
            args.current_phase = 'test'
            student.eval()
            tt_loss = epoch_distil_run(DistillDatasets[dtype],
                                       model, InvMapping, student,
                                       attr_info, student_args, epoch, args)

    #if tr_loss < student_args['best_loss']:
    if (epoch + 1) % 10 == 0 or epoch == 0:
        # save model
        epoch_str = str(epoch).zfill(4)
        if args.num_condition == 1:
            torch.save(student.state_dict(),
                       os.path.join(args.file_path,
                                    f'./knowledge_distill_model_lambda{args.inverse_lambda}_reg{args.student_reg}/student_model_epoch{epoch_str}.pth')
                       )
        elif args.dataset_name == 'wikiart' and args.num_condition == 2:
            torch.save(student.state_dict(),
                       os.path.join(args.file_path,
                                    f'./knowledge_distill_model_lambda{args.inverse_lambda}_reg{args.student_reg}_cond2/student_model_epoch{epoch_str}.pth')
                       )
        else:
            assert(print('need proper model name'))

        student_args['best_loss'] = tr_loss
        print(f'best loss {tr_loss}: save model at epoch {epoch}')
