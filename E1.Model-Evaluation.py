import argparse
import glob
import os
# import random
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datautils import pickle_loader, QueryDataset, omit_nontag_data
from src.evalutils import evaluation_recall_run, compute_eval_metric
from src.models import load_encoder, ConditionalIRModel, Student
from src.utils import logger, load_model, InverseMapping

parser = argparse.ArgumentParser(description='CIR Model')
parser.add_argument('--mname', type=str, default='CIR_Model')
parser.add_argument('--dataset_name', type=str, default='wikiart')
parser.add_argument('--category_type', type=str, default='stylegenre')  # birds, category, style, stylegenre
parser.add_argument('--num_condition', type=int, default=1)
parser.add_argument('--encoder_model', type=str,
                    default='convnext_base')  # # resnet50, convnext_base, wideresnet50, vit_b, wideresnet50
parser.add_argument('--encoding_size', type=int, default=256)  # 2048
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


if args.evaluation_distill:
    args.infer_method_type = 'feature_student_inv'
else:
    args.infer_method_type = 'feature_inv'

invmapping_args = {'inv_max_itr': args.inverse_max_itr,
                   'inv_lr': args.inverse_lr,
                   'inv_lambda': args.inverse_lambda,
                   'batch_size': 512,
                   'early_stopping': args.inverse_earlystopping,
                   'device': args.device,
                   "criterion": {'ce': nn.CrossEntropyLoss(),
                                 'mse': nn.MSELoss(),
                                 'be': nn.BCEWithLogitsLoss()},
                   }

attr_info_path = f'./data/{args.dataset_name}/attributes/classes.label2num.pkl'
attr_info = pickle_loader(attr_info_path)
if args.dataset_name == 'wikiart' and args.category_type == 'stylegenre':
    attr_info.pop('artist')
elif args.dataset_name == 'wikiart' and args.category_type == 'style':
    attr_info.pop('artist')
    attr_info.pop('genre')

if args.dataset_name != 'wikiart' and args.num_condition != 1:
    assert ('model only support 2 condition on wikiart')

if args.dataset_name == 'wikiart' and args.num_condition == 1:
    dtype_list = ['wikiart_test']
    intr_attr = ['style']
    infer_dtype_list = ['wikiart_test']
elif args.dataset_name == 'wikiart' and args.num_condition == 2:
    dtype_list = ['wikiart_test']
    intr_attr = ['style', 'genre']
    infer_dtype_list = ['wikiart_test']
elif args.dataset_name == 'CUB' and args.category_type == 'birds':
    dtype_list = ['CUB_test']
    intr_attr = ['birds']
    infer_dtype_list = ['CUB_test']
elif args.dataset_name == 'APY' and args.category_type == 'category':
    dtype_list = ['APY_test']
    intr_attr = ['category']
    infer_dtype_list = ['APY_test']
else:
    assert ('error. please check dataset_name, category_type argument')

invmapping_args['intr_attr'] = intr_attr
args.modelname = f'{args.mname}-dataset{args.dataset_name}-cate{args.category_type}-basemodel{args.encoder_model}' \
                 f'-encodingsize{args.encoding_size}-enc_update{args.enc_update}-num_epoch{args.num_epoch}-final'

log = logger(loggername='CIR Logger',
             outputpath=args.modelname,
             phase='CIR')

log.info(f'info: Model : {args.modelname}')
log.info(f'info: Model : {args}')
log.info('info: Loading datasets')
log.info('info: Load Dataset')

log.info('info: Load Model')
model_encoder, enc_output_size = load_encoder(args.encoder_model,
                                              pretrained=True,
                                              enc_update=args.enc_update)
model_encoder.to(args.device)
model = ConditionalIRModel(basemodel=model_encoder,
                           basemodel_output_size=enc_output_size,
                           encoding_size=args.encoding_size,
                           attr_info=attr_info,
                           args=args)  # args.embedding_size

model, optimizer, scheduler = load_model(model, optimizer=None, scheduler=None, args=args)

for param in model.parameters():
    param.requires_grad = False
model.eval()

attr_classes_list = []
for _attr in intr_attr:
    attr_classes_list += list(attr_info[_attr]['label2num'].keys())

if args.evaluation_distill == True:
    if args.num_condition == 1:
        distil_path = f'./model/{args.modelname}/knowledge_distill_model_lambda{args.inverse_lambda}_reg{args.student_reg}'
    elif args.dataset_name == 'wikiart' and args.num_condition == 2:
        distil_path = f'./model/{args.modelname}/knowledge_distill_model_lambda{args.inverse_lambda}_reg{args.student_reg}_cond2'
    else:
        assert ('please check knowledge distill model path')
    student = Student(args.encoding_size, len(attr_classes_list)).to(device)
    load_path = glob.glob(f'{distil_path}/*49.pth')[-1]
    print(f'Load student : {load_path}')
    student.load_state_dict(torch.load(load_path))
    student.eval()
else:
    student = None

# load tag dataset
tag_info_path = f'./data/{args.dataset_name}/attributes/filename.tag.pkl'
tag_info = pickle_loader(tag_info_path)

args.file_path = os.path.join('./model', args.modelname)
query_data = {}
inference_data = {}
for dtype in dtype_list:
    intr_attr_str = '-'.join(intr_attr)
    ######### 중요
    with open(os.path.join(f'./data/{args.dataset_name}/attributes/', f'query_{intr_attr_str}_{dtype}_random.pkl'),
              'rb') as f:
        query_data[dtype] = pickle.load(f)

    _inference_data = np.load(os.path.join(args.file_path,
                                           f'inference_{dtype}_embedding_epoch{args.num_epoch}.npy'),
                              allow_pickle=True).item()
    inference_data[dtype] = _inference_data
    print(f'{dtype} DB images : {len(inference_data[dtype])}')

inference_data = omit_nontag_data(inference_data, tag_info, dtype_list, args)
# print(f'{dtype} DB images : {len(inference_data[dtype])}')

DistillDatasets = {}
for dtype in dtype_list:
    torch_dataset = QueryDataset(query_data[dtype],
                                 attr_info, intr_attr, args)

    torch_dataloader = DataLoader(torch_dataset,
                                  batch_size=invmapping_args['batch_size'],
                                  shuffle=False,
                                  num_workers=0)
    DistillDatasets[dtype] = torch_dataloader

InvMapping = InverseMapping(model, invmapping_args)

updated_query_dict = {}
map_result = {}
for dtype in infer_dtype_list:
    updated_query_dict[dtype] = evaluation_recall_run(dataloader=DistillDatasets[dtype], database=inference_data[dtype],
                                                      model=model, student=student, InvMapping=InvMapping,
                                                      invmapping_args=invmapping_args,
                                                      attr_info=attr_info, dtype=dtype, args=args, log=log)

    _inference_data = inference_data[dtype]
    result = compute_eval_metric(query_data, dtype, updated_query_dict, _inference_data, tag_info,
                                 top_k=args.search_topk, num_query=1000, infer_method_type=args.infer_method_type,
                                 args=args)


    if args.save_result == True:
        intr_attr_str = '-'.join(intr_attr)
        meta = [args.dataset_name, intr_attr_str, args.encoder_model, str(args.enc_update),
                str(args.encoding_size), str(args.inverse_lr), str(args.inverse_lambda), str(args.search_topk)]
        result = meta + result
        filename = '-'.join(meta)
        if not os.path.exists(f'./tmp/eval/'):
            os.makedirs(f'./tmp/eval/', exist_ok=True)
        with open(f'./tmp/eval/{filename}.pkl', 'wb') as f:
            pickle.dump(result, f)

