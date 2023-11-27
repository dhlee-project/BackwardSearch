import glob
import logging
import logging.handlers
import os
import time

import numpy as np
import torch
import torch.nn as nn


def read_data(path, sep=True):
    with open(path, 'rt') as f:
        data = f.read()
    data = data.split('\n')[:-1]
    if sep:
        dat = []
        for i in data:
            dat.append(i.split(' '))
    else:
        dat = data
    return dat


def load_model(model, optimizer, scheduler, model_prefix='model', args=None):
    load_path = glob.glob(f'./model/{args.modelname}/{model_prefix}_*.pth')[-1]
    print(f'load weight : {load_path}')
    _load_weight = torch.load(load_path, map_location='cuda:0')
    if model_prefix == 'model':
        model.load_state_dict(_load_weight['model_state_dict'])
    else:
        model.load_state_dict(_load_weight)
    if optimizer:
        optimizer.load_state_dict(_load_weight['optimizer_state_dict'])
    if scheduler:
        scheduler.load_state_dict(_load_weight['scheduler_state_dict'])
    return model, optimizer, scheduler


def save_model(model, optimizer, scheduler, args):
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()},
               f'./model/{args.modelname}/model_{args.num_epoch}.pth'
               )
    print('saved model completely')


def logger(loggername, outputpath, phase='train'):
    log = logging.getLogger(f'{loggername}')
    log.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fileHandler = logging.FileHandler(f'./model/{outputpath}/log_{phase}.txt')
    streamHandler = logging.StreamHandler()
    fileHandler.setFormatter(formatter)
    streamHandler.setFormatter(formatter)
    log.addHandler(fileHandler)
    log.addHandler(streamHandler)
    return log


def vis_loss(running, epoch, len_dataloader, itr, start_time, student_args):
    _total_loss = running['total_loss'] / itr
    _l2 = running['updated_mse'] / itr
    _reg = running['regularization'] / itr
    due_time = time.time() - start_time
    print(f'[{epoch}/{student_args["num_epochs"]}][{itr}/{len_dataloader}], updated_mse_f:{_l2:.3},'
          f' input_regularization_f:{_reg:.3}, time:{due_time:.3}')
    return _total_loss


def evaluation(dataloader, model, attr_info, dtype, args, log):
    log.info('info: model evaluation')
    attr_name_list = list(attr_info.keys())
    len_data_loader = len(dataloader)

    result_ = {};
    kk = 0
    for idx, batch in enumerate(dataloader):
        images = batch['image']
        label_dict_raw = batch['label']
        image_path_dict = batch['image_path']

        images = images.to(args.device)
        features = model.encoder(images)[0]

        label_dict = {}
        if args.dataset_name == 'APY':
            label_dict['category'] = np.array(list(label_dict_raw.values())).transpose(1, 0)
        else:
            label_dict = label_dict_raw

        # features = F.relu(features)
        for i in range(len(features)):
            result_[kk] = {}
            path = image_path_dict[i]
            result_[kk]['path'] = path
            result_[kk]['name'] = os.path.basename(path)

            for _attrname in attr_name_list:
                _label_id = label_dict[_attrname][i]
                _label = attr_info[_attrname]['num2label'][int(_label_id)]
                result_[kk][f'{_attrname}'] = _label
                result_[kk][f'{_attrname}_id'] = _label_id

            result_[kk]['feature'] = features[i].tolist()
            kk += 1

        if idx % 100 == 0:
            log.info(f'info: [{idx}/{len_data_loader}]')

    np.save(f'./model/{args.modelname}/inference_{dtype}_embedding_epoch{args.num_epoch}.npy', result_)
    log.info('info: Finish eval')


def epoch_run(dataloader, model, attr_info, optimizer, criterion, epoch, args, log=None):
    attr_name_list = list(attr_info.keys())

    # save training status
    loss_dict = {'loss_total': []}

    for _val in attr_name_list:
        loss_dict.update({f'loss_{_val}': []})
    for _val in attr_name_list:
        loss_dict.update({f'acc_{_val}': []})

    if args.current_phase == 'train':
        model.train()
        log.info('info: model train')
    else:
        model.eval()
        log.info('info: model evaluation')

    due_time = time.time()
    len_data_loader = len(dataloader)
    for idx, batch in enumerate(dataloader):
        # get data
        images = batch['image']
        label_dict = batch['label']

        for _idx, _attr_name in enumerate(attr_name_list):
            _label = np.array(label_dict[_attr_name]).astype(int)
            label_dict[f'{_attr_name}'] = torch.tensor(_label).to(args.device)

        images = images.to(args.device)
        embedings, out_dict = model(images)

        # compute loss
        _loss = {}
        for _attr_name in attr_name_list:
            _loss[f'loss_{_attr_name}'] = criterion['ce'](out_dict[f'{_attr_name}'], label_dict[f'{_attr_name}'])

        # compute total loss
        len_loss = len(_loss)
        loss = 0
        for key in _loss:
            loss += _loss[key] / len_loss

        if args.current_phase == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # loss
        loss_dict['loss_total'].append(loss.item())

        for _attr_name in attr_name_list:
            loss_dict[f'loss_{_attr_name}'].append(_loss[f'loss_{_attr_name}'].item())

        # accuracy
        for _attr_name in attr_name_list:
            _pred_attr = torch.argmax(out_dict[f'{_attr_name}'], 1)
            _correct = (_pred_attr == label_dict[f'{_attr_name}']).tolist()
            loss_dict[f'acc_{_attr_name}'] += _correct

        if (idx + 1) % args.verbose_step == 0 or (idx + 1) == len_data_loader:
            s1 = time.time()

            _v_loss = {}
            _v_loss['total_loss'] = np.mean(loss_dict['loss_total'])

            for _attr_name in attr_name_list:
                _v_loss[f'loss_{_attr_name}'] = np.mean(loss_dict[f'loss_{_attr_name}'])
                _v_loss[f'acc_{_attr_name}'] = np.mean(loss_dict[f'acc_{_attr_name}'])

            loss_str = ''
            for key in _v_loss:
                loss_str += f'{key} : {round(_v_loss[key], 3)} - '
            loss_str += f'time : {s1 - due_time:.1f}'

            log.info(f'info: [{epoch}/{args.num_epoch}][{(idx + 1)}/{len_data_loader}] - {loss_str}')
            due_time = time.time()

    return _v_loss


def epoch_distil_run(dataloader, model, InvMapping, student, attr_info, student_args, epoch, args):
    start_time = time.time()
    running = {'loss_b': 0, 'updated_mse': 0, 'regularization': 0, 'metric': 0, 'total_loss': 0}

    len_dataloader = len(dataloader)
    for itr, batch in enumerate(dataloader):
        itr += 1
        model.to(args.device)
        input_feature = batch['feature']
        condition_dict = batch['condition']
        # target = condition_dict[1]

        _conditions = {}
        for i in range(len(condition_dict)):
            _conditions[i] = {}
            _conditions[i]['attr'] = condition_dict[i][0]
            _conditions[i]['label'] = condition_dict[i][1].tolist()

        input_features = input_feature.tolist()
        updated_feature = InvMapping.batch_compute(input_features.copy(), condition_dict)

        input_feature = np.array(input_feature).squeeze()
        input_feature = torch.FloatTensor(input_feature).squeeze().to(args.device)
        updated_feature = np.array(updated_feature).squeeze()
        updated_feature = torch.FloatTensor(updated_feature).to(args.device)

        targets = {}
        for i in range(len(condition_dict)):
            _attr = list(set(_conditions[i]['attr']))[0]
            targets[_attr] = torch.tensor(_conditions[i]['label']).to(args.device)

        teacher_output_cls = {}

        for i, _attr in enumerate(student_args['intr_attr']):
            teacher_output_cls[_attr] = model.attr_clssifiers[_attr](updated_feature)

            cond = np.zeros((len(input_feature), len(attr_info[_attr]['label2num'])))
            for ii in range(len(cond)):
                cond[ii, int(targets[_attr][ii])] = 1
            if i == 0:
                condition = cond
            else:
                condition = np.concatenate((condition, cond), axis=1)

        condition = torch.FloatTensor(condition).to(args.device)
        cond_input_feature = torch.concat((input_feature, condition), axis=1)
        student_output_feature = student(cond_input_feature)

        student_output_cls = {}
        for i, _attr in enumerate(student_args['intr_attr']):
            student_output_cls[_attr] = model.attr_clssifiers[_attr](student_output_feature)

        loss_mse = torch.mean(torch.square(student_output_feature - updated_feature), axis=1).mean()
        loss_reg = torch.mean(torch.abs(student_output_feature - input_feature), axis=1).mean()

        total_loss = loss_mse + args.student_reg * loss_reg

        if args.current_phase == 'train':
            student_args['optimizer'].zero_grad()
            total_loss.backward()
            student_args['optimizer'].step()

        running['total_loss'] += total_loss.item()
        running['updated_mse'] += loss_mse.item()
        running['regularization'] += loss_reg.item()

        if itr % 50 == 0 or itr == len_dataloader - 1:
            total_mean_loss = vis_loss(running, epoch, len_dataloader, itr, start_time, student_args)

    return total_mean_loss


class InverseMapping():
    def __init__(self, model, invmapping_args):
        self.model = model
        self.invmapping_args = invmapping_args
        self.input_feature = None
        self.condition_list = None

    def batch_compute(self, features, condition_dict):
        self.invmapping_args['criterion']['ce'].reduction = 'none'

        target_list = []
        attr_list = []
        for i, _cond in enumerate(condition_dict):
            target_list.append(_cond[1].to(self.invmapping_args['device']))
            attr_list += list(set(_cond[0]))

        if len(attr_list) >= 2:
            assert ('inverse mapping fn can handle upto two attr.')

        # features
        inv_features = torch.tensor(np.array(features.copy()),
                                    requires_grad=True,
                                    dtype=torch.float32,
                                    device=self.invmapping_args['device'])
        query_features = torch.tensor(np.array(features.copy())).to(self.invmapping_args['device'])
        optim_feature = torch.optim.Adam([inv_features], lr=self.invmapping_args['inv_lr'])

        inv_features2 = torch.zeros_like(inv_features).detach().cpu()
        best_loss = np.array([np.inf] * len(inv_features))
        early_stop_counter = np.array([0] * len(inv_features))
        for step in range(1, self.invmapping_args['inv_max_itr'] + 1):
            optim_feature.zero_grad()

            loss_cls = torch.zeros  ###
            # weight = [0.3, 0.7]
            for i in range(len(condition_dict)):
                updated_y = self.model.attr_clssifiers[attr_list[i]](inv_features)
                _loss_cls = self.invmapping_args['criterion']['ce'](updated_y.squeeze(),
                                                                    target_list[i])
                if i == 0:
                    loss_cls = _loss_cls
                else:
                    loss_cls += _loss_cls

            regular_l1 = self.invmapping_args['inv_lambda'] * torch.mean(torch.abs(inv_features - query_features),
                                                                         axis=1)

            condition_loss = loss_cls / len(condition_dict)
            total_loss = condition_loss + regular_l1  # + regular_cos# + regular_cls * 0.00001 # + regular_kd#+ regular_cos* 0.001
            total_loss.backward(torch.ones_like(total_loss))
            optim_feature.step()

            if self.invmapping_args['early_stopping']:
                monitors = total_loss
                for ii in range(len(monitors)):
                    monitor = monitors[ii].item()
                    if best_loss[ii] >= monitor and early_stop_counter[ii] <= 10:
                        best_loss[ii] = monitor
                        inv_features2[ii, :] = inv_features[ii, :].detach().cpu()
                        early_stop_counter[ii] = 0
                    else:
                        early_stop_counter[ii] += 1

        if not self.invmapping_args['early_stopping']:
            inv_features2 = inv_features.detach().cpu().numpy()
        else:
            inv_features2 = inv_features2.numpy()
        return inv_features2

    def compute(self, features, condition_dict):

        given_condition = {}
        for key in condition_dict:
            _label, lambda_a = condition_dict[key]
            given_condition[key] = [torch.LongTensor([int(float(_label))]).view(-1),
                                    lambda_a
                                    ]
        # features
        inv_features = torch.tensor(np.array(features.copy()).reshape(1, -1),
                                    requires_grad=True,
                                    dtype=torch.float32,
                                    device=self.invmapping_args['device'])

        query_features = torch.tensor(np.array(features.copy())).to(self.invmapping_args['device'])
        optim_feature = torch.optim.Adam([inv_features], lr=self.invmapping_args['inv_lr'])

        ### log
        _log_invmap = {'init': {'init_features': features.copy()}}
        best_loss = None  # best loss

        for step in range(1, self.invmapping_args['inv_max_itr'] + 1):
            optim_feature.zero_grad()
            loss_cls = {}

            for cond in given_condition:
                _cond_id, lambda_a = given_condition[cond]
                updated_y = self.model.attr_clssifiers[cond](inv_features)
                loss_cls[cond] = self.invmapping_args['criterion']['ce'](updated_y,
                                                                         _cond_id.to(
                                                                             self.invmapping_args['device'])) * lambda_a

            regular_l1 = self.invmapping_args['inv_lambda'] * torch.sum(torch.abs(inv_features - query_features))
            condition_loss = sum([loss_cls[key] for key in loss_cls])
            total_loss = condition_loss + regular_l1
            total_loss.backward()
            optim_feature.step()

            monitor = condition_loss
            inv_features2 = inv_features.detach().cpu().clone().numpy().copy()
            if self.invmapping_args['early_stopping']:
                if not best_loss:
                    best_loss = monitor.item()
                    best_inv_features = inv_features2
                    freeze_step = 0
                elif best_loss >= monitor.item():
                    best_loss = monitor.item()
                    best_inv_features = inv_features2
                    freeze_step = 0
                else:
                    freeze_step += 1
                    if freeze_step == 10:
                        break
            else:
                if not best_loss:
                    best_loss = monitor.item()
                    best_inv_features = inv_features2
                elif best_loss >= monitor.item():
                    best_loss = monitor.item()
                    best_inv_features = inv_features2
                else:
                    pass

            _log_invmap[step] = {'loss': total_loss.item(),
                                 'updated_features': inv_features.detach().cpu().numpy()}

            if (step) % 1 == 0 and self.invmapping_args['inv_vis']:
                _loss = round(total_loss.item(), 4)
                _str = f'[{step}] loss:{_loss}'
                for key in loss_cls:
                    _loss_cls = round(loss_cls[key].item(), 4)
                    _str += f', loss_{key} : {_loss_cls}'
                noises = round(regular_l1.item(), 4)
                _str += f', Sum_of_z:{noises}'
                print(_str)

        return best_inv_features, _log_invmap
