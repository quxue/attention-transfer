# -*- coding:utf-8 -*-
"""
    PyTorch training code for
    "Paying More Attention to Attention: Improving the Performance of
                Convolutional Neural Networks via Attention Transfer"
    https://arxiv.org/abs/1612.03928
    
    This file includes:
     * CIFAR ResNet and Wide ResNet training code which exactly reproduces
       https://github.com/szagoruyko/wide-residual-networks
     * Activation-based attention transfer
     * Knowledge distillation implementation

    2017 Sergey Zagoruyko
"""

import argparse
import os
import json
import numpy as np
from tqdm import tqdm
import torch
from torch.optim import SGD
import torchvision.transforms as T
from torchvision import datasets
import torch.nn.functional as F
import torchnet as tnt
from torchnet.engine import Engine
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import utils
from torch.autograd import Variable
#�������������ά�Ȼ������ϱ仯����,��������һ��flag������������Ч��
cudnn.benchmark = True
#��װWRN����
parser = argparse.ArgumentParser(description='Wide Residual Networks')
# Model options
#WRN-16-1
parser.add_argument('--depth', default=16, type=int)
parser.add_argument('--width', default=1, type=float)
parser.add_argument('--dataset', default='CIFAR10', type=str)
parser.add_argument('--dataroot', default='.', type=str)
parser.add_argument('--dtype', default='float', type=str)
parser.add_argument('--nthread', default=4, type=int)
parser.add_argument('--teacher_id', default='', type=str)

# Training options
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--weight_decay', default=0.0005, type=float)
parser.add_argument('--epoch_step', default='[60,120,160]', type=str,
                    help='json list with epochs to drop lr on')
parser.add_argument('--lr_decay_ratio', default=0.2, type=float)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--randomcrop_pad', default=4, type=float)
parser.add_argument('--temperature', default=4, type=float)
parser.add_argument('--alpha', default=0, type=float)
parser.add_argument('--beta', default=0, type=float)

# Device options
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--save', default='', type=str,
                    help='save parameters and logs in this folder')
parser.add_argument('--ngpu', default=1, type=int,
                    help='number of GPUs to use for training')
parser.add_argument('--gpu_id', default='1', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

#����Ԥ����
def create_dataset(opt, train):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                    np.array([63.0, 62.1, 66.7]) / 255.0),
    ])
#ѵ��ʱ����Ҫ����ˮƽ��ת�����crop
    if train:
        transform = T.Compose([
            T.Pad(4, padding_mode='reflect'),
            T.RandomHorizontalFlip(),
            T.RandomCrop(32),
            transform
        ])
    return getattr(datasets, opt.dataset)(opt.dataroot, train=train, download=True, transform=transform)

#����WRM
def resnet(depth, width, num_classes):
    assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
    #nΪ�в�����
    n = (depth - 4) // 6
    #widthsΪ����ÿһ���filter����
    widths = [int(v * width) for v in (16, 32, 64)]
#��ʼ�������ɲв��Ĳ���
    def gen_block_params(ni, no):
        return {
            'conv0': utils.conv_params(ni, no, 3),
            'conv1': utils.conv_params(no, no, 3),
            'bn0': utils.bnparams(ni),
            'bn1': utils.bnparams(no),
            'convdim': utils.conv_params(ni, no, 1) if ni != no else None,
        }
#��ʼ�������ɲв���Ĳ�����������count���в��
    def gen_group_params(ni, no, count):
        return {'block%d' % i: gen_block_params(ni if i == 0 else no, no)
                for i in range(count)}
#��ʼ�������������ܵĲ��������-��0-��1-��2-
#flatten�������е�paramƽ�̳���
    flat_params = utils.cast(utils.flatten({
        'conv0': utils.conv_params(3, 16, 3),
        'group0': gen_group_params(16, widths[0], n),
        #ÿһ��Ŀ�Ȳ�һ��
        'group1': gen_group_params(widths[0], widths[1], n),
        'group2': gen_group_params(widths[1], widths[2], n),
        'bn': utils.bnparams(widths[2]),
        'fc': utils.linear_params(widths[2], num_classes),
    }))

    utils.set_requires_grad_except_bn_(flat_params)
#��һ���в��
#RELU-���-relu-���
    def block(x, params, base, mode, stride):
        o1 = F.relu(utils.batch_norm(x, params, base + '.bn0', mode), inplace=True)
        y = F.conv2d(o1, params[base + '.conv0'], stride=stride, padding=1)
        o2 = F.relu(utils.batch_norm(y, params, base + '.bn1', mode), inplace=True)
        z = F.conv2d(o2, params[base + '.conv1'], stride=1, padding=1)
#�������bottlneck?1*1�ľ����
        if base + '.convdim' in params:
            return z + F.conv2d(o1, params[base + '.convdim'], stride=stride)
        else:
            return z + x
#��һ���в���
    def group(o, params, base, mode, stride):
        for i in range(n):
            #o = block(o, params, f'{base}.block{i}',mode,stride if i == 0 else 1)
            o = block(o, params, '%s.block%d' % (base, i), mode, stride if i == 0 else 1)
        return o
#����������
#���-��һ��-�ڶ���-������-relu-�ػ�-��ƽ����ȫ���ӣ�
    def f(input, params, mode, base=''):
        x = F.conv2d(input, params[base+'conv0'], padding=1)
        g0 = group(x, params, base+'group0', mode, 1)
        g1 = group(g0, params, base+'group1', mode, 2)
        g2 = group(g1, params, base+'group2', mode, 2)
        o = F.relu(utils.batch_norm(g2, params, base+'bn', mode))
        o = F.avg_pool2d(o, 8, 1, 0)
#�����е�tensor����Ϊһ��
        o = o.view(o.size(0), -1)
        #o = F.linear(o, params[base+'Connection timed outfc.weight'], params[base+'fc.bias'])
        o = F.linear(o, params[base+'fc.weight'], params[base+'fc.bias'])
        return o, (g0, g1, g2)
        #���ص�������������Լ�ÿ���������tuple
    return f, flat_params
    #���ص���һ����������Լ����в���


def main():
    opt = parser.parse_args()
    #��ӡ����
    #print('parsed options:', vars(opt))
    #�ı�lr�Ĳ���

    epoch_step = json.loads(opt.epoch_step)
    #����������
    num_classes = 10 if opt.dataset == 'CIFAR10' else 100

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
#�������ݼ�
    def create_iterator(mode):
        return DataLoader(create_dataset(opt, mode), opt.batch_size, shuffle=mode,
                          num_workers=opt.nthread, pin_memory=torch.cuda.is_available())

    #����ѵ����
    train_loader = create_iterator(False)
    #���ز��Լ�
    test_loader = create_iterator(False)


    # deal with student first
    #��һ����������ʦ���磬40-1,16-2,40-1
    #�ڶ����͵������������16-1��ѧ������
    f_s, params_s = resnet(opt.depth, opt.width, num_classes)
    print("1111,student",params_s)

    # deal with teacher
    #��֮ǰѵ���õ���������������������һ����ʦ����
    #����֮ǰѵ���õĽ�ʦ�����ģ�Ͳ���
    if opt.teacher_id:
        #��һ�ε�teacher_id��resent_16_2_teacher
        #��teacher_id�ж�ȡ����
        with open(os.path.join('logs', opt.teacher_id, 'log.txt'), 'r') as ff:
            line = ff.readline()
            r = line.find('json_stats')
            info = json.loads(line[r + 12:])
        #�ڶ�������һ��16-2�Ľ�ʦ������
        f_t = resnet(info['depth'], info['width'], num_classes)[0]
        #��ȡģ�Ͳ���
        model_data = torch.load(os.path.join('logs', opt.teacher_id, 'model.pt7'))
        params_t = model_data['params']

        # merge teacher and student params
        # �ϲ�ѧ������ͽ�ʦ����Ĳ���,ΪʲôҪ�ϲ�����69�����139
        params = {'student.' + k: v for k, v in params_s.items()}
        print("student",params)
        for k, v in params_t.items():
            params['teacher.' + k] = v.detach().requires_grad_(False)
        #�ڵڶ����ֽ�����f
        #ִ����ʧ�����Ľ�ģ
        def f(inputs, params, mode):
            #y_s��ѧ�������������
            #g_s��ÿ���������tuple
            y_s, g_s = f_s(inputs, params, mode, 'student.')
            with torch.no_grad():
                #y_t����ʦ�����������
                #g_t:��ʦ����ÿ���������tuple
                y_t, g_t = f_t(inputs, params, False, 'teacher.')
            #���ص���ѧ������ͽ�ʦ�����������
            #����������ʦ������attention��attention map���
            return y_s, y_t, [utils.at_loss(x, y) for x, y in zip(g_s, g_t)]
    else:
    #��һ��ִ�е���else
    #f:��Ӧ�����f������������ѧ������Ĵ��ֵ��f
    #params����ѧ�������ȫ��������ֵ��params
        f, params = f_s, params_s

#�Ż�,���ģ���Ժ������Ż�����
    def create_optimizer(opt, lr):
        print('creating optimizer with lr = ', lr)
        return SGD((v for v in params.values() if v.requires_grad), lr,
                   momentum=0.9, weight_decay=opt.weight_decay)
    optimizer =create_optimizer(opt, opt.lr)
    epoch = 0
    #����ģ�Ͳ���
    #��ʼ��resume����ŵ���" "
    #�ڶ�������ŵ�Ҳ�� ' '
    if opt.resume != '':
        print("come in resume")
        #��ѵ��ѧ������ʱ�������ȥ���if
        state_dict = torch.load(opt.resume)
        epoch = state_dict['epoch']
        params_tensors = state_dict['params']
        for k, v in params.items():
            v.data.copy_(params_tensors[k])
        optimizer.load_state_dict(state_dict['optimizer'])
    #print('\nParameters:')
    #utils.print_tensor_dict(params[])
    n_parameters = sum(p.numel() for p in list(params_s.values()))
    #print(r'\nTotal number of parameters:', n_parameters)
#����ģ�ͣ������ʧ������
    #����ͳ��������ӵı����ķ���;�ֵ��������������ƽ����ʧ��
    #�������ϵ��tensor����
    meter_loss = tnt.meter.AverageValueMeter()
    #ͳ�Ʒ������
    classacc = tnt.meter.ClassErrorMeter(accuracy=True)
    #���Meter����ͳ��events֮���ʱ�䣬Ҳ��������ͳ��batch���ݵ�ƽ����������
    timer_train = tnt.meter.TimeMeter('s')
    timer_test = tnt.meter.TimeMeter('s')
    #ƽ����ʧ
    meters_at = [tnt.meter.AverageValueMeter() for i in range(3)]
#����ģ�ʹ洢���ļ���
    if not os.path.exists(opt.save):
        os.mkdir(opt.save)
#ģ��ѵ��
#�����˱����ĵ�˼��
    def h(sample):
        #input ����������
        #target�Ǳ�ǩ
        inputs = utils.cast(sample[0], opt.dtype).detach()
        targets = utils.cast(sample[1], 'long')
        #���ģ����ѧ��ģ��
        #�ø�������ʧ����ѵ��
        if opt.teacher_id != '':
            y_s, y_t, loss_groups = utils.data_parallel(f, inputs, params, sample[2], range(opt.ngpu))
            #ȡ���ܵ�loss
            loss_groups = [v.sum() for v in loss_groups]
            #�ܵ���ʧ��
            [m.add(v.item()) for m, v in zip(meters_at, loss_groups)]
            #��һ����������#y_s:ѧ����������#y_t����ʦ��������#target����ʵ��ǩ
            #�ڶ�������AD��ʧ��������
            #����������ѧ����������
            #����AT�㷨ʱ��alpha����0����һ���֡���ʣ����ѧ���������ʵ��ǩ�Ľ�����
            #��ΪKD�㷨ʱ��beta����0����ʣ������ʧ����,�����ʵ�ִ�1�ӵ�c
            return utils.distillation(y_s, y_t, targets, opt.temperature, opt.alpha) \
                    + opt.beta * sum(loss_groups), y_s
        #����ǽ�ʦ����
        #�ñ�׼������ѵ��
        else:
            #y����������
            y = utils.data_parallel(f, inputs, params, sample[2], range(opt.ngpu))[0]
            return F.cross_entropy(y, targets), y
#�洢����
    def log(t, state):
        torch.save(dict(params={k: v.data for k, v in params.items()},
                        optimizer=state['optimizer'].state_dict(),
                        epoch=t['epoch']),
                   os.path.join(opt.save, 'model.pt7'))
        z = vars(opt).copy(); z.update(t)
        logname = os.path.join(opt.save, 'log.txt')
        with open(logname, 'a') as f:
            f.write('json_stats: ' + json.dumps(z) + '\n')
        print(z)
#state�Ǹ�tensor
#ÿ�β���һ������֮��Ĳ���
    def on_sample(state):
        state['sample'].append(state['train'])
# ��model:forward()֮��Ĳ�
    def on_forward(state):
        classacc.add(state['output'].data, state['sample'][1])
        meter_loss.add(state['loss'].item())
#����ѵ����ʼǰ�����úͳ�ʼ��
    def on_start(state):

        state['epoch'] = epoch
#ÿһ��epoch��ʼʱ�Ĳ���
    def on_start_epoch(state):
        classacc.reset()
        meter_loss.reset()
        timer_train.reset()
        [meter.reset() for meter in meters_at]
        state['iterator'] = tqdm(train_loader)

        epoch = state['epoch'] + 1
        if epoch in epoch_step:
            lr = state['optimizer'].param_groups[0]['lr']
            state['optimizer'] = create_optimizer(opt, lr * opt.lr_decay_ratio)
#ÿһ��epoch����ʱ�Ĳ���
    def on_end_epoch(state):
        train_loss = meter_loss.mean
        train_acc = classacc.value()
        train_time = timer_train.value()
        meter_loss.reset()
        classacc.reset()
        timer_test.reset()

        engine.test(h, test_loader)

        test_acc = classacc.value()[0]
        print(log({
            "train_loss": train_loss,
            "train_acc": train_acc[0],
            "test_loss": meter_loss.mean,
            "test_acc": test_acc,
            "epoch": state['epoch'],
            "num_classes": num_classes,
            "n_parameters": n_parameters,
            "train_time": train_time,
            "test_time": timer_test.value(),
            "at_losses": [m.value() for m in meters_at],
           }, state))
        print('==> id: %s (%d/%d), test_acc: \33[91m%.2f\033[0m' % \
                       (opt.save, state['epoch'], opt.epochs, test_acc))
#Engine��ѵ�������ṩ��һ��ģ�壬��ģ�彨����model��DatasetIterator��Criterion��Meter֮�����ϵ
#ѵ�����磬�õ��Զ����ѵ�����̽ӿڣ�hook
    engine = Engine()
# ÿ�β���һ������֮��Ĳ���
    engine.hooks['on_sample'] = on_sample
# ��model:forward()֮��Ĳ���
    engine.hooks['on_forward'] = on_forward
    #ÿһ��epochǰ�Ĳ���
    engine.hooks['on_start_epoch'] = on_start_epoch
    #ÿһ��epoch����ʱ�Ĳ���
    engine.hooks['on_end_epoch'] = on_end_epoch
    #����ѵ����ʼǰ�����úͳ�ʼ��
    engine.hooks['on_start'] = on_start
    #��ʼѵ������,����100��
    engine.train(h, train_loader, opt.epochs, optimizer)


if __name__ == '__main__':
    main()
