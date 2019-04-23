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
#网络的输入数据维度或类型上变化不大,设置这样一个flag可以增加运行效率
cudnn.benchmark = True
#封装WRN名称
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

#数据预处理
def create_dataset(opt, train):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                    np.array([63.0, 62.1, 66.7]) / 255.0),
    ])
#训练时，还要加入水平翻转，随机crop
    if train:
        transform = T.Compose([
            T.Pad(4, padding_mode='reflect'),
            T.RandomHorizontalFlip(),
            T.RandomCrop(32),
            transform
        ])
    return getattr(datasets, opt.dataset)(opt.dataroot, train=train, download=True, transform=transform)

#创建WRM
def resnet(depth, width, num_classes):
    assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
    #n为残差块个数
    n = (depth - 4) // 6
    #widths为网络每一层的filter个数
    widths = [int(v * width) for v in (16, 32, 64)]
#初始化，生成残差块的参数
    def gen_block_params(ni, no):
        return {
            'conv0': utils.conv_params(ni, no, 3),
            'conv1': utils.conv_params(no, no, 3),
            'bn0': utils.bnparams(ni),
            'bn1': utils.bnparams(no),
            'convdim': utils.conv_params(ni, no, 1) if ni != no else None,
        }
#初始化，生成残差组的参数，里面有count个残差块
    def gen_group_params(ni, no, count):
        return {'block%d' % i: gen_block_params(ni if i == 0 else no, no)
                for i in range(count)}
#初始化，整个网络框架的参数，卷积-组0-组1-组2-
#flatten：将所有的param平铺出来
    flat_params = utils.cast(utils.flatten({
        'conv0': utils.conv_params(3, 16, 3),
        'group0': gen_group_params(16, widths[0], n),
        #每一层的宽度不一样
        'group1': gen_group_params(widths[0], widths[1], n),
        'group2': gen_group_params(widths[1], widths[2], n),
        'bn': utils.bnparams(widths[2]),
        'fc': utils.linear_params(widths[2], num_classes),
    }))

    utils.set_requires_grad_except_bn_(flat_params)
#大一个残差块
#RELU-卷积-relu-卷积
    def block(x, params, base, mode, stride):
        o1 = F.relu(utils.batch_norm(x, params, base + '.bn0', mode), inplace=True)
        y = F.conv2d(o1, params[base + '.conv0'], stride=stride, padding=1)
        o2 = F.relu(utils.batch_norm(y, params, base + '.bn1', mode), inplace=True)
        z = F.conv2d(o2, params[base + '.conv1'], stride=1, padding=1)
#加入的是bottlneck?1*1的卷积层
        if base + '.convdim' in params:
            return z + F.conv2d(o1, params[base + '.convdim'], stride=stride)
        else:
            return z + x
#搭一个残差组
    def group(o, params, base, mode, stride):
        for i in range(n):
            #o = block(o, params, f'{base}.block{i}',mode,stride if i == 0 else 1)
            o = block(o, params, '%s.block%d' % (base, i), mode, stride if i == 0 else 1)
        return o
#搭整个网络
#卷积-第一组-第二组-第三组-relu-池化-扁平化（全连接）
    def f(input, params, mode, base=''):
        x = F.conv2d(input, params[base+'conv0'], padding=1)
        g0 = group(x, params, base+'group0', mode, 1)
        g1 = group(g0, params, base+'group1', mode, 2)
        g2 = group(g1, params, base+'group2', mode, 2)
        o = F.relu(utils.batch_norm(g2, params, base+'bn', mode))
        o = F.avg_pool2d(o, 8, 1, 0)
#将多行的tensor，变为一行
        o = o.view(o.size(0), -1)
        #o = F.linear(o, params[base+'Connection timed outfc.weight'], params[base+'fc.bias'])
        o = F.linear(o, params[base+'fc.weight'], params[base+'fc.bias'])
        return o, (g0, g1, g2)
        #返回的是最后的输出，以及每个组输出的tuple
    return f, flat_params
    #返回的是一个搭建函数，以及所有参数


def main():
    opt = parser.parse_args()
    #打印参数
    #print('parsed options:', vars(opt))
    #改变lr的步数

    epoch_step = json.loads(opt.epoch_step)
    #输出的类别数
    num_classes = 10 if opt.dataset == 'CIFAR10' else 100

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
#下载数据集
    def create_iterator(mode):
        return DataLoader(create_dataset(opt, mode), opt.batch_size, shuffle=mode,
                          num_workers=opt.nthread, pin_memory=torch.cuda.is_available())

    #加载训练集
    train_loader = create_iterator(False)
    #加载测试集
    test_loader = create_iterator(False)


    # deal with student first
    #第一步搭三个教师网络，40-1,16-2,40-1
    #第二步和第三步，搭的是16-1的学生网络
    f_s, params_s = resnet(opt.depth, opt.width, num_classes)
    print("1111,student",params_s)

    # deal with teacher
    #用之前训练好的三个网络的网络参数，搭一个教师网络
    #加载之前训练好的教师网络的模型参数
    if opt.teacher_id:
        #第一次的teacher_id是resent_16_2_teacher
        #从teacher_id中读取参数
        with open(os.path.join('logs', opt.teacher_id, 'log.txt'), 'r') as ff:
            line = ff.readline()
            r = line.find('json_stats')
            info = json.loads(line[r + 12:])
        #第二步搭了一个16-2的教师网络框架
        f_t = resnet(info['depth'], info['width'], num_classes)[0]
        #提取模型参数
        model_data = torch.load(os.path.join('logs', opt.teacher_id, 'model.pt7'))
        params_t = model_data['params']

        # merge teacher and student params
        # 合并学生网络和教师网络的参数,为什么要合并，从69变成了139
        params = {'student.' + k: v for k, v in params_s.items()}
        print("student",params)
        for k, v in params_t.items():
            params['teacher.' + k] = v.detach().requires_grad_(False)
        #在第二部分进入了f
        #执行损失函数的建模
        def f(inputs, params, mode):
            #y_s：学生网络最后的输出
            #g_s：每个组输出的tuple
            y_s, g_s = f_s(inputs, params, mode, 'student.')
            with torch.no_grad():
                #y_t：教师网络最后的输出
                #g_t:教师网络每个组输出的tuple
                y_t, g_t = f_t(inputs, params, False, 'teacher.')
            #返回的是学生网络和教师网络最后的输出
            #第三部分是师生网络attention的attention map结果
            return y_s, y_t, [utils.at_loss(x, y) for x, y in zip(g_s, g_t)]
    else:
    #第一步执行的是else
    #f:对应上面的f函数，将整个学生网络的搭建赋值给f
    #params：将学生网络的全部参数赋值给params
        f, params = f_s, params_s

#优化,搭好模型以后，设置优化函数
    def create_optimizer(opt, lr):
        print('creating optimizer with lr = ', lr)
        return SGD((v for v in params.values() if v.requires_grad), lr,
                   momentum=0.9, weight_decay=opt.weight_decay)
    optimizer =create_optimizer(opt, opt.lr)
    epoch = 0
    #加载模型参数
    #开始是resume里面放的是" "
    #第二步里面放的也是 ' '
    if opt.resume != '':
        print("come in resume")
        #在训练学生网络时，不会进去这个if
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
#评估模型，获得损失函数？
    #用于统计任意添加的变量的方差和均值，可以用来测量平均损失等
    #输出是以系列tensor矩阵
    meter_loss = tnt.meter.AverageValueMeter()
    #统计分类误差
    classacc = tnt.meter.ClassErrorMeter(accuracy=True)
    #这个Meter用于统计events之间的时间，也可以用来统计batch数据的平均处理数据
    timer_train = tnt.meter.TimeMeter('s')
    timer_test = tnt.meter.TimeMeter('s')
    #平均损失
    meters_at = [tnt.meter.AverageValueMeter() for i in range(3)]
#创建模型存储的文件夹
    if not os.path.exists(opt.save):
        os.mkdir(opt.save)
#模型训练
#加入了本论文的思想
    def h(sample):
        #input 是输入样本
        #target是标签
        inputs = utils.cast(sample[0], opt.dtype).detach()
        targets = utils.cast(sample[1], 'long')
        #如果模型是学生模型
        #用给出的损失函数训练
        if opt.teacher_id != '':
            y_s, y_t, loss_groups = utils.data_parallel(f, inputs, params, sample[2], range(opt.ngpu))
            #取出总的loss
            loss_groups = [v.sum() for v in loss_groups]
            #总的损失？
            [m.add(v.item()) for m, v in zip(meters_at, loss_groups)]
            #第一部分是蒸馏#y_s:学生网络的输出#y_t：教师网络的输出#target：真实标签
            #第二部分是AD损失函数部分
            #第三部分是学生网络的输出
            #当是AT算法时，alpha等于0，第一部分。就剩的是学生网络和真实标签的交叉熵
            #当为KD算法时，beta等于0，就剩蒸馏损失函数,在这儿实现从1加到c
            return utils.distillation(y_s, y_t, targets, opt.temperature, opt.alpha) \
                    + opt.beta * sum(loss_groups), y_s
        #如果是教师网络
        #用标准交叉熵训练
        else:
            #y是网络的输出
            y = utils.data_parallel(f, inputs, params, sample[2], range(opt.ngpu))[0]
            return F.cross_entropy(y, targets), y
#存储参数
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
#state是个tensor
#每次采样一个样本之后的操作
    def on_sample(state):
        state['sample'].append(state['train'])
# 在model:forward()之后的操
    def on_forward(state):
        classacc.add(state['output'].data, state['sample'][1])
        meter_loss.add(state['loss'].item())
#用于训练开始前的设置和初始化
    def on_start(state):

        state['epoch'] = epoch
#每一个epoch开始时的操作
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
#每一个epoch结束时的操作
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
#Engine给训练过程提供了一个模板，该模板建立了model，DatasetIterator，Criterion和Meter之间的联系
#训练网络，用的自定义的训练过程接口：hook
    engine = Engine()
# 每次采样一个样本之后的操作
    engine.hooks['on_sample'] = on_sample
# 在model:forward()之后的操作
    engine.hooks['on_forward'] = on_forward
    #每一个epoch前的操作
    engine.hooks['on_start_epoch'] = on_start_epoch
    #每一个epoch结束时的操作
    engine.hooks['on_end_epoch'] = on_end_epoch
    #用于训练开始前的设置和初始化
    engine.hooks['on_start'] = on_start
    #开始训练网络,迭代100次
    engine.train(h, train_loader, opt.epochs, optimizer)


if __name__ == '__main__':
    main()
