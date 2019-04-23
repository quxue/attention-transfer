# -*- coding:utf-8 -*-
from nested_dict import nested_dict
from functools import partial
import torch
from torch.nn.init import *
from torch.nn.parallel._functions import Broadcast
from torch.nn.parallel import scatter, parallel_apply, gather
import torch.nn.functional as F

#完成蒸馏操作，加上T=4的交叉熵
#y:学生网络的输出
#teacher_scores:教师网络的输出
#lable：样本的真实标签
def distillation(y, teacher_scores, labels, T, alpha):
    p = F.log_softmax(y/T, dim=1)	#学生网络的soft target
    q = F.softmax(teacher_scores/T, dim=1)	#教师网络的soft-target
    l_kl = F.kl_div(p, q, size_average=False) * (T**2) / y.shape[0] #两个soft-target的距离
    l_ce = F.cross_entropy(y, labels) #学生网络和真实标签的交叉熵
    return l_kl * alpha + l_ce * (1. - alpha)


def at(x):
	#.mean(1):对行进行压缩，返回一个列
	#.view():拉伸为一行
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

#AT的损失函数后面那部分
def at_loss(x, y):
    return (at(x) - at(y)).pow(2).mean()

#变量类型转为float
def cast(params, dtype='float'):
    if isinstance(params, dict):
    	#如果param的形式是字典
    	#把items里面的value值变为float形式
        return {k: cast(v, dtype) for k,v in params.items()}
    else:
    	#不是字典的形式
    	#getattr用于返回一个对象属性值,并转为float
        return getattr(params.cuda() if torch.cuda.is_available() else params, dtype)()

#针对Relu的初始化方法，返回的是一个tensor，里面装的是初始化的参数
def conv_params(ni, no, k=1):
    return torch.nn.init.kaiming_normal(torch.Tensor(no, ni, k, k))


def linear_params(ni, no):
	#初始化线性关系的权重和偏置
    return {'weight': torch.nn.init.kaiming_normal(torch.Tensor(no, ni)), 'bias': torch.zeros(no)}


#BN的参数初始化，权重，偏置，均值，方差
#BN层参数初始化
def bnparams(n):
    return {'weight': torch.rand(n),
            'bias': torch.zeros(n),
            'running_mean': torch.zeros(n),
            'running_var': torch.ones(n)}

#处理多GPU数据并行？反正不重要
def data_parallel(f, input, params, mode, device_ids, output_device=None):
    device_ids = list(device_ids)
    if output_device is None:
        output_device = device_ids[0]

    if len(device_ids) == 1:
        return f(input, params, mode)

    params_all = Broadcast.apply(device_ids, *params.values())
    params_replicas = [{k: params_all[i + j*len(params)] for i, k in enumerate(params.keys())}
                       for j in range(len(device_ids))]

    replicas = [partial(f, params=p, mode=mode)
                for p in params_replicas]
    inputs = scatter([input], device_ids)
    outputs = parallel_apply(replicas, inputs)
    return gather(outputs, output_device)


def flatten(params):
	 #nested_dict：用于多次分组。生成一个深度的字典，比如a["name"][1]["age"]=30
	 #对长度不一样的dict进行操作
    return {'.'.join(k): v for k, v in nested_dict(params).items_flat() if v is not None}


def batch_norm(x, params, base, mode):
    return F.batch_norm(x, weight=params[base + '.weight'],
                        bias=params[base + '.bias'],
                        running_mean=params[base + '.running_mean'],
                        running_var=params[base + '.running_var'],
                        training=mode)

#打印参数
def print_tensor_dict(params):
    kmax = max(len(key) for key in params.keys())
    for i, (key, v) in enumerate(params.items()):
        print(str(i).ljust(5), key.ljust(kmax + 3), str(tuple(v.shape)).ljust(23), torch.typename(v), v.requires_grad)

#endswith方法用于判断字符串是否以指定后缀结尾，如果以指定后缀结尾返回True，否则返回False
def set_requires_grad_except_bn_(params):
	#将那些不是以方差和标准差结尾的参数，设置为可求导
    for k, v in params.items():
        if not k.endswith('running_mean') and not k.endswith('running_var'):
            v.requires_grad = True
