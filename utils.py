import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import linear_sum_assignment
import os
import os.path
import torch.nn.functional as F


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

# 单纯为了方便计算损失的均值从而定义的函数
# 每次只需要将新的损失加入到这个对象中，调用update函数就可以实时更新


class AverageMeter(object):

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    # 初始状态下。将对象的值清零，保证后期可以计算均值

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    # 将当前的值传入对象中，自动更新对象的均值

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    # 打印当前的均值

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target):

    num_correct = np.sum(output == target)
    res = num_correct / len(target)

    return res

# 计算预测的精度


def cluster_acc(y_pred, y_true):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    return w[row_ind, col_ind].sum() / y_pred.size

# 计算正则化损失


def entropy(x):
    """ 
    Helper function to compute the entropy over the batch 
    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """
    EPS = 1e-8
    x_ = torch.clamp(x, min=EPS)
    b = x_ * torch.log(x_)

    if len(b.size()) == 2:  # Sample-wise entropy
        return - b.sum(dim=1).mean()
    elif len(b.size()) == 1:  # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' % (len(b.size())))

# 定义的第一个损失函数，可以自适应的控制有标签数据的训练
# 从而使得有标签数据的训练和新类聚类之间的速度变得尽可能统一
# 防止出现一快一慢的情况

# 计算样本之间的距离


class MarginLoss(nn.Module):
    # 这三个参数各有什么用
    def __init__(self, m=0.2, weight=None, s=10):
        super(MarginLoss, self).__init__()
        # 用来度量样本之间的距离
        self.m = m
        # 控制交叉熵损失的缩放系数
        self.s = s
        # 优化参数，应该是分类头的参数
        self.weight = weight

    def forward(self, x, target):
        # 自适应的调整网络对于有标签数据的预测值，使其学习的不是那么快
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        x_m = x - self.m * self.s

        # 重新调整过的输出与目标之间计算交叉熵损失函数
        output = torch.where(index, x_m, x)
        return F.cross_entropy(output, target, weight=self.weight)
