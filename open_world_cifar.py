from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torchvision
import torch.utils.data as data
from torchvision import transforms
import itertools
from torch.utils.data.sampler import Sampler


class OPENWORLDCIFAR100(torchvision.datasets.CIFAR100):

    def __init__(self, root, labeled=True, labeled_num=50, labeled_ratio=0.5, rand_number=0, transform=None, target_transform=None,
                 download=False, unlabeled_idxs=None):
        super(OPENWORLDCIFAR100, self).__init__(
            root, True, transform, target_transform, download)

        downloaded_list = self.train_list
        self.data = []
        self.targets = []
        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        labeled_classes = range(labeled_num)
        np.random.seed(rand_number)

        if labeled:
            self.labeled_idxs, self.unlabeled_idxs = self.get_labeled_index(
                labeled_classes, labeled_ratio)
            self.shrink_data(self.labeled_idxs)
        else:
            self.shrink_data(unlabeled_idxs)

    def get_labeled_index(self, labeled_classes, labeled_ratio):
        labeled_idxs = []
        unlabeled_idxs = []
        for idx, label in enumerate(self.targets):
            if label in labeled_classes and np.random.rand() < labeled_ratio:
                labeled_idxs.append(idx)
            else:
                unlabeled_idxs.append(idx)
        return labeled_idxs, unlabeled_idxs

    def shrink_data(self, idxs):
        targets = np.array(self.targets)
        self.targets = targets[idxs].tolist()
        self.data = self.data[idxs, ...]
# 以cifar10为例


class OPENWORLDCIFAR10(torchvision.datasets.CIFAR10):
    # 在init函数中接收传递过来的参数，默认不需要下载，但是传递进来的参数需要下载
    def __init__(self, root, labeled=True, labeled_num=5, labeled_ratio=0.5, rand_number=0, transform=None, target_transform=None,
                 download=False, unlabeled_idxs=None):
        # 这一句话可以从标准的cifar10数据集中下载数据，这是从父类中下载
        # 下载好的数据集存在self中
        super(OPENWORLDCIFAR10, self).__init__(
            root, True, transform, target_transform, download)
        # 拿到数据集中原始的MD5值，从而判断数据集是否在下载的过程中损坏
        downloaded_list = self.train_list
        self.data = []
        self.targets = []
        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            # 构建数据的完整路径
            file_path = os.path.join(self.root, self.base_folder, file_name)
            # 按照文件路径打开文件，拿到数据和标签
            with open(file_path, 'rb') as f:
                # 加载数据，latin1格式
                entry = pickle.load(f, encoding='latin1')
                # 依次取出数据和标签
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        # 转换样本的维度和形状
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        # 设置样本随机抽取标签数据的范围以及设置随机种子
        # 固定有标记数据是前五个类
        labeled_classes = range(labeled_num)
        np.random.seed(rand_number)
        # 获取标记数据和未标记数据
        if labeled:
            self.labeled_idxs, self.unlabeled_idxs = self.get_labeled_index(
                labeled_classes, labeled_ratio)
            self.shrink_data(self.labeled_idxs)
        # 只获取未标记数据，数据集中只保留无标签数据
        else:
            self.shrink_data(unlabeled_idxs)
    # 随机抽样获得样本的标签
    # 抽样之后得到标记数据和未标记数据

    # 根据给定的范围和比例，拿到标记数据和未标记数据的索引
    def get_labeled_index(self, labeled_classes, labeled_ratio):
        labeled_idxs = []
        unlabeled_idxs = []
        # 根据标签拿到索引
        for idx, label in enumerate(self.targets):
            # 当前类别数据前五个类，并且当前的概率小于给定的概率0.5
            # 也就是说，前五个类中，有一半的数据成为了有标签数据，剩下的一半成为了无标签数据
            # 从而有标签数据由四分之一，无标签数据由四分之三，其中无标签数据中的四分之一数据是有类别的，只是没有标签
            # 从而数据集更加符合真实情况
            if label in labeled_classes and np.random.rand() < labeled_ratio:
                labeled_idxs.append(idx)
            else:
                unlabeled_idxs.append(idx)
        return labeled_idxs, unlabeled_idxs

    def shrink_data(self, idxs):
        targets = np.array(self.targets)
        self.targets = targets[idxs].tolist()
        self.data = self.data[idxs, ...]


# 对数据集进行处理
# Dictionary of transforms
dict_transform = {
    'cifar_train': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ]),
    'cifar_test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])
}
