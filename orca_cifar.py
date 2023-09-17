import argparse
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import models
# 将自己定义的文件当成数据集，因为这个文件中对cifar数据集进行下载和详细的处理，形成了自己的数据集
import open_world_cifar as datasets
from utils import cluster_acc, AverageMeter, entropy, MarginLoss, accuracy, TransformTwice
from sklearn import metrics
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from itertools import cycle

# 按照论文所说，有标签的训练通过自适应阈值来控制快慢
# 无标签的训练从聚类改成了伪标签，新类分类的分类头提前定义好了
# 没有裁剪多余分类头的代码
# 其中m是测试阶段得到的模型的不确定性，从而动态的控制有标签数据的训练


def train(args, model, device, train_label_loader, train_unlabel_loader, optimizer, m, epoch, tf_writer):
    model.train()
    m = min(m, 0.5)
    # 定义两个损失函数
    # 这个损失函数是计算成对目标之间的损失
    bce = nn.BCELoss()
    # 这个是自适应边界损失
    ce = MarginLoss(m=-1*m)
    # 这个是第三个损失
    # entropy_loss = entropy(torch.mean(prob, 0))
    # 循环遍历数据集
    unlabel_loader_iter = cycle(train_unlabel_loader)
    # 保存损失的变量，后期一旦多了损失，直接调用update函数更新当前损失的均值
    bce_losses = AverageMeter('bce_loss', ':.4e')
    ce_losses = AverageMeter('ce_loss', ':.4e')
    entropy_losses = AverageMeter('entropy_loss', ':.4e')
    # 遍历完训练集中的所有数据，称为一次训练，也叫做一个epoch
    # 分别从有标签数据集和无标签数据集中取出数据，每个数据经过两次增强
    # 其中x和x2是同一图像的不同增强，也就是说他们对应同一个标签
    for batch_idx, ((x, x2), target) in enumerate(train_label_loader):
        # 无标签数据中，其实都有标签，只是不取标签，将其当成无标签数据处理
        ((ux, ux2), _) = next(unlabel_loader_iter)
        # 将一个有标签数据集中的样本和一个无标签数据集中的样本组合在一起进行训练
        # 应该是批次，相当于一个批次中既有有标签数据，又有无标签数据
        x = torch.cat([x, ux], 0)
        x2 = torch.cat([x2, ux2], 0)
        # 标签的维度
        labeled_len = len(target)

        x, x2, target = x.to(device), x2.to(device), target.to(device)
        # 直接将两种增强之后的样本输入进网络中，得到预测值
        output, feat = model(x)
        output2, feat2 = model(x2)
        # 得到分类的概率
        prob = F.softmax(output, dim=1)
        prob2 = F.softmax(output2, dim=1)

        '''这里在干什么？？？'''
        '''计算元素之间的余弦相似度'''
        '''为了找到最近的元素'''
        # 相当于根据x计算得到的输出从而计算余弦相似度矩阵
        feat_detach = feat.detach()  # 不再参与梯度更新
        feat_norm = feat_detach / \
            torch.norm(feat_detach, 2, 1, keepdim=True)  # 对输出的预测进行归一化处理
        cosine_dist = torch.mm(feat_norm, feat_norm.t())  # 计算各个元素之间的余弦相似度
        labeled_len = len(target)  # 一共有多少个元素

        pos_pairs = []
        target_np = target.cpu().numpy()

        '''训练分为两步'''
        '''第一步是有标签数据参与训练'''
        '''训练分为两步'''

        # label part
        '''构建正样本对的索引列表'''
        '''对于每一个有标签的类，从同类样本中选择一个样本构成正样本对'''
        # 对于当前取出的所有有标签样本进行遍历
        for i in range(labeled_len):
            # 这是当前样本的目标标签
            target_i = target_np[i]
            # 找到当前样本标签的索引值
            idxs = np.where(target_np == target_i)[0]
            # 只有一个样本与自己的标签相同
            if len(idxs) == 1:
                pos_pairs.append(idxs[0])
            # 有多个样本与自己的标签相同
            else:
                # 从与自己有着相同标签的样本中随机选择一个
                # 也就是从同类样本中随机选择一个
                selec_idx = np.random.choice(idxs, 1)
                while selec_idx == i:
                    selec_idx = np.random.choice(idxs, 1)
                pos_pairs.append(int(selec_idx))
        '''核心就是训练，如何与论文中的代码对应'''
        # unlabel part
        # 选择前两个余弦相似度最接近的样本
        # 相当于给无标签数据挑选伪标签
        unlabel_cosine_dist = cosine_dist[labeled_len:, :]
        #
        vals, pos_idx = torch.topk(unlabel_cosine_dist, 2, dim=1)
        # 选择最相似的两个样本中的第二个作为正样本对
        pos_idx = pos_idx[:, 1].cpu().numpy().flatten().tolist()
        pos_pairs.extend(pos_idx)

        pos_prob = prob2[pos_pairs, :]
        pos_sim = torch.bmm(prob.view(args.batch_size, 1, -1),
                            pos_prob.view(args.batch_size, -1, 1)).squeeze()
        ones = torch.ones_like(pos_sim)
        # 同时计算三个函数
        # 计算有标签的损失，只是加入了自适应的阈值，使得模型对于有标签的数据学习的速率变慢
        ce_loss = ce(output[:labeled_len], target)
        # 这个损失函数中用到了样本对计算损失
        # 将样本对于全1的tensor之间计算损失是为什么？
        bce_loss = bce(pos_sim, ones)
        '''这个损失还得细分'''
        # 计算最终的正则化损失
        entropy_loss = entropy(torch.mean(prob, 0))

        # 总损失是三个损失加起来
        loss = - entropy_loss + ce_loss + bce_loss

        bce_losses.update(bce_loss.item(), args.batch_size)
        ce_losses.update(ce_loss.item(), args.batch_size)
        entropy_losses.update(entropy_loss.item(), args.batch_size)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # 将计算的损失展示到tensorboard中
    tf_writer.add_scalar('loss/bce', bce_losses.avg, epoch)
    tf_writer.add_scalar('loss/ce', ce_losses.avg, epoch)
    tf_writer.add_scalar('loss/entropy', entropy_losses.avg, epoch)

# 测试阶段主要是得到模型的精度以及一个辅助模型训练的不确定性参数


def test(args, model, labeled_num, device, test_loader, epoch, tf_writer):
    model.eval()
    preds = np.array([])
    targets = np.array([])
    confs = np.array([])
    with torch.no_grad():
        for batch_idx, (x, label) in enumerate(test_loader):
            x, label = x.to(device), label.to(device)
            output, _ = model(x)
            prob = F.softmax(output, dim=1)
            # 得到最终预测的类别
            conf, pred = prob.max(1)
            targets = np.append(targets, label.cpu().numpy())
            preds = np.append(preds, pred.cpu().numpy())
            confs = np.append(confs, conf.cpu().numpy())
    targets = targets.astype(int)
    preds = preds.astype(int)

    # 用来区分哪些类是旧类，哪些类是新类
    seen_mask = targets < labeled_num
    unseen_mask = ~seen_mask
    # 由下面两个函数计算精度
    # 旧类的精度
    # 整体的精度
    overall_acc = cluster_acc(preds, targets)
    # 已知类别的额精度
    seen_acc = accuracy(preds[seen_mask], targets[seen_mask])
    # 新类的精度
    unseen_acc = cluster_acc(preds[unseen_mask], targets[unseen_mask])
    # 未知类别的归一化互信息
    unseen_nmi = metrics.normalized_mutual_info_score(
        targets[unseen_mask], preds[unseen_mask])
    # 通过置信度来计算模型的不确定性
    mean_uncert = 1 - np.mean(confs)
    print('Test overall acc {:.4f}, seen acc {:.4f}, unseen acc {:.4f}'.format(
        overall_acc, seen_acc, unseen_acc))
    tf_writer.add_scalar('acc/overall', overall_acc, epoch)
    tf_writer.add_scalar('acc/seen', seen_acc, epoch)
    tf_writer.add_scalar('acc/unseen', unseen_acc, epoch)
    tf_writer.add_scalar('nmi/unseen', unseen_nmi, epoch)
    tf_writer.add_scalar('uncert/test', mean_uncert, epoch)
    # 返回这个不确定性支持模型训练
    return mean_uncert

# 给定的代码无法复现，因为代码中的预训练模型没有给出


def main():
    parser = argparse.ArgumentParser(description='orca')
    # 指定在训练到第40个epoch和第80个epoch时，对学习率进行衰减
    parser.add_argument('--milestones', nargs='+',
                        type=int, default=[40, 80])
    parser.add_argument('--dataset', default='cifar10',
                        help='dataset setting')
    # 因为默认使用的是cifar10，所以默认有标签的数据只有一半，这是论文中的设置
    parser.add_argument('--labeled-num', default=5, type=int)
    parser.add_argument('--labeled-ratio', default=0.5, type=float)
    # 固定随机数种子，使得实验结果可以复现，固定之后，参数的随机初始化，数据集的shuffle等随机性都变得一致
    parser.add_argument('--seed', type=int, default=1,
                        metavar='S', help='random seed (default: 1)')
    parser.add_argument('--name', type=str, default='debug')
    parser.add_argument('--exp_root', type=str, default='./results/')
    # 默认训练100个epoch
    parser.add_argument('--epochs', type=int, default=100)
    # batch_size的默认大小
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N',
                        help='mini-batch size')
    # 将所有参数保存在args中，后期需要使用时，直接调用args即可
    args = parser.parse_args()
    # 定义训练所在的设备
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")
    print(f'use:{device}')
    # 新增一个参数，定义模型保存的路径
    args.savedir = os.path.join(args.exp_root, args.name)
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    # 使用已定义的函数对数据集进行进一步处理，使其符合论文中的设置
    # 处理数据集的文件就是open_world_cifar，文中没有使用imagenet的数据集，故没有定义对imagenet数据集的处理文件
    if args.dataset == 'cifar10':
        # 得到了一个新的数据集对象，里面的数据经过重新处理
        # 最终得到了一个一半标记数据，一半未标记数据，并且数据经过随机裁剪翻转之后形成了不同的视角，进一步扩充了数据集
        # labeled_num控制了前五个类中形成有标记数据
        # labeled_ratio控制前五个类中每个样本只有0.5的概率成为有标记数据
        # 从而有标签数据有四分之一，无标签数据有四分之三，其中无标签数据中的四分之一数据是有类别的，只是没有标签
        # 从而数据集更加符合真实情况，有标签数据和无标签数据分布重叠，但是不完全相同
        '''最重要的就是这个'''
        '''最重要的就是这个'''
        '''最重要的就是这个'''
        train_label_set = datasets.OPENWORLDCIFAR10(root='./orca/datasets', labeled=True, labeled_num=args.labeled_num,
                                                    labeled_ratio=args.labeled_ratio, download=True, transform=TransformTwice(datasets.dict_transform['cifar_train']))
        # 从上一步的训练集中单独取出无标签数据集供后面使用
        # 无标签数据中一部分是属于旧类的样本（前五类），剩下的大多数都是属于新类（后五类）
        train_unlabel_set = datasets.OPENWORLDCIFAR10(root='./orca/datasets', labeled=False, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio,
                                                      download=True, transform=TransformTwice(datasets.dict_transform['cifar_train']), unlabeled_idxs=train_label_set.unlabeled_idxs)
        # 测试机中只有未标记数据集，这更加符合常理
        # 测试集中的无标签数据包含旧类和新类，索引是从训练集中得到的
        test_set = datasets.OPENWORLDCIFAR10(root='./orca/datasets', labeled=False, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio,
                                             download=True, transform=datasets.dict_transform['cifar_test'], unlabeled_idxs=train_label_set.unlabeled_idxs)
        # 如果是cifar10，那么目标类别数量预先定义为10
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_label_set = datasets.OPENWORLDCIFAR100(root='./orca/datasets', labeled=True, labeled_num=args.labeled_num,
                                                     labeled_ratio=args.labeled_ratio, download=True, transform=TransformTwice(datasets.dict_transform['cifar_train']))
        train_unlabel_set = datasets.OPENWORLDCIFAR100(root='./orca/datasets', labeled=False, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio,
                                                       download=True, transform=TransformTwice(datasets.dict_transform['cifar_train']), unlabeled_idxs=train_label_set.unlabeled_idxs)
        test_set = datasets.OPENWORLDCIFAR100(root='./orca/datasets', labeled=False, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio,
                                              download=True, transform=datasets.dict_transform['cifar_test'], unlabeled_idxs=train_label_set.unlabeled_idxs)
        num_classes = 100
    # 使用了规定之外的数据集
    else:
        warnings.warn('Dataset is not listed')
        return

    labeled_len = len(train_label_set)
    unlabeled_len = len(train_unlabel_set)
    labeled_batch_size = int(
        args.batch_size * labeled_len / (labeled_len + unlabeled_len))

    # Initialize the splits
    # 一共有两个训练集，包括无标签数据和有标签数据
    # 在数据集下载之后就对数据集进行了处理，后期的刀片数据集的加载器就已经是处理过的
    # 数据集的处理文件在open_world_cifar
    # 训练时取出的数据包含新类和旧类，有标签和无标签
    train_label_loader = torch.utils.data.DataLoader(
        train_label_set, batch_size=labeled_batch_size, shuffle=True, num_workers=2, drop_last=True)
    # 训练时取出的数据包含新类和旧类，无标签
    train_unlabel_loader = torch.utils.data.DataLoader(
        train_unlabel_set, batch_size=args.batch_size - labeled_batch_size, shuffle=True, num_workers=2, drop_last=True)
    # 测试时取出的数据包含新类和旧类，无标签
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=100, shuffle=False, num_workers=1)

    # First network intialization: pretrain the RotNet network
    # 预先定义的分类头
    model = models.resnet18(num_classes=num_classes)
    model = model.to(device)
    # 尝试加载预训练的模型，预训练的模型使用simCLR训练
    # 但是pretrained模型没给，所以模型无法重新训练
    '''尝试去掉预训练的步骤'''
    # if args.dataset == 'cifar10':
    #     state_dict = torch.load('./pretrained/simclr_cifar_10.pth.tar')
    # elif args.dataset == 'cifar100':
    #     state_dict = torch.load('./pretrained/simclr_cifar_100.pth.tar')
    # model.load_state_dict(state_dict, strict=False)
    # model = model.to(device)

    # Freeze the earlier filters
    # 部分层不更新，相当于预训练之后只更新部分层
    '''这个认真看'''
    '''这个认真看'''
    '''这个认真看'''
    # 和论文中说的一样，冻结第一层，只更新最后一层和分类器
    for name, param in model.named_parameters():
        # 除了线性层和第四层更新，其余的都不更新
        # 不是linear和layer4的都不更新，不需要梯度
        if 'linear' not in name and 'layer4' not in name:
            param.requires_grad = False

    # Set the optimizer
    optimizer = optim.SGD(model.parameters(), lr=1e-1,
                          weight_decay=5e-4, momentum=0.9)
    # 更新学习率的策略函数
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.milestones, gamma=0.1)
    # tensorboard的东西
    tf_writer = SummaryWriter(log_dir=args.savedir)
    # 训练，每一个epoch都train一次，遍历所有的数据集
    for epoch in range(args.epochs):
        # 每一个epoch既进行测试又进行训练
        # 这样的目的是为了测试阶段得到的数据的不确定性，从而动态调整有标签数据的训练速度
        mean_uncert = test(args, model, args.labeled_num,
                           device, test_loader, epoch, tf_writer)
        train(args, model, device, train_label_loader,
              train_unlabel_loader, optimizer, mean_uncert, epoch, tf_writer)
        # 每一个epoch都需要动态调整学习率
        scheduler.step()


if __name__ == '__main__':
    main()
