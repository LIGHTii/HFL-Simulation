#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics


def restricted_softmax(logits, user_classes, args):
    """
    FedRS 核心算法: 受限制的 softmax 函数
    Args:
        logits: 模型的原始输出
        user_classes: 客户端拥有的类别列表
        args: 包含 fedrs_alpha 参数的配置
    Returns:
        受限制后的 logits
    """
    m_logits = torch.ones_like(logits[0]).to(args.device) * args.fedrs_alpha
    
    # 对客户端拥有的类别不做限制 (设为 1.0)
    for c in user_classes:
        m_logits[c] = 1.0
    
    # 对所有 logits 应用掩码
    for i in range(len(logits)):
        logits[i] = torch.mul(logits[i], m_logits)
    
    return logits


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, user_classes=None, use_cluster_ep=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True) # 数据加载器，
        self.user_classes = user_classes  # 客户端拥有的类别信息
        self.use_cluster_ep = use_cluster_ep  # 是否使用聚类专用的local_ep

    def train(self, net):
        net.train()
        
        # 根据使用场景选择不同的local_ep
        if self.use_cluster_ep and hasattr(self.args, 'cluster_local_ep') and self.args.cluster_local_ep is not None:
            local_ep = self.args.cluster_local_ep
            print(f"  [聚类训练] 使用cluster_local_ep={local_ep}")
        elif self.args.method == 'fedrs' and hasattr(self.args, 'min_le') and hasattr(self.args, 'max_le'):
            local_ep = random.randint(self.args.min_le, self.args.max_le)
        else:
            local_ep = self.args.local_ep
        
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum) # 定义了优化器，采用随机梯度下降SGD

        epoch_loss = []
        for iter in range(local_ep):
            batch_loss = [] # 记录每个批次地损失值
            for batch_idx, (images, labels) in enumerate(self.ldr_train): # 遍历客户端数据集，按批次返回图像和对应标签
                images, labels = images.to(self.args.device), labels.to(self.args.device) # 将图像和标签转移到gpu或者cpu
                net.zero_grad() # 清除上一次的梯度，以免梯度叠加
                log_probs = net(images) # 使用模型向前，计算输出 即预测值
                
                # FedRS 算法: 应用受限制的 softmax
                if self.args.method == 'fedrs' and self.user_classes is not None:
                    log_probs = restricted_softmax(log_probs, self.user_classes, self.args)
                
                loss = self.loss_func(log_probs, labels) # 计算损失
                loss.backward() # 反向传播，计算梯度
                optimizer.step() # 使用优化器更新模型参数，完成一次梯度更新
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item()) # 将本次损失值加入列表
            epoch_loss.append(sum(batch_loss)/len(batch_loss)) # 将当前轮次的平均值加入列表
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss) # 返回训练后的模型参数和该客户端的平均损失值

