#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)##计算每个用户应该拥有的样本数量
    dict_users, all_idxs = {}, [i for i in range(len(dataset))] ##dict_users:每个用户的样本索引；all_idxs:包含所有数据索引的列表
    for i in range(num_users):  ##给每个用户不重复地分配数据的索引
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300 ##num_shards：将数据划分为的片段数量；num_imgs:每个片段的图像数量
    idx_shard = [i for i in range(num_shards)]  ##每个片段的索引
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}  ##每个用户对应的样本索引
    idxs = np.arange(num_shards*num_imgs) ##创建一个包含所有图像索引的数组，就是一个0-的有序数组
    labels = dataset.train_labels.numpy() ##将数据集中样本的标签转换为numpy

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()] ##根据标签对图像进行排序 ，返回为有序数列的原位置的序号例如，如果标签的原始顺序是 [5, 0, 4, 1, 2]，使用 argsort() 后会返回 [1, 4, 3, 2, 0]，表示标签1排在最前，标签5排在最后。
    idxs = idxs_labels[0,:] ##idxs排序后的索引，相当于标签排序得到有序，对应索引跟随改变，得到新位置
       ##其将相同类别的图像集中在一起

    # divide and assign 将数据分配给每个用户
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))##随机选择两个片段且不重复
        idx_shard = list(set(idx_shard) - rand_set) ##移除已被选择的片段
        for rand in rand_set: ##将所选片段
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)  ##计算每个用户应该拥有的样本数量
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):  ##给每个用户不重复地分配数据的索引
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
