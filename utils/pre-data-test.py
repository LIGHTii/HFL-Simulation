#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
import torch


def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)  ##计算每个用户应该拥有的样本数量
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]  ##dict_users:每个用户的样本索引；all_idxs:包含所有数据索引的列表
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
    # 60,000 training images, 200 shards, 300 images per shard
    num_shards, num_imgs = 200, 300  ##num_shards：将数据划分为的片段数量；num_imgs:每个片段的图像数量
    idx_shard = [i for i in range(num_shards)]  ##每个片段的索引
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}  ##每个用户对应的样本索引
    idxs = np.arange(num_shards * num_imgs)  ##创建一个包含所有图像索引的数组，就是一个0-59999的有序数组

    # 使用 .targets 属性, .train_labels 已被弃用
    try:
        labels = dataset.targets.numpy()
    except:
        labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    # 根据标签对图像进行排序 ，返回为有序数列的原位置的序号
    # 例如，如果标签的原始顺序是 [5, 0, 4, 1, 2]，使用 argsort() 后会返回 [1, 4, 3, 2, 0]
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]  ##idxs排序后的索引，相当于标签排序得到有序，对应索引跟随改变，得到新位置
    ##其将相同类别的图像集中在一起

    # divide and assign
    # 每个客户端分配2个shards, 2 * 300 = 600个样本
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))  ##随机选择两个片段且不重复
        idx_shard = list(set(idx_shard) - rand_set)  ##移除已被选择的片段
        for rand in rand_set:  ##将所选片段
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)  ##计算每个用户应该拥有的样本数量
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):  ##给每个用户不重复地分配数据的索引
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


# ===================================================================================
# ============================== 新增的分析函数 ========================================
# ===================================================================================
def analyze_distribution(dataset, dict_users):
    """
    分析并打印每个客户端的数据分布情况
    :param dataset: 数据集对象
    :param dict_users: 客户端索引字典 {client_id: [image_indices]}
    """
    try:
        labels = dataset.targets.numpy()
    except:
        labels = dataset.train_labels.numpy()

    # 获取所有类别
    num_classes = len(np.unique(labels))

    print("开始分析每个客户端的数据分布...\n" + "-" * 50)

    # 遍历每个客户端
    for i in range(len(dict_users)):
        user_indices = list(dict_users[i])
        user_labels = labels[user_indices]

        # 计算总样本数
        total_samples = len(user_indices)

        print(f"客户端 {i}:")
        print(f"  - 总样本数: {total_samples}")

        # 计算每个标签下的样本数量
        label_counts = {j: 0 for j in range(num_classes)}
        # 使用np.unique高效地统计
        unique, counts = np.unique(user_labels, return_counts=True)
        label_distribution = dict(zip(unique, counts))

        # 更新完整标签计数字典
        for label, count in label_distribution.items():
            label_counts[label] = count

        print(f"  - 各标签样本数: {label_counts}\n")


# ===================================================================================
# ============================== 修改后的主程序入口 ====================================
# ===================================================================================
if __name__ == '__main__':
    # 加载数据集
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))

    # 设置一个较小的客户端数量以便观察
    num_users = 10

    # ---------------- IID 划分并分析 ----------------
    print("=" * 20, " IID 分布分析 ", "=" * 20)
    dict_users_iid = mnist_iid(dataset_train, num_users)
    analyze_distribution(dataset_train, dict_users_iid)

    # ---------------- Non-IID 划分并分析 ----------------
    print("\n" + "=" * 20, " Non-IID 分布分析 ", "=" * 20)
    # 注意: mnist_noniid 函数的设计是固定的(200个shards, 300个imgs/shard),
    # 这意味着它最适合处理60000个样本，且每个用户固定分到600个样本。
    dict_users_noniid = mnist_noniid(dataset_train, num_users)
    analyze_distribution(dataset_train, dict_users_noniid)