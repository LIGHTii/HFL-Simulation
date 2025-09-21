#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据划分模块 - 适配项目的non-iid数据分布方式
基于utils.py中的partition_data函数修改
支持多种数据集和分区方式
"""

import os
import numpy as np
import torch
import torchvision.transforms as transforms
import random
from torchvision import datasets
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def mkdirs(dirpath):
    """创建目录"""
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass


def load_dataset(dataset_type, data_path):
    """
    加载数据集
    
    Args:
        dataset_type (str): 数据集类型 ('mnist', 'cifar10', 'cifar100', etc.)
        data_path (str): 数据保存路径
    
    Returns:
        tuple: (X_train, y_train, X_test, y_test)
    """
    
    if dataset_type == 'mnist':
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.MNIST(data_path, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(data_path, train=False, download=True, transform=transform)
        
        X_train = train_dataset.data.numpy()
        y_train = train_dataset.targets.numpy()
        X_test = test_dataset.data.numpy()
        y_test = test_dataset.targets.numpy()
        
    elif dataset_type == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
        
        X_train = np.array(train_dataset.data)
        y_train = np.array(train_dataset.targets)
        X_test = np.array(test_dataset.data)
        y_test = np.array(test_dataset.targets)
        
    elif dataset_type == 'cifar100':
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.CIFAR100(data_path, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(data_path, train=False, download=True, transform=transform)
        
        X_train = np.array(train_dataset.data)
        y_train = np.array(train_dataset.targets)
        X_test = np.array(test_dataset.data)
        y_test = np.array(test_dataset.targets)
        
    else:
        raise ValueError(f"不支持的数据集类型: {dataset_type}")
    
    return X_train, y_train, X_test, y_test


def partition_data_custom(dataset_type, num_clients, data_path, partition_method, noniid_param=0.4):
    """
    自定义数据划分函数
    
    Args:
        dataset_type (str): 数据集类型 ('mnist', 'cifar10', 'cifar100')
        num_clients (int): 客户端数量
        data_path (str): 数据保存路径
        partition_method (str): 数据分区方式
            - 'homo': 同质分布（IID）
            - 'noniid-labeldir': non-IID标签方向性分布
            - 'noniid-#label1' to 'noniid-#label9': 每个客户端包含特定数量标签
            - 'iid-diff-quantity': IID但数据量不同
        noniid_param (float): non-iid分布参数（如Dirichlet参数beta）
    
    Returns:
        tuple: (X_train, y_train, X_test, y_test, client_data_mapping, data_stats)
    """
    
    # 加载数据集
    X_train, y_train, X_test, y_test = load_dataset(dataset_type, data_path)
    
    n_train = y_train.shape[0]
    
    # 根据数据集确定类别数
    if dataset_type == 'mnist':
        num_classes = 10
    elif dataset_type == 'cifar10':
        num_classes = 10
    elif dataset_type == 'cifar100':
        num_classes = 100
    else:
        num_classes = len(np.unique(y_train))
    
    # 初始化客户端数据映射
    client_data_mapping = {}
    
    if partition_method == "homo":
        # 同质分布（IID）
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, num_clients)
        client_data_mapping = {i: batch_idxs[i] for i in range(num_clients)}

    elif partition_method == "noniid-labeldir":
        # Non-IID标签方向性分布（基于Dirichlet分布）- 每个客户端数据量相等，标签分布不均
        samples_per_client = 600# n_train // num_clients
        beta = noniid_param

        # 初始化每个客户端的数据索引列表
        idx_batch = [[] for _ in range(num_clients)]

        # 为每个类别生成Dirichlet分布的比例
        for k in range(num_classes):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)

            # 生成Dirichlet分布的客户端比例
            proportions = np.random.dirichlet(np.repeat(beta, num_clients))

            # 根据比例分配每个类别的样本到不同客户端
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            class_splits = np.split(idx_k, proportions)

            for i, split in enumerate(class_splits):
                idx_batch[i].extend(split.tolist())

        # 确保每个客户端的数据量完全相等
        for j in range(num_clients):
            np.random.shuffle(idx_batch[j])
            current_size = len(idx_batch[j])

            if current_size > samples_per_client:
                # 如果数据过多，随机移除多余数据
                excess_data = idx_batch[j][samples_per_client:]
                idx_batch[j] = idx_batch[j][:samples_per_client]

                # 将多余数据重新分配给数据不足的客户端
                for k in range(num_clients):
                    if len(idx_batch[k]) < samples_per_client and len(excess_data) > 0:
                        needed = samples_per_client - len(idx_batch[k])
                        take = min(needed, len(excess_data))
                        idx_batch[k].extend(excess_data[:take])
                        excess_data = excess_data[take:]

            elif current_size < samples_per_client:
                # 如果数据不足，从其他客户端借用数据
                needed = samples_per_client - current_size
                for k in range(num_clients):
                    if len(idx_batch[k]) > samples_per_client and needed > 0:
                        excess = len(idx_batch[k]) - samples_per_client
                        take = min(needed, excess)
                        idx_batch[j].extend(idx_batch[k][-take:])
                        idx_batch[k] = idx_batch[k][:-take]
                        needed -= take

        # 最终确保所有客户端数据量完全相等
        total_assigned = sum(len(batch) for batch in idx_batch)
        remaining_data = list(set(range(n_train)) - set(idx for batch in idx_batch for idx in batch))

        # 分配剩余数据
        for i, remaining_idx in enumerate(remaining_data):
            client_idx = i % num_clients
            if len(idx_batch[client_idx]) < samples_per_client:
                idx_batch[client_idx].append(remaining_idx)

        # 最终平衡：确保每个客户端恰好有samples_per_client个样本
        for j in range(num_clients):
            current_size = len(idx_batch[j])
            if current_size != samples_per_client:
                if current_size > samples_per_client:
                    # 移除多余样本
                    idx_batch[j] = idx_batch[j][:samples_per_client]
                else:
                    # 这种情况在正确实现下不应该发生
                    print(f"警告：客户端 {j} 数据量不足: {current_size}/{samples_per_client}")

        # 最终打乱每个客户端的数据并转换为numpy数组
        for j in range(num_clients):
            np.random.shuffle(idx_batch[j])
            client_data_mapping[j] = np.array(idx_batch[j])
    
    elif partition_method.startswith("noniid-#label") and len(partition_method) > 13:
        # 每个客户端包含指定数量的标签
        try:
            num_labels = int(partition_method[13:])
        except ValueError:
            raise ValueError(f"无效的分区方法: {partition_method}")
            
        if num_labels >= num_classes:
            num_labels = num_classes
            
        # 为每个客户端分配标签
        times = [0 for _ in range(num_classes)]
        contain = []
        
        for i in range(num_clients):
            current = [i % num_classes]
            times[i % num_classes] += 1
            j = 1
            while j < num_labels:
                ind = random.randint(0, num_classes - 1)
                if ind not in current:
                    j += 1
                    current.append(ind)
                    times[ind] += 1
            contain.append(current)
        
        client_data_mapping = {i: np.ndarray(0, dtype=np.int64) for i in range(num_clients)}
        
        for i in range(num_classes):
            idx_k = np.where(y_train == i)[0]
            np.random.shuffle(idx_k)
            split = np.array_split(idx_k, times[i]) if times[i] > 0 else []
            ids = 0
            for j in range(num_clients):
                if i in contain[j]:
                    client_data_mapping[j] = np.append(client_data_mapping[j], split[ids])
                    ids += 1
    
    elif partition_method == "iid-diff-quantity":
        # IID但数据量不同
        idxs = np.random.permutation(n_train)
        min_size = 0
        beta = noniid_param
        
        while min_size < 10:
            proportions = np.random.dirichlet(np.repeat(beta, num_clients))
            proportions = proportions / proportions.sum()
            min_size = np.min(proportions * len(idxs))
        
        proportions = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
        batch_idxs = np.split(idxs, proportions)
        client_data_mapping = {i: batch_idxs[i] for i in range(num_clients)}
    
    else:
        raise ValueError(f"不支持的分区方法: {partition_method}")
    
    # 记录数据统计
    data_stats = record_data_statistics(y_train, client_data_mapping, num_classes)
    
    return X_train, y_train, X_test, y_test, client_data_mapping, data_stats


def record_data_statistics(y_train, client_data_mapping, num_classes):
    """
    记录客户端数据统计信息

    Args:
        y_train: 训练标签
        client_data_mapping: 客户端数据映射
        num_classes: 类别数量

    Returns:
        dict: 数据统计信息
    """
    client_class_counts = {}
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    print("\n各客户端数据分布统计:")
    print("客户端ID\t总样本数\t" + "\t".join([f"类别{i}" for i in range(num_classes)]))

    for client_id, data_idxs in client_data_mapping.items():
        unq, unq_cnt = np.unique(y_train[data_idxs], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        client_class_counts[client_id] = tmp

        # 打印每个客户端的数据分布
        total_samples = len(data_idxs)
        class_counts = [tmp.get(i, 0) for i in range(num_classes)]
        # print(f"{client_id}\t\t{total_samples}\t\t" + "\t".join([str(count) for count in class_counts]))

    return client_class_counts


def get_client_datasets(dataset_type, num_clients, data_path, partition_method, noniid_param=0.4):
    """
    获取客户端数据集的便捷函数
    
    Args:
        dataset_type (str): 数据集类型
        num_clients (int): 客户端数量  
        data_path (str): 数据保存路径
        partition_method (str): 数据分区方式
        noniid_param (float): non-iid分布参数
    
    Returns:
        tuple: (训练数据集, 测试数据集, 客户端数据映射, 客户端类别映射)
    """
    
    # 确保数据路径存在
    mkdirs(data_path)
    
    # 划分数据
    X_train, y_train, X_test, y_test, client_mapping, data_stats = partition_data_custom(
        dataset_type, num_clients, data_path, partition_method, noniid_param
    )
    
    # 构建数据集对象（这里返回numpy数组，可以根据需要转换为torch tensor）
    train_data = {'data': X_train, 'labels': y_train}
    test_data = {'data': X_test, 'labels': y_test}
    
    # 生成客户端类别映射
    client_classes = {}
    for client_id, data_indices in client_mapping.items():
        # 获取该客户端的所有标签
        client_labels = y_train[data_indices]
        # 获取唯一的类别
        unique_classes = np.unique(client_labels).tolist()
        client_classes[client_id] = unique_classes
    
    return train_data, test_data, client_mapping, client_classes


if __name__ == "__main__":
    # 测试示例
    dataset_type = "mnist"
    num_clients = 10
    data_path = "./data/mnist"
    partition_method = "noniid-labeldir"
    noniid_param = 0.1
    
    train_data, test_data, client_mapping, client_classes = get_client_datasets(
        dataset_type, num_clients, data_path, partition_method, noniid_param
    )
    
    print(f"训练数据形状: {train_data['data'].shape}")
    print(f"测试数据形状: {test_data['data'].shape}")
    print(f"客户端数量: {len(client_mapping)}")
    
    # 打印每个客户端的数据数量和类别信息
    for client_id in range(min(5, num_clients)):  # 只打印前5个客户端
        data_indices = client_mapping[client_id]
        classes = client_classes[client_id]
        print(f"客户端 {client_id}: {len(data_indices)} 个样本, 类别: {classes}")