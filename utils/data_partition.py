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
        # Non-IID标签方向性分布（基于Dirichlet分布）
        min_size = 0
        min_require_size = 10
        beta = noniid_param
        
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(num_clients)]
            for k in range(num_classes):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, num_clients))
                
                # 平衡处理
                proportions = np.array([p * (len(idx_j) < n_train / num_clients) 
                                      for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in 
                           zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
        
        for j in range(num_clients):
            np.random.shuffle(idx_batch[j])
            client_data_mapping[j] = idx_batch[j]
    
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
    data_stats = record_data_statistics(y_train, client_data_mapping)
    
    return X_train, y_train, X_test, y_test, client_data_mapping, data_stats


def record_data_statistics(y_train, client_data_mapping):
    """
    记录客户端数据统计信息
    
    Args:
        y_train: 训练标签
        client_data_mapping: 客户端数据映射
    
    Returns:
        dict: 数据统计信息
    """
    client_class_counts = {}
    
    for client_id, data_idxs in client_data_mapping.items():
        unq, unq_cnt = np.unique(y_train[data_idxs], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        client_class_counts[client_id] = tmp
    
    print('客户端数据分布统计:', client_class_counts)
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
        tuple: (训练数据集, 测试数据集, 客户端数据映射)
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
    
    return train_data, test_data, client_mapping


if __name__ == "__main__":
    # 测试示例
    dataset_type = "mnist"
    num_clients = 10
    data_path = "./data/mnist"
    partition_method = "noniid-labeldir"
    noniid_param = 0.1
    
    train_data, test_data, client_mapping = get_client_datasets(
        dataset_type, num_clients, data_path, partition_method, noniid_param
    )
    
    print(f"训练数据形状: {train_data['data'].shape}")
    print(f"测试数据形状: {test_data['data'].shape}")
    print(f"客户端数量: {len(client_mapping)}")
    
    # 打印每个客户端的数据数量
    for client_id, data_indices in client_mapping.items():
        print(f"客户端 {client_id}: {len(data_indices)} 个样本")