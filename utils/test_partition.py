#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import numpy as np
import torch
from torchvision import datasets
import copy
from torch.utils.data import Dataset, DataLoader, Subset

def generate_similar_test_distribution(B, es_label_distributions, test_data, num_samples_per_eh=100):
    """
    根据B矩阵和ES的标签分布情况，为每个EH生成具有相似分布的测试集
    
    Args:
        B (numpy.ndarray): ES-EH关联矩阵，形状为(num_ESs, num_EHs)
        es_label_distributions (numpy.ndarray): 每个ES的标签分布，形状为(num_ESs, num_classes)
        test_data: 完整的测试数据集
        num_samples_per_eh (int): 每个EH测试集中的样本数量
        
    Returns:
        dict: EH测试集字典，键为EH索引，值为测试数据索引列表
    """
    num_ESs, num_EHs = B.shape
    num_classes = es_label_distributions.shape[1]
    
    # 计算每个EH的标签分布（根据连接到它的ES的标签分布）
    eh_label_distributions = np.zeros((num_EHs, num_classes))
    for eh_idx in range(num_EHs):
        # 找到连接到当前EH的所有ES
        es_indices = np.where(B[:, eh_idx] == 1)[0]
        if len(es_indices) > 0:
            # 将这些ES的标签分布相加
            for es_idx in es_indices:
                eh_label_distributions[eh_idx] += es_label_distributions[es_idx]
            
            # 归一化以获得概率分布
            if np.sum(eh_label_distributions[eh_idx]) > 0:
                eh_label_distributions[eh_idx] = eh_label_distributions[eh_idx] / np.sum(eh_label_distributions[eh_idx])
    
    # 获取测试数据的标签
    if isinstance(test_data, tuple) and len(test_data) == 2:
        # 如果test_data是tuple(data, labels)格式
        test_labels = test_data[1]
    elif hasattr(test_data, 'targets'):
        # 如果test_data是torchvision数据集
        if isinstance(test_data.targets, list):
            test_labels = np.array(test_data.targets)
        else:
            test_labels = test_data.targets.numpy()
    else:
        # 尝试其他可能的格式
        try:
            test_labels = np.array([y for _, y in test_data])
        except:
            raise ValueError("无法从提供的测试数据中提取标签")
    
    # 为每个EH创建测试集
    eh_test_indices = {}
    
    for eh_idx in range(num_EHs):
        eh_test_indices[eh_idx] = []
        
        # 如果该EH没有连接的ES，跳过
        if np.sum(eh_label_distributions[eh_idx]) == 0:
            continue
        
        # 为每个类别分配样本数
        class_samples = np.round(eh_label_distributions[eh_idx] * num_samples_per_eh).astype(int)
        
        # 确保总数等于num_samples_per_eh
        diff = num_samples_per_eh - np.sum(class_samples)
        if diff > 0:
            # 如果总数不足，将差值添加到概率最高的类别
            class_samples[np.argmax(eh_label_distributions[eh_idx])] += diff
        elif diff < 0:
            # 如果总数过多，从最不重要的类别开始减少
            sorted_indices = np.argsort(eh_label_distributions[eh_idx])
            idx = 0
            while diff < 0:
                if class_samples[sorted_indices[idx]] > 0:
                    class_samples[sorted_indices[idx]] -= 1
                    diff += 1
                else:
                    idx += 1
                    if idx >= len(sorted_indices):
                        break
        
        # 为每个类别选择样本
        for class_idx in range(num_classes):
            if class_samples[class_idx] > 0:
                # 找到该类别的所有样本
                class_indices = np.where(test_labels == class_idx)[0]
                
                # 如果没有足够的样本，就重复使用
                if len(class_indices) < class_samples[class_idx]:
                    selected_indices = np.random.choice(class_indices, class_samples[class_idx], replace=True)
                else:
                    selected_indices = np.random.choice(class_indices, class_samples[class_idx], replace=False)
                
                eh_test_indices[eh_idx].extend(selected_indices)
        
        # 打乱索引
        np.random.shuffle(eh_test_indices[eh_idx])
    
    return eh_test_indices

class EHTestDataset(Dataset):
    """
    EH测试数据集封装类
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

def create_eh_test_datasets(B_random, B_cluster, es_label_distributions, dataset_test, num_samples_per_eh=1000):
    """
    创建基于随机B矩阵和聚类B矩阵的EH测试数据集
    
    Args:
        B_random (numpy.ndarray): 随机生成的ES-EH关联矩阵
        B_cluster (numpy.ndarray): 基于聚类生成的ES-EH关联矩阵
        es_label_distributions (numpy.ndarray): 每个ES的标签分布
        dataset_test: 完整的测试数据集
        num_samples_per_eh (int): 每个EH测试集的样本数
        
    Returns:
        tuple: (random_eh_test_datasets, cluster_eh_test_datasets)
            - random_eh_test_datasets: 基于随机B矩阵的测试数据集字典
            - cluster_eh_test_datasets: 基于聚类B矩阵的测试数据集字典
    """
    # 生成基于随机B矩阵的测试集索引
    random_eh_indices = generate_similar_test_distribution(
        B_random, es_label_distributions, dataset_test, num_samples_per_eh)
    
    # 生成基于聚类B矩阵的测试集索引
    cluster_eh_indices = generate_similar_test_distribution(
        B_cluster, es_label_distributions, dataset_test, num_samples_per_eh)
    
    # 创建测试数据集
    random_eh_test_datasets = {}
    cluster_eh_test_datasets = {}
    
    for eh_idx, indices in random_eh_indices.items():
        if len(indices) > 0:
            random_eh_test_datasets[eh_idx] = EHTestDataset(dataset_test, indices)
    
    for eh_idx, indices in cluster_eh_indices.items():
        if len(indices) > 0:
            cluster_eh_test_datasets[eh_idx] = EHTestDataset(dataset_test, indices)
    
    return random_eh_test_datasets, cluster_eh_test_datasets