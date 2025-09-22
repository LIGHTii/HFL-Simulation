#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt

class EHTestsetGenerator:
    """生成EH专属测试集的工具类"""
    
    @staticmethod
    def get_client_label_distribution(dataset, dict_users, client_indices):
        """
        获取一组客户端的标签分布情况
        
        Args:
            dataset: 训练数据集
            dict_users: 客户端数据索引字典
            client_indices: 要计算的客户端索引列表
            
        Returns:
            label_distribution: 标签分布数组，形状为 (num_classes,)
        """
        # 初始化标签分布计数
        if hasattr(dataset, 'targets'):
            num_classes = len(set(dataset.targets))
        else:
            # 对于MNIST数据集
            num_classes = len(set(dataset.train_labels.numpy()))
        
        label_distribution = np.zeros(num_classes)
        
        # 遍历所有指定客户端的数据
        for client_idx in client_indices:
            if client_idx in dict_users:
                client_data_idxs = dict_users[client_idx]
                for idx in client_data_idxs:
                    # 根据不同数据集获取标签
                    if hasattr(dataset, 'targets'):
                        label = dataset.targets[idx]
                        if isinstance(label, torch.Tensor):
                            label = label.item()
                    else:
                        # 对于MNIST数据集
                        label = dataset.train_labels[idx].item()
                    
                    label_distribution[label] += 1
        
        # 归一化分布
        if label_distribution.sum() > 0:
            label_distribution = label_distribution / label_distribution.sum()
            
        return label_distribution
        
    @staticmethod
    def create_eh_testsets(dataset_test, A, B, C1, C2, dataset_train=None, dict_users=None, visualize=False):
        """
        为每个EH创建专属测试集，数据分布与其下游客户端的训练数据分布类似
        
        Args:
            dataset_test: 全局测试数据集
            A: 客户端-ES关联矩阵
            B: ES-EH关联矩阵
            C1: ES->客户端映射
            C2: EH->ES映射
            dataset_train: 训练数据集（用于计算标签分布）
            dict_users: 客户端数据索引字典
            visualize: 是否可视化EH的标签分布和对应测试集
            
        Returns:
            eh_testsets: 每个EH的专属测试集，字典 {eh_idx: testset_indices}
            eh_label_distributions: 每个EH的标签分布
        """
        num_ESs = B.shape[0]
        num_EHs = B.shape[1]
        
        # 获取测试集中每个样本的标签
        if hasattr(dataset_test, 'targets'):
            test_labels = np.array(dataset_test.targets)
            if isinstance(test_labels, torch.Tensor):
                test_labels = test_labels.numpy()
        else:
            # 对于MNIST数据集
            test_labels = dataset_test.test_labels.numpy()
        
        num_classes = len(set(test_labels))
        
        # 计算每个EH下客户端的标签分布
        eh_label_distributions = {}
        
        # 每个EH对应的客户端列表
        eh_clients = {}
        
        # 为每个EH找到所有下游客户端
        for eh_idx in range(num_EHs):
            es_indices = C2[eh_idx]
            clients = []
            for es_idx in es_indices:
                clients.extend(C1[es_idx])
            eh_clients[eh_idx] = clients
        
        # 计算每个EH的标签分布
        for eh_idx, client_indices in eh_clients.items():
            eh_label_distributions[eh_idx] = EHTestsetGenerator.get_client_label_distribution(
                dataset_train, dict_users, client_indices
            )
        
        # 创建每个EH的专属测试集
        eh_testsets = {}
        
        # 按类别索引测试数据
        test_indices_by_class = {}
        for c in range(num_classes):
            test_indices_by_class[c] = np.where(test_labels == c)[0]
            np.random.shuffle(test_indices_by_class[c])
        
        # 按照标签分布比例为每个EH采样测试数据
        test_size_per_eh = len(dataset_test) // num_EHs  # 每个EH获得的测试集大小
        
        for eh_idx, label_dist in eh_label_distributions.items():
            testset_indices = []
            
            # 确保分布中没有零值（为了防止数值错误）
            epsilon = 1e-10
            smoothed_dist = label_dist + epsilon
            smoothed_dist = smoothed_dist / smoothed_dist.sum()
            
            # 按类别采样
            for c in range(num_classes):
                # 根据标签分布计算该类别应采样的样本数
                num_samples = int(test_size_per_eh * smoothed_dist[c])
                
                # 确保不超过该类别的可用样本数
                available = len(test_indices_by_class[c])
                num_samples = min(num_samples, available)
                
                # 从该类别中采样
                if num_samples > 0:
                    selected_indices = test_indices_by_class[c][:num_samples]
                    test_indices_by_class[c] = test_indices_by_class[c][num_samples:]  # 更新剩余样本
                    testset_indices.extend(selected_indices)
            
            # 为了确保每个EH有足够多的测试样本，如果不足，从剩余样本中随机补充
            remaining_needed = test_size_per_eh - len(testset_indices)
            if remaining_needed > 0:
                remaining_indices = []
                for c in range(num_classes):
                    remaining_indices.extend(test_indices_by_class[c])
                
                # 如果还有足够的剩余样本，则随机补充
                if len(remaining_indices) >= remaining_needed:
                    np.random.shuffle(remaining_indices)
                    testset_indices.extend(remaining_indices[:remaining_needed])
                else:
                    # 如果剩余样本不足，则使用所有剩余样本
                    testset_indices.extend(remaining_indices)
            
            eh_testsets[eh_idx] = np.array(testset_indices)
        
        # 可视化EH的标签分布和对应的测试集分布
        if visualize:
            EHTestsetGenerator.visualize_eh_distributions(eh_label_distributions, eh_testsets, test_labels, num_classes)
        
        return eh_testsets, eh_label_distributions
    
    @staticmethod
    def visualize_eh_distributions(eh_label_distributions, eh_testsets, test_labels, num_classes):
        """
        可视化EH的标签分布和对应测试集的分布
        
        Args:
            eh_label_distributions: 每个EH的标签分布
            eh_testsets: 每个EH的测试集索引
            test_labels: 测试集的标签
            num_classes: 类别数
        """
        num_ehs = len(eh_label_distributions)
        
        try:
            # 计算测试集的分布
            eh_test_distributions = {}
            for eh_idx, test_indices in eh_testsets.items():
                test_dist = np.zeros(num_classes)
                for idx in test_indices:
                    label = test_labels[idx]
                    if isinstance(label, np.ndarray):
                        label = label.item()  # 将numpy数组转换为标量
                    test_dist[label] += 1
                
                # 归一化
                if test_dist.sum() > 0:
                    test_dist = test_dist / test_dist.sum()
                
                eh_test_distributions[eh_idx] = test_dist
            
            # 检查数据维度
            for eh_idx in sorted(eh_label_distributions.keys()):
                if len(eh_label_distributions[eh_idx]) != num_classes:
                    print(f"警告：EH {eh_idx} 标签分布维度 {len(eh_label_distributions[eh_idx])} 与类别数 {num_classes} 不匹配")
                    # 修复维度问题，填充为零向量
                    corrected_dist = np.zeros(num_classes)
                    for i in range(min(len(eh_label_distributions[eh_idx]), num_classes)):
                        corrected_dist[i] = eh_label_distributions[eh_idx][i]
                    eh_label_distributions[eh_idx] = corrected_dist
            
            # 绘制对比图
            fig, axes = plt.subplots(num_ehs, 2, figsize=(12, 3 * num_ehs))
            
            # 处理只有一个EH的情况
            if num_ehs == 1:
                axes = np.array([axes])
            
            for i, eh_idx in enumerate(sorted(eh_label_distributions.keys())):
                # 确保索引不越界
                if i >= len(axes):
                    print(f"警告：图表索引 {i} 超出范围，已跳过 EH {eh_idx}")
                    continue
                
                # 训练数据分布
                ax1 = axes[i, 0]
                x_range = np.arange(num_classes)  # 明确创建x轴数值范围
                ax1.bar(x_range, eh_label_distributions[eh_idx])
                ax1.set_title(f'EH {eh_idx} 训练数据分布')
                ax1.set_xlabel('类别')
                ax1.set_ylabel('比例')
                ax1.set_xticks(range(num_classes))
                
                # 测试数据分布
                ax2 = axes[i, 1]
                ax2.bar(x_range, eh_test_distributions[eh_idx])
                ax2.set_title(f'EH {eh_idx} 测试数据分布')
                ax2.set_xlabel('类别')
                ax2.set_ylabel('比例')
                ax2.set_xticks(range(num_classes))
            
            plt.tight_layout()
            plt.savefig('./save/eh_distributions.png')
            plt.close()
            print("EH分布对比图已保存到 ./save/eh_distributions.png")
        except Exception as e:
            print(f"可视化EH分布时出错: {e}")
            print("跳过可视化，继续执行程序")

class EHSubset(Dataset):
    """表示EH专属测试集的数据集类"""
    
    def __init__(self, dataset, indices):
        """
        创建一个基于原始数据集和索引的子集
        
        Args:
            dataset: 原始数据集
            indices: 要包含的数据索引
        """
        self.dataset = dataset
        self.indices = indices
    
    def __getitem__(self, idx):
        """获取指定索引的样本"""
        return self.dataset[self.indices[idx]]
    
    def __len__(self):
        """返回数据集的大小"""
        return len(self.indices)

def test_eh_model(net, dataset_test, eh_test_indices, args):
    """
    在EH专属测试集上测试模型
    
    Args:
        net: 要测试的模型
        dataset_test: 测试数据集
        eh_test_indices: EH专属测试集的索引
        args: 参数对象
        
    Returns:
        accuracy: 准确率
        test_loss: 测试损失
    """
    from torch import nn
    import torch.nn.functional as F
    
    net.eval()
    
    # 创建EH专属测试集的子集
    eh_testset = EHSubset(dataset_test, eh_test_indices)
    
    # 创建数据加载器
    test_loader = DataLoader(eh_testset, batch_size=args.bs)
    
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(args.device), target.to(args.device)
            
            # 前向传播
            output = net(data)
            
            # 计算损失
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            
            # 计算准确率
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    # 计算平均损失和准确率
    test_loss /= len(eh_testset)
    accuracy = 100. * correct / len(eh_testset)
    
    return accuracy, test_loss