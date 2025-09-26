#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import numpy as np
import torch
from torch.utils.da        for eh_idx in range(num_EHs):
            print(f"\nGenerating personalized test set for EH {eh_idx}...")
            testset_indices = []
            
            # 处理零分布情况：如果某个EH没有任何训练数据，使用均匀分布
            if np.sum(label_dist) == 0:
                print(f"  Warning: EH {eh_idx} has no downstream clients, using uniform distribution")
                uniform_dist = np.ones(num_classes) / num_classes
                smoothed_dist = uniform_dist
            else:
                # 确保分布中没有零值（为了防止数值错误），但保持原有分布特性
                epsilon = 1e-6  # 减小epsilon值，减少对原分布的影响
                smoothed_dist = label_dist + epsilon
                smoothed_dist = smoothed_dist / smoothed_dist.sum()
            
            print(f"  EH {eh_idx} label distribution: {smoothed_dist}")t, DataLoader, Subset
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
        # 自动获取正确类别数
        if hasattr(dataset, 'targets'):
            # 转换为numpy数组便于计算
            targets = dataset.targets.numpy() if isinstance(dataset.targets, torch.Tensor) else dataset.targets
            num_classes = int(np.max(targets)) + 1  # 最大标签值+1（如MNIST最大是9，9+1=10）
        else:
            targets = dataset.train_labels.numpy()
            num_classes = int(np.max(targets)) + 1
        
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
        
        # 按类别索引测试数据（允许重复使用）
        test_indices_by_class = {}
        for c in range(num_classes):
            test_indices_by_class[c] = np.where(test_labels == c)[0]
            np.random.shuffle(test_indices_by_class[c])
        
        # 计算每个EH的目标测试集大小
        # 使用固定大小确保所有EH都有足够的测试样本
        min_test_size_per_eh = max(500, len(dataset_test) // (num_EHs * 2))  # 至少500个样本，或者平均值的一半
        print(f"Target test set size per EH: {min_test_size_per_eh}")
        
        for eh_idx, label_dist in eh_label_distributions.items():
            print(f"\n为EH {eh_idx} 生成个性化测试集...")
            testset_indices = []
            
            # 处理零分布情况：如果某个EH没有任何训练数据，使用均匀分布
            if np.sum(label_dist) == 0:
                print(f"  警告: EH {eh_idx} 没有下游客户端，使用均匀分布")
                uniform_dist = np.ones(num_classes) / num_classes
                smoothed_dist = uniform_dist
            else:
                # 确保分布中没有零值（为了防止数值错误），但保持原有分布特性
                epsilon = 1e-6  # 减小epsilon值，减少对原分布的影响
                smoothed_dist = label_dist + epsilon
                smoothed_dist = smoothed_dist / smoothed_dist.sum()
            
            print(f"  EH {eh_idx} 标签分布: {smoothed_dist}")
            
            # 按类别采样（允许重复使用测试样本）
            actual_samples_per_class = []
            for c in range(num_classes):
                # 根据标签分布计算该类别应采样的样本数
                target_samples = int(min_test_size_per_eh * smoothed_dist[c])
                
                # 获取该类别的所有可用样本
                available_indices = test_indices_by_class[c]
                
                if len(available_indices) == 0:
                    print(f"    Class {c}: No available samples")
                    actual_samples_per_class.append(0)
                    continue
                
                # 如果需要的样本数超过可用样本数，则进行重复采样
                if target_samples <= len(available_indices):
                    # 直接采样，不重复
                    selected_indices = np.random.choice(available_indices, size=target_samples, replace=False)
                else:
                    # 需要重复采样
                    selected_indices = np.random.choice(available_indices, size=target_samples, replace=True)
                    print(f"    Class {c}: Need {target_samples} samples, but only {len(available_indices)} available, performing repeated sampling")
                
                testset_indices.extend(selected_indices)
                actual_samples_per_class.append(len(selected_indices))
            
            print(f"  Actual samples per class: {actual_samples_per_class}")
            print(f"  EH {eh_idx} total test set size: {len(testset_indices)}")
            
            # 如果总样本数仍然不足，进行全局补充采样
            if len(testset_indices) < min_test_size_per_eh:
                remaining_needed = min_test_size_per_eh - len(testset_indices)
                print(f"  Need to supplement {remaining_needed} samples")
                
                # 从所有测试样本中按分布比例补充
                all_test_indices = list(range(len(dataset_test)))
                
                # 按当前分布补充采样
                additional_indices = []
                for c in range(num_classes):
                    additional_needed = int(remaining_needed * smoothed_dist[c])
                    if additional_needed > 0:
                        class_indices = test_indices_by_class[c]
                        if len(class_indices) > 0:
                            additional_selected = np.random.choice(
                                class_indices, 
                                size=min(additional_needed, len(class_indices)), 
                                replace=True
                            )
                            additional_indices.extend(additional_selected)
                
                testset_indices.extend(additional_indices)
                print(f"  Total test set size after supplementation: {len(testset_indices)}")
            
            eh_testsets[eh_idx] = np.array(testset_indices)
        
        # 验证和统计生成的测试集
        print(f"\n=== Test Set Generation Result Validation ===")
        total_unique_samples = set()
        total_samples_used = 0
        
        for eh_idx, testset in eh_testsets.items():
            unique_samples = set(testset)
            total_unique_samples.update(unique_samples)
            total_samples_used += len(testset)
            
            print(f"EH {eh_idx}: test set size={len(testset)}, unique samples={len(unique_samples)}")
        
        coverage_rate = len(total_unique_samples) / len(dataset_test)
        avg_reuse_rate = total_samples_used / len(total_unique_samples) if len(total_unique_samples) > 0 else 0
        
        print(f"\nOverall Statistics:")
        print(f"  Original test set size: {len(dataset_test)}")
        print(f"  Unique samples used: {len(total_unique_samples)}")
        print(f"  Test set coverage rate: {coverage_rate:.1%}")
        print(f"  Average sample reuse rate: {avg_reuse_rate:.2f}x")
        print(f"  Total samples used: {total_samples_used}")
        print("=" * 35)
        
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
            print("\n=== EH Test Set Distribution Statistics ===")
            for eh_idx, test_indices in eh_testsets.items():
                test_dist = np.zeros(num_classes)
                unique_indices = len(np.unique(test_indices))  # 统计唯一样本数
                total_samples = len(test_indices)  # 统计总样本数（包含重复）
                
                for idx in test_indices:
                    label = test_labels[idx]
                    if isinstance(label, np.ndarray):
                        label = label.item()  # 将numpy数组转换为标量
                    test_dist[label] += 1
                
                # 归一化
                if test_dist.sum() > 0:
                    test_dist = test_dist / test_dist.sum()
                
                eh_test_distributions[eh_idx] = test_dist
                
                # 打印统计信息
                print(f"EH {eh_idx}: total samples={total_samples}, unique samples={unique_indices}, duplicate rate={1-unique_indices/total_samples:.2%}")
                print(f"  Test set distribution: {test_dist}")
                
                # 计算与期望分布的差异
                expected_dist = eh_label_distributions[eh_idx]
                if np.sum(expected_dist) > 0:
                    expected_dist = expected_dist / np.sum(expected_dist)
                    kl_divergence = np.sum(test_dist * np.log((test_dist + 1e-10) / (expected_dist + 1e-10)))
                    print(f"  KL divergence from expected distribution: {kl_divergence:.4f}")
            print("=" * 30)
            
            # 检查数据维度
            for eh_idx in sorted(eh_label_distributions.keys()):
                if len(eh_label_distributions[eh_idx]) != num_classes:
                    print(
                        f"Warning: EH {eh_idx} label distribution dimension {len(eh_label_distributions[eh_idx])} does not match number of classes {num_classes}")
                    # 修复维度问题，填充为零向量
                    corrected_dist = np.zeros(num_classes)
                    for i in range(min(len(eh_label_distributions[eh_idx]), num_classes)):
                        corrected_dist[i] = eh_label_distributions[eh_idx][i]
                    eh_label_distributions[eh_idx] = corrected_dist
            
            # 绘制对比图 - 增加一列显示分布差异
            fig, axes = plt.subplots(num_ehs, 3, figsize=(18, 3 * num_ehs))
            
            # 处理只有一个EH的情况
            if num_ehs == 1:
                axes = np.array([axes])
            
            for i, eh_idx in enumerate(sorted(eh_label_distributions.keys())):
                # 确保索引不越界
                if i >= len(axes):
                    print(f"Warning: Chart index {i} out of range, skipped EH {eh_idx}")
                    continue
                
                x_range = np.arange(num_classes)  # 明确创建x轴数值范围
                
                # 训练数据分布（期望分布）
                ax1 = axes[i, 0]
                expected_dist = eh_label_distributions[eh_idx]
                if np.sum(expected_dist) > 0:
                    expected_dist = expected_dist / np.sum(expected_dist)
                ax1.bar(x_range, expected_dist, alpha=0.7, color='skyblue')
                ax1.set_title(f'EH {eh_idx} Expected Distribution\n(Based on Downstream Clients)')
                ax1.set_xlabel('Class')
                ax1.set_ylabel('Proportion')
                ax1.set_xticks(range(num_classes))
                ax1.set_ylim(0, max(0.1, np.max(expected_dist) * 1.1))
                
                # 测试数据分布（实际分布）
                ax2 = axes[i, 1]
                actual_dist = eh_test_distributions[eh_idx]
                ax2.bar(x_range, actual_dist, alpha=0.7, color='lightcoral')
                ax2.set_title(f'EH {eh_idx} Actual Test Distribution\n(Samples: {len(eh_testsets[eh_idx])})')
                ax2.set_xlabel('Class')
                ax2.set_ylabel('Proportion')
                ax2.set_xticks(range(num_classes))
                ax2.set_ylim(0, max(0.1, np.max(actual_dist) * 1.1))
                
                # 分布差异对比
                ax3 = axes[i, 2]
                width = 0.35
                x_pos1 = x_range - width/2
                x_pos2 = x_range + width/2
                ax3.bar(x_pos1, expected_dist, width, label='Expected', alpha=0.7, color='skyblue')
                ax3.bar(x_pos2, actual_dist, width, label='Actual', alpha=0.7, color='lightcoral')
                ax3.set_title(f'EH {eh_idx} Distribution Comparison')
                ax3.set_xlabel('Class')
                ax3.set_ylabel('Proportion')
                ax3.set_xticks(range(num_classes))
                ax3.legend()
                
                # 计算并显示相似度指标
                if np.sum(expected_dist) > 0 and np.sum(actual_dist) > 0:
                    # 计算余弦相似度
                    cosine_sim = np.dot(expected_dist, actual_dist) / (
                        np.linalg.norm(expected_dist) * np.linalg.norm(actual_dist) + 1e-10
                    )
                    ax3.text(0.02, 0.98, f'Cosine Similarity: {cosine_sim:.3f}', 
                            transform=ax3.transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            
            # 确保保存目录存在
            import os
            os.makedirs('./save', exist_ok=True)
            
            plt.savefig('./save/eh_distributions_enhanced.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("Enhanced EH distribution comparison chart saved to ./save/eh_distributions_enhanced.png")
        except Exception as e:
            print(f"Error occurred while visualizing EH distributions: {e}")
            print("Skipping visualization, continuing execution")

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