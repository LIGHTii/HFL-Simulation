#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强的本地更新类 - 支持客户端本地测试功能
扩展原有的LocalUpdate类以支持双重评估架构
"""

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
import random
from models.Update import LocalUpdate, DatasetSplit, restricted_softmax


class LocalUpdateWithLocalTest(LocalUpdate):
    """
    支持本地测试的增强版LocalUpdate类
    在原有训练功能基础上，添加客户端本地测试能力
    """
    
    def __init__(self, args, dataset=None, train_idxs=None, test_idxs=None, user_classes=None):
        """
        初始化增强版本地更新器
        
        Args:
            args: 参数配置
            dataset: 完整数据集
            train_idxs: 客户端本地训练数据索引
            test_idxs: 客户端本地测试数据索引  
            user_classes: 客户端拥有的类别信息（用于FedRS）
        """
        # 调用父类初始化（但需要重写数据加载器部分）
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.user_classes = user_classes
        
        # 创建本地训练数据加载器
        if train_idxs is not None:
            self.ldr_train = DataLoader(
                DatasetSplit(dataset, train_idxs), 
                batch_size=self.args.local_bs, 
                shuffle=True
            )
            self.train_data_size = len(train_idxs)
        else:
            self.ldr_train = None
            self.train_data_size = 0
        
        # 创建本地测试数据加载器
        if test_idxs is not None:
            self.ldr_test = DataLoader(
                DatasetSplit(dataset, test_idxs), 
                batch_size=self.args.bs, 
                shuffle=False
            )
            self.test_data_size = len(test_idxs)
        else:
            self.ldr_test = None
            self.test_data_size = 0
        
        print(f"客户端本地数据: 训练集={self.train_data_size}, 测试集={self.test_data_size}")
    
    def train(self, net):
        """
        客户端本地训练（继承原有逻辑）
        
        Args:
            net: 神经网络模型
            
        Returns:
            tuple: (模型参数, 平均损失)
        """
        if self.ldr_train is None:
            print("警告: 客户端没有训练数据")
            return net.state_dict(), 0.0
        
        net.train()
        
        # 为 FedRS 算法支持动态 local epochs
        if self.args.method == 'fedrs' and hasattr(self.args, 'min_le') and hasattr(self.args, 'max_le'):
            local_ep = random.randint(self.args.min_le, self.args.max_le)
        else:
            local_ep = self.args.local_ep
        
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                
                # FedRS 算法: 应用受限制的 softmax
                if self.args.method == 'fedrs' and self.user_classes is not None:
                    log_probs = restricted_softmax(log_probs, self.user_classes, self.args)
                
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
    def local_test(self, net):
        """
        客户端本地测试功能
        
        Args:
            net: 神经网络模型
            
        Returns:
            tuple: (准确率, 损失)
        """
        if self.ldr_test is None:
            print("警告: 客户端没有本地测试数据")
            return 0.0, float('inf')
        
        net.eval()
        test_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in self.ldr_test:
                data, target = data.to(self.args.device), target.to(self.args.device)
                output = net(data)
                
                # sum up batch loss
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                
                # get the index of the max log-probability
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        test_loss /= len(self.ldr_test.dataset)
        accuracy = 100. * correct / len(self.ldr_test.dataset)
        
        if self.args.verbose:
            print(f'本地测试结果: 准确率: {correct}/{len(self.ldr_test.dataset)} ({accuracy:.2f}%), 平均损失: {test_loss:.4f}')
        
        return accuracy, test_loss
    
    def get_data_statistics(self):
        """
        获取客户端数据统计信息
        
        Returns:
            dict: 数据统计
        """
        stats = {
            'train_size': self.train_data_size,
            'test_size': self.test_data_size,
            'total_size': self.train_data_size + self.test_data_size,
            'test_ratio': self.test_data_size / (self.train_data_size + self.test_data_size) if (self.train_data_size + self.test_data_size) > 0 else 0
        }
        
        # 分析标签分布（如果有本地训练数据）
        if self.ldr_train is not None:
            label_counts = {}
            for _, labels in self.ldr_train:
                for label in labels:
                    label_item = label.item()
                    label_counts[label_item] = label_counts.get(label_item, 0) + 1
            stats['train_label_distribution'] = label_counts
        
        # 分析测试标签分布（如果有本地测试数据）
        if self.ldr_test is not None:
            label_counts = {}
            for _, labels in self.ldr_test:
                for label in labels:
                    label_item = label.item()
                    label_counts[label_item] = label_counts.get(label_item, 0) + 1
            stats['test_label_distribution'] = label_counts
        
        return stats
    
    def validate_data_consistency(self):
        """
        验证训练集和测试集的数据一致性
        检查标签分布是否合理
        
        Returns:
            dict: 验证结果
        """
        validation_result = {
            'is_valid': True,
            'issues': [],
            'recommendations': []
        }
        
        # 检查是否有数据
        if self.train_data_size == 0:
            validation_result['is_valid'] = False
            validation_result['issues'].append("客户端缺少训练数据")
        
        if self.test_data_size == 0:
            validation_result['issues'].append("客户端缺少测试数据")
            validation_result['recommendations'].append("建议增加测试数据比例")
        
        # 检查数据比例是否合理
        total_size = self.train_data_size + self.test_data_size
        if total_size > 0:
            test_ratio = self.test_data_size / total_size
            if test_ratio < 0.1:
                validation_result['recommendations'].append(f"测试集比例较低 ({test_ratio:.2%})")
            elif test_ratio > 0.4:
                validation_result['recommendations'].append(f"测试集比例较高 ({test_ratio:.2%})")
        
        return validation_result


def create_enhanced_local_updates(args, dataset, dict_users_train, dict_users_test, client_classes):
    """
    批量创建增强版本地更新器
    
    Args:
        args: 参数配置
        dataset: 完整数据集
        dict_users_train: 客户端训练数据映射
        dict_users_test: 客户端测试数据映射  
        client_classes: 客户端类别映射
        
    Returns:
        dict: {client_id: LocalUpdateWithLocalTest}
    """
    enhanced_local_updates = {}
    
    print(f"\n创建 {len(dict_users_train)} 个增强版本地更新器...")
    
    for client_id in dict_users_train.keys():
        train_idxs = dict_users_train.get(client_id, set())
        test_idxs = dict_users_test.get(client_id, set())
        user_classes = client_classes.get(client_id, [])
        
        enhanced_local_updates[client_id] = LocalUpdateWithLocalTest(
            args=args,
            dataset=dataset,
            train_idxs=train_idxs,
            test_idxs=test_idxs,
            user_classes=user_classes
        )
        
        # 验证数据一致性（只对前几个客户端显示详情）
        if client_id < 3:
            validation = enhanced_local_updates[client_id].validate_data_consistency()
            if not validation['is_valid'] or validation['issues']:
                print(f"  客户端 {client_id} 数据验证:")
                for issue in validation['issues']:
                    print(f"    ⚠️  {issue}")
                for rec in validation['recommendations']:
                    print(f"    💡 {rec}")
    
    print("增强版本地更新器创建完成\n")
    return enhanced_local_updates


if __name__ == "__main__":
    print("LocalUpdateWithLocalTest 测试")
    # 这里可以添加单元测试代码