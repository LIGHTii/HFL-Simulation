#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
双重评估系统 - 支持客户端本地测试集和全局统一测试集的联合评估
实现方案A：双重评估架构
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from utils.sampling import get_data
import copy


class DatasetSplit(Dataset):
    """数据集分割工具类"""
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def get_client_data_with_local_test(args, test_split_ratio=0.2):
    """
    为每个客户端提供完整的本地数据划分（训练集+测试集）
    同时保留全局测试集用于基准对比
    
    Args:
        args: 命令行参数
        test_split_ratio: 本地测试集比例，默认0.2 (20%)
    
    Returns:
        tuple: (full_dataset, global_test, dict_users_local_train, dict_users_local_test, client_classes)
    """
    print("\n" + "="*50)
    print("初始化双重评估数据架构")
    print("="*50)
    
    # 1. 获取原始的IID/Non-IID划分数据
    dataset_train, dataset_test, dict_users_original, client_classes = get_data(args)
    
    print(f"原始训练集大小: {len(dataset_train)}")
    print(f"原始测试集大小: {len(dataset_test)}")
    print(f"客户端数量: {len(dict_users_original)}")
    
    # 2. 为每个客户端创建本地训练/测试划分
    dict_users_local_train = {}
    dict_users_local_test = {}
    
    print("\n开始为每个客户端划分本地训练/测试集...")
    
    for client_id in dict_users_original.keys():
        # 获取该客户端的所有数据索引
        client_indices = list(dict_users_original[client_id])
        
        # 随机打乱索引
        np.random.shuffle(client_indices)
        
        # 计算分割点 (80% 训练, 20% 测试)
        split_point = int(len(client_indices) * (1 - test_split_ratio))
        
        # 分割数据
        dict_users_local_train[client_id] = set(client_indices[:split_point])
        dict_users_local_test[client_id] = set(client_indices[split_point:])
        
        # 打印客户端数据统计
        train_size = len(dict_users_local_train[client_id])
        test_size = len(dict_users_local_test[client_id])
        
        if client_id < 5:  # 只显示前5个客户端的详情
            print(f"  客户端 {client_id}: 训练集={train_size}, 本地测试集={test_size}")
    
    print(f"... (共 {len(dict_users_original)} 个客户端)")
    
    # 3. 统计整体数据分布
    total_local_train = sum([len(indices) for indices in dict_users_local_train.values()])
    total_local_test = sum([len(indices) for indices in dict_users_local_test.values()])
    
    print(f"\n数据划分统计:")
    print(f"  客户端本地训练集总数: {total_local_train}")
    print(f"  客户端本地测试集总数: {total_local_test}")
    print(f"  全局统一测试集: {len(dataset_test)}")
    print(f"  本地测试集比例: {test_split_ratio*100:.1f}%")
    
    print("="*50 + "\n")
    
    return dataset_train, dataset_test, dict_users_local_train, dict_users_local_test, client_classes


def test_img_local(net_g, dataset, idxs, args):
    """
    客户端本地测试函数
    
    Args:
        net_g: 神经网络模型
        dataset: 完整数据集
        idxs: 客户端本地测试数据索引
        args: 参数配置
    
    Returns:
        tuple: (accuracy, loss)
    """
    net_g.eval()
    test_loss = 0
    correct = 0
    
    # 创建本地测试数据加载器
    local_test_dataset = DatasetSplit(dataset, idxs)
    data_loader = DataLoader(local_test_dataset, batch_size=args.bs, shuffle=False)
    
    with torch.no_grad():
        for idx, (data, target) in enumerate(data_loader):
            if args.gpu != -1:
                data, target = data.to(args.device), target.to(args.device)
            
            log_probs = net_g(data)
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
    
    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    
    return accuracy, test_loss


def dual_evaluation(epoch, models_dict, full_dataset, global_test, dict_users_local_test, args, verbose=False):
    """
    双重评估系统主函数
    
    Args:
        epoch: 当前轮次
        models_dict: 模型字典 {'model_name': model}
        full_dataset: 完整训练数据集（用于获取客户端本地测试数据）
        global_test: 全局统一测试集
        dict_users_local_test: 客户端本地测试数据映射
        args: 参数配置
        verbose: 是否显示详细输出
    
    Returns:
        dict: 评估结果
    """
    from models.test import test_img  # 导入全局测试函数
    
    results = {
        'epoch': epoch,
        'global_performance': {},      # 全局统一测试结果
        'local_performance': {},       # 客户端本地测试结果
        'performance_comparison': {}   # 性能对比分析
    }
    
    if verbose:
        print(f"\n--- 轮次 {epoch} 双重评估 ---")
    
    for model_name, model in models_dict.items():
        model.eval()
        
        # 1. 全局统一测试评估
        global_acc, global_loss = test_img(model, global_test, args)
        results['global_performance'][model_name] = {
            'accuracy': global_acc,
            'loss': global_loss
        }
        
        if verbose:
            print(f"{model_name} - 全局测试: 准确率={global_acc:.2f}%, 损失={global_loss:.4f}")
        
        # 2. 客户端本地测试评估
        local_accuracies = []
        local_losses = []
        local_details = []
        
        for client_id, test_indices in dict_users_local_test.items():
            if len(test_indices) > 0:  # 确保客户端有测试数据
                local_acc, local_loss = test_img_local(model, full_dataset, test_indices, args)
                local_accuracies.append(local_acc)
                local_losses.append(local_loss)
                local_details.append({
                    'client_id': client_id,
                    'accuracy': local_acc,
                    'loss': local_loss,
                    'test_size': len(test_indices)
                })
        
        # 计算本地测试统计
        if local_accuracies:
            mean_local_acc = np.mean(local_accuracies)
            std_local_acc = np.std(local_accuracies)
            mean_local_loss = np.mean(local_losses)
            std_local_loss = np.std(local_losses)
            
            results['local_performance'][model_name] = {
                'individual_results': local_details,
                'mean_accuracy': mean_local_acc,
                'std_accuracy': std_local_acc,
                'mean_loss': mean_local_loss,
                'std_loss': std_local_loss,
                'num_clients': len(local_accuracies)
            }
            
            # 3. 性能对比分析
            performance_gap_acc = global_acc - mean_local_acc
            performance_gap_loss = global_loss - mean_local_loss
            
            results['performance_comparison'][model_name] = {
                'accuracy_gap': performance_gap_acc,      # 正值表示全局测试更好
                'loss_gap': performance_gap_loss,         # 负值表示全局测试损失更小
                'local_variance_acc': std_local_acc,      # 客户端间准确率差异
                'local_variance_loss': std_local_loss     # 客户端间损失差异
            }
            
            if verbose:
                print(f"{model_name} - 本地平均: 准确率={mean_local_acc:.2f}±{std_local_acc:.2f}%, 损失={mean_local_loss:.4f}±{std_local_loss:.4f}")
                print(f"{model_name} - 性能差距: 准确率差={performance_gap_acc:+.2f}%, 损失差={performance_gap_loss:+.4f}")
        
        else:
            print(f"警告: {model_name} 没有有效的客户端本地测试数据")
    
    if verbose:
        print("--- 双重评估完成 ---\n")
    
    return results


def analyze_dual_evaluation_results(results_history, save_path=None):
    """
    分析双重评估历史结果，生成深度分析报告
    
    Args:
        results_history: 历史评估结果列表
        save_path: 保存路径（可选）
    
    Returns:
        dict: 分析报告
    """
    if not results_history:
        return {}
    
    analysis_report = {
        'summary': {},
        'trends': {},
        'model_comparison': {}
    }
    
    # 提取所有模型名称
    model_names = list(results_history[0]['global_performance'].keys())
    
    for model_name in model_names:
        # 提取时间序列数据
        epochs = [r['epoch'] for r in results_history]
        global_accs = [r['global_performance'][model_name]['accuracy'] for r in results_history]
        local_mean_accs = [r['local_performance'][model_name]['mean_accuracy'] for r in results_history if model_name in r['local_performance']]
        acc_gaps = [r['performance_comparison'][model_name]['accuracy_gap'] for r in results_history if model_name in r['performance_comparison']]
        
        # 总结统计
        analysis_report['summary'][model_name] = {
            'final_global_acc': global_accs[-1] if global_accs else 0,
            'final_local_mean_acc': local_mean_accs[-1] if local_mean_accs else 0,
            'final_acc_gap': acc_gaps[-1] if acc_gaps else 0,
            'avg_acc_gap': np.mean(acc_gaps) if acc_gaps else 0,
            'acc_gap_trend': 'improving' if len(acc_gaps) > 1 and acc_gaps[-1] < acc_gaps[0] else 'stable'
        }
        
        # 趋势分析
        analysis_report['trends'][model_name] = {
            'epochs': epochs,
            'global_accuracy': global_accs,
            'local_mean_accuracy': local_mean_accs,
            'accuracy_gaps': acc_gaps
        }
    
    print("双重评估分析报告生成完成")
    return analysis_report


if __name__ == "__main__":
    print("双重评估系统测试")
    # 这里可以添加测试代码