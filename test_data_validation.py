#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据划分验证测试脚本
用于测试客户端数据分配的正确性和一致性
"""

import sys
import os
import torch
import numpy as np
from utils.options import args_parser
from utils.sampling import get_data
from utils.bipartite_bandwidth import run_bandwidth_allocation

def test_data_distribution_consistency():
    """测试数据分配的一致性"""
    print("="*60)
    print("客户端数据分配一致性测试")
    print("="*60)
    
    # 解析参数
    args = args_parser()
    args.device = torch.device('cuda' if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    
    print(f"初始参数设置:")
    print(f"  数据集: {args.dataset}")
    print(f"  设定客户端数: {args.num_users}")
    print(f"  IID模式: {args.iid}")
    print(f"  Beta参数: {getattr(args, 'beta', 'N/A')}")
    print(f"  网络文件: {args.graphml_file}")
    
    # 第一步：获取网络拓扑信息
    print("\n第一步：分析网络拓扑...")
    try:
        bipartite_graph, client_nodes, active_es_nodes, A_design, r_client_to_es, r_es, r_es_to_cloud, r_client_to_cloud = run_bandwidth_allocation(
            graphml_file=args.graphml_file, 
            es_ratio=args.es_ratio, 
            max_capacity=args.max_capacity, 
            visualize=False)
        
        if bipartite_graph is None:
            print("❌ 网络拓扑构建失败")
            return False
        
        actual_num_users = len(client_nodes)
        print(f"✅ 网络拓扑分析完成")
        print(f"  实际客户端数量: {actual_num_users}")
        print(f"  边缘服务器数量: {len(active_es_nodes)}")
        
        # 更新参数
        original_num_users = args.num_users
        args.num_users = actual_num_users
        print(f"  参数更新: {original_num_users} -> {actual_num_users}")
        
    except Exception as e:
        print(f"❌ 网络拓扑分析失败: {e}")
        return False
    
    # 第二步：获取数据分配
    print("\n第二步：生成客户端数据分配...")
    try:
        dataset_train, dataset_test, dict_users, client_classes = get_data(args)
        print(f"✅ 数据分配生成完成")
        print(f"  训练集大小: {len(dataset_train)}")
        print(f"  测试集大小: {len(dataset_test)}")
        print(f"  分配的客户端数: {len(dict_users)}")
        
    except Exception as e:
        print(f"❌ 数据分配生成失败: {e}")
        return False
    
    # 第三步：详细验证数据分配
    print("\n第三步：验证数据分配一致性...")
    
    # 检查1：客户端数量一致性
    if len(dict_users) != args.num_users:
        print(f"❌ 客户端数量不一致: dict_users={len(dict_users)}, args.num_users={args.num_users}")
        return False
    
    # 检查2：客户端ID连续性
    expected_ids = set(range(args.num_users))
    actual_ids = set(dict_users.keys())
    if expected_ids != actual_ids:
        print(f"❌ 客户端ID不连续")
        print(f"  期望: {sorted(expected_ids)}")
        print(f"  实际: {sorted(actual_ids)}")
        return False
    
    # 检查3：数据索引有效性
    total_samples = 0
    sample_counts = []
    invalid_clients = []
    
    for client_id, data_indices in dict_users.items():
        # 处理不同的数据结构
        if isinstance(data_indices, set):
            indices_list = list(data_indices)
        elif isinstance(data_indices, np.ndarray):
            indices_list = data_indices.tolist()
        else:
            indices_list = list(data_indices)
        
        sample_counts.append(len(indices_list))
        total_samples += len(indices_list)
        
        # 检查索引范围
        if indices_list:
            min_idx = min(indices_list)
            max_idx = max(indices_list)
            
            if min_idx < 0 or max_idx >= len(dataset_train):
                invalid_clients.append({
                    'client_id': client_id,
                    'min_idx': min_idx,
                    'max_idx': max_idx,
                    'count': len(indices_list)
                })
    
    if invalid_clients:
        print(f"❌ 发现无效数据索引:")
        for client in invalid_clients:
            print(f"  客户端{client['client_id']}: 索引范围[{client['min_idx']}, {client['max_idx']}], 数据集大小{len(dataset_train)}")
        return False
    
    # 检查4：数据分配统计
    print(f"✅ 数据分配验证通过:")
    print(f"  总分配样本数: {total_samples}")
    print(f"  数据集总大小: {len(dataset_train)}")
    print(f"  样本利用率: {total_samples/len(dataset_train)*100:.1f}%")
    print(f"  平均每客户端: {np.mean(sample_counts):.1f}样本")
    print(f"  样本数范围: [{min(sample_counts)}, {max(sample_counts)}]")
    print(f"  样本数标准差: {np.std(sample_counts):.2f}")
    
    # 检查5：客户端类别信息（如果有的话）
    if client_classes:
        print(f"  客户端类别信息: {len(client_classes)}个客户端有类别信息")
        
        # 统计类别分布
        class_counts = {}
        for client_id, classes in client_classes.items():
            num_classes = len(classes) if isinstance(classes, (list, set, np.ndarray)) else 1
            class_counts[num_classes] = class_counts.get(num_classes, 0) + 1
        
        print(f"  类别数分布: {dict(sorted(class_counts.items()))}")
    
    print(f"\n🎉 所有验证都通过！数据分配是正确和一致的。")
    return True

def test_data_persistence():
    """测试数据持久化功能"""
    print("\n" + "="*60)
    print("数据持久化功能测试")
    print("="*60)
    
    # TODO: 添加数据保存和加载的测试
    print("数据持久化测试待实现...")
    return True

if __name__ == "__main__":
    print("开始数据分配验证测试...\n")
    
    # 测试数据分配一致性
    success1 = test_data_distribution_consistency()
    
    # 测试数据持久化
    success2 = test_data_persistence()
    
    print("\n" + "="*60)
    if success1 and success2:
        print("🎉 所有测试都通过！")
        sys.exit(0)
    else:
        print("❌ 部分测试失败！")
        sys.exit(1)