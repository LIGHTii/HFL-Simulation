#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试修改后的数据分割函数是否能处理大量用户
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

import numpy as np
from torchvision import datasets, transforms
from utils.sampling import mnist_noniid, mnist_iid, cifar_noniid_adapted, cifar_iid

def test_sampling_functions():
    """测试修改后的采样函数"""
    print("=== 测试修改后的数据分割函数 ===")
    
    # 加载MNIST数据集
    transform = transforms.Compose([transforms.ToTensor()])
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=False, transform=transform)
    
    # 测试大量用户的情况
    test_cases = [50, 100, 150, 200, 300]  # 不同的用户数量
    
    for num_users in test_cases:
        print(f"\n--- 测试 {num_users} 个用户 ---")
        
        try:
            # 测试 MNIST Non-IID
            dict_users_noniid = mnist_noniid(dataset_train, num_users)
            print(f"✅ MNIST Non-IID: 成功为 {num_users} 个用户分配数据")
            
            # 验证数据分配
            total_samples = sum(len(user_data) for user_data in dict_users_noniid.values())
            avg_samples = total_samples / num_users
            print(f"   平均每个用户: {avg_samples:.1f} 个样本")
            
            # 测试 MNIST IID
            dict_users_iid = mnist_iid(dataset_train, num_users)
            print(f"✅ MNIST IID: 成功为 {num_users} 个用户分配数据")
            
        except Exception as e:
            print(f"❌ 用户数量 {num_users} 失败: {e}")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_sampling_functions()
