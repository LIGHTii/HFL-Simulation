#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 测试数据获取函数的修复

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 模拟命令行参数
class Args:
    def __init__(self):
        self.dataset = 'mnist'
        self.num_users = 10
        self.data_path = './data/'
        self.partition = 'noniid-labeldir'
        self.beta = 0.1
        self.use_sampling = False

# 测试数据获取函数
def test_data_loading():
    print("测试数据加载函数修复...")
    
    try:
        # 导入相关函数
        from utils.sampling import get_data
        
        # 创建测试参数
        args = Args()
        
        print(f"测试参数: dataset={args.dataset}, num_users={args.num_users}")
        print("开始调用get_data函数...")
        
        # 调用函数
        dataset_train, dataset_test, dict_users, client_classes = get_data(args)
        
        print("✅ 数据加载成功！")
        print(f"训练数据集类型: {type(dataset_train)}")
        print(f"测试数据集类型: {type(dataset_test)}")
        print(f"客户端数据映射: {len(dict_users)} 个客户端")
        print(f"客户端类别映射: {len(client_classes)} 个客户端")
        
        # 显示前几个客户端的信息
        print("\n前5个客户端信息:")
        for i in range(min(5, len(client_classes))):
            client_samples = len(dict_users[i]) if i in dict_users else 0
            client_labels = client_classes.get(i, [])
            print(f"  客户端 {i}: {client_samples} 个样本, 类别: {client_labels}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_data_loading()
    if success:
        print("\n🎉 修复验证成功！现在可以正常运行main_fed.py了")
    else:
        print("\n💥 还有问题需要进一步修复")