#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试客户端数据保存和加载功能
"""

import sys
import os
import argparse
import shutil

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.sampling import get_data
from utils.data_persistence import print_available_data_files


def test_data_persistence():
    """测试数据保存和加载功能"""
    
    print("="*60)
    print("客户端数据保存和加载功能测试")
    print("="*60)
    
    # 创建测试目录
    test_save_dir = './test_saved_data/'
    if os.path.exists(test_save_dir):
        shutil.rmtree(test_save_dir)
    
    # 第一步：测试数据生成和保存
    print("\n📝 第一步：生成新数据并保存...")
    
    # 创建测试参数
    args1 = argparse.Namespace(
        dataset='mnist',
        data_path='../data/',
        num_users=10,  # 使用较小的客户端数量进行测试
        partition='noniid-labeldir',
        beta=0.1,
        iid=False,
        use_sampling=False,
        seed=42,
        save_data=True,  # 保存数据
        load_data=None,  # 不加载数据
        data_save_dir=test_save_dir
    )
    
    try:
        # 生成数据
        dataset_train1, dataset_test1, dict_users1, client_classes1 = get_data(args1)
        
        print(f"✅ 成功生成数据:")
        print(f"   训练集大小: {len(dataset_train1)}")
        print(f"   测试集大小: {len(dataset_test1)}")
        print(f"   客户端数: {len(dict_users1)}")
        print(f"   客户端类别信息: {len(client_classes1)}")
        
        # 显示每个客户端的数据分布
        for client_id in range(min(5, len(dict_users1))):  # 只显示前5个客户端
            data_count = len(dict_users1[client_id])
            class_count = len(client_classes1[client_id]) if client_id in client_classes1 else 0
            print(f"   客户端 {client_id}: {data_count} 个样本, {class_count} 个类别")
        
        print("\n📂 查看保存的文件:")
        print_available_data_files(test_save_dir)
        
    except Exception as e:
        print(f"❌ 数据生成和保存失败: {str(e)}")
        return False
    
    # 第二步：测试数据加载
    print("\n📖 第二步：从文件加载数据...")
    
    # 找到保存的文件
    saved_files = []
    if os.path.exists(test_save_dir):
        for filename in os.listdir(test_save_dir):
            if filename.endswith('.pkl') and 'client_data_' in filename:
                saved_files.append(os.path.join(test_save_dir, filename))
    
    if not saved_files:
        print("❌ 没有找到保存的数据文件")
        return False
    
    # 使用第一个找到的文件
    data_file = saved_files[0]
    print(f"📁 使用文件: {os.path.basename(data_file)}")
    
    # 创建加载参数
    args2 = argparse.Namespace(
        dataset='mnist',
        data_path='../data/',
        num_users=10,
        partition='noniid-labeldir',
        beta=0.1,
        iid=False,
        use_sampling=False,
        seed=42,
        save_data=False,  # 不保存数据
        load_data=data_file,  # 加载数据
        data_save_dir=test_save_dir
    )
    
    try:
        # 加载数据
        dataset_train2, dataset_test2, dict_users2, client_classes2 = get_data(args2)
        
        print(f"✅ 成功加载数据:")
        print(f"   训练集大小: {len(dataset_train2)}")
        print(f"   测试集大小: {len(dataset_test2)}")
        print(f"   客户端数: {len(dict_users2)}")
        print(f"   客户端类别信息: {len(client_classes2)}")
        
    except Exception as e:
        print(f"❌ 数据加载失败: {str(e)}")
        return False
    
    # 第三步：验证数据一致性
    print("\n🔍 第三步：验证数据一致性...")
    
    try:
        # 检查客户端数量
        if len(dict_users1) != len(dict_users2):
            print(f"❌ 客户端数量不一致: {len(dict_users1)} vs {len(dict_users2)}")
            return False
        
        # 检查每个客户端的数据是否一致
        for client_id in dict_users1.keys():
            if client_id not in dict_users2:
                print(f"❌ 客户端 {client_id} 在加载的数据中不存在")
                return False
            
            # 将 set 转换为 sorted list 进行比较
            indices1 = sorted(list(dict_users1[client_id]))
            indices2 = sorted(list(dict_users2[client_id]))
            
            if indices1 != indices2:
                print(f"❌ 客户端 {client_id} 的数据索引不一致")
                return False
        
        # 检查客户端类别信息
        for client_id in client_classes1.keys():
            if client_id not in client_classes2:
                print(f"❌ 客户端 {client_id} 的类别信息在加载的数据中不存在")
                return False
            
            classes1 = sorted(client_classes1[client_id])
            classes2 = sorted(client_classes2[client_id])
            
            if classes1 != classes2:
                print(f"❌ 客户端 {client_id} 的类别信息不一致")
                print(f"   原始: {classes1}")
                print(f"   加载: {classes2}")
                return False
        
        print("✅ 数据一致性验证通过!")
        
    except Exception as e:
        print(f"❌ 数据一致性验证失败: {str(e)}")
        return False
    
    # 第四步：测试配置不一致的情况
    print("\n⚠️  第四步：测试配置不一致的情况...")
    
    # 创建不同配置的参数
    args3 = argparse.Namespace(
        dataset='mnist',
        data_path='../data/',
        num_users=20,  # 不同的客户端数量
        partition='noniid-labeldir',
        beta=0.2,  # 不同的beta参数
        iid=False,
        use_sampling=False,
        seed=42,
        save_data=False,
        load_data=data_file,  # 加载相同的文件
        data_save_dir=test_save_dir
    )
    
    try:
        print("尝试使用不一致的配置加载数据（应该显示警告）...")
        # 这里应该会显示配置不一致的警告
        dataset_train3, dataset_test3, dict_users3, client_classes3 = get_data(args3)
        
        if dict_users3 is None:
            print("✅ 正确拒绝了不一致的配置")
        else:
            print("⚠️  警告：接受了不一致的配置（可能用户选择了继续）")
        
    except SystemExit:
        print("✅ 正确退出了不一致的配置加载")
    except Exception as e:
        print(f"❓ 配置不一致测试结果: {str(e)}")
    
    # 清理测试文件
    print(f"\n🧹 清理测试文件: {test_save_dir}")
    if os.path.exists(test_save_dir):
        shutil.rmtree(test_save_dir)
    
    print("\n" + "="*60)
    print("✅ 客户端数据保存和加载功能测试完成!")
    print("="*60)
    
    return True


if __name__ == '__main__':
    # 模拟用户输入 'y' 来继续不一致的配置（用于自动测试）
    import builtins
    original_input = builtins.input
    
    def mock_input(prompt):
        if "是否继续使用不一致的配置" in prompt:
            print("模拟用户输入: N")
            return 'N'
        return original_input(prompt)
    
    builtins.input = mock_input
    
    try:
        success = test_data_persistence()
        if success:
            print("🎉 所有测试通过!")
        else:
            print("💥 测试失败!")
    finally:
        # 恢复原始的input函数
        builtins.input = original_input