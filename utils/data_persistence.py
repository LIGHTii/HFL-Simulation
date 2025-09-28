#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
客户端数据分配保存和加载工具模块
用于保存和复用客户端数据分配方案
"""

import os
import pickle
import json
import numpy as np
from datetime import datetime
import hashlib


def generate_data_config_hash(args):
    """
    生成数据配置的哈希值，用于验证数据分配方案的一致性
    
    Args:
        args: 命令行参数对象
    
    Returns:
        str: 配置的MD5哈希值
    """
    config_dict = {
        'dataset': args.dataset,
        'num_users': args.num_users,
        'partition': getattr(args, 'partition', 'noniid-labeldir'),
        'beta': getattr(args, 'beta', 0.1),
        'iid': getattr(args, 'iid', False),
        'use_sampling': getattr(args, 'use_sampling', False),
        'seed': args.seed,
        # 新增影响客户端数量的网络拓扑参数
        'graphml_file': getattr(args, 'graphml_file', None),
        'es_ratio': getattr(args, 'es_ratio', None),
        'max_capacity': getattr(args, 'max_capacity', None),
        # 新增其他可能影响数据划分的参数
        'data_path': getattr(args, 'data_path', './data/'),
        'local_ep': getattr(args, 'local_ep', 5),
        'method': getattr(args, 'method', 'fedavg')
    }
    
    # 将字典转换为排序的字符串
    config_str = json.dumps(config_dict, sort_keys=True)
    
    # 生成MD5哈希
    return hashlib.md5(config_str.encode()).hexdigest()


def save_client_data_distribution(dict_users, client_classes, args, custom_name=None):
    """
    保存客户端数据分配信息到文件
    
    Args:
        dict_users: 客户端数据索引映射字典
        client_classes: 客户端类别信息字典
        args: 命令行参数对象
        custom_name: 自定义文件名（可选）
    
    Returns:
        str: 保存的文件路径
    """
    # 确保保存目录存在
    save_dir = args.data_save_dir
    os.makedirs(save_dir, exist_ok=True)
    
    # 生成配置哈希
    config_hash = generate_data_config_hash(args)
    
    # 生成文件名
    if custom_name:
        filename = f"{custom_name}.pkl"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"client_data_{args.dataset}_{args.num_users}clients_{config_hash[:8]}_{timestamp}.pkl"
    
    filepath = os.path.join(save_dir, filename)
    
    # 准备保存的数据
    save_data = {
        'dict_users': dict_users,
        'client_classes': client_classes,
        'config_hash': config_hash,
        'config': {
            'dataset': args.dataset,
            'num_users': args.num_users,
            'partition': getattr(args, 'partition', 'noniid-labeldir'),
            'beta': getattr(args, 'beta', 0.1),
            'iid': getattr(args, 'iid', False),
            'use_sampling': getattr(args, 'use_sampling', False),
            'seed': args.seed,
            'save_timestamp': datetime.now().isoformat()
        },
        'metadata': {
            'total_clients': len(dict_users),
            'total_samples': sum(len(indices) for indices in dict_users.values()),
            'classes_per_client': {client_id: len(classes) for client_id, classes in client_classes.items()},
            'unique_classes': list(set(class_id for classes in client_classes.values() for class_id in classes))
        }
    }
    
    # 保存到文件
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"✅ 客户端数据分配已保存到: {filepath}")
        print(f"   配置哈希: {config_hash}")
        print(f"   总客户端数: {save_data['metadata']['total_clients']}")
        print(f"   总样本数: {save_data['metadata']['total_samples']}")
        
        # 保存一个可读的配置文件
        config_filepath = filepath.replace('.pkl', '_config.json')
        with open(config_filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data['config'], f, indent=2, ensure_ascii=False)
        
        return filepath
        
    except Exception as e:
        print(f"❌ 保存客户端数据分配失败: {str(e)}")
        return None


def load_client_data_distribution(filepath, args, verify_config=True):
    """
    从文件加载客户端数据分配信息
    
    Args:
        filepath: 数据文件路径
        args: 命令行参数对象
        verify_config: 是否验证配置一致性
    
    Returns:
        tuple: (dict_users, client_classes) 或 None (如果加载失败)
    """
    if not os.path.exists(filepath):
        print(f"❌ 数据文件不存在: {filepath}")
        return None, None
    
    try:
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        # 验证数据格式
        if not all(key in save_data for key in ['dict_users', 'client_classes', 'config']):
            print("❌ 数据文件格式不正确")
            return None, None
        
        dict_users = save_data['dict_users']
        client_classes = save_data['client_classes']
        saved_config = save_data['config']
        
        print(f"✅ 成功加载客户端数据分配: {filepath}")
        print(f"   数据集: {saved_config['dataset']}")
        print(f"   客户端数: {saved_config['num_users']}")
        print(f"   分区方法: {saved_config['partition']}")
        print(f"   保存时间: {saved_config.get('save_timestamp', 'Unknown')}")
        
        # 自动使用保存数据时的随机种子，确保实验一致性
        if 'seed' in saved_config and saved_config['seed'] != args.seed:
            old_seed = args.seed
            args.seed = saved_config['seed']
            print(f"🔄 自动更新随机种子: {old_seed} -> {args.seed} (使用保存数据时的种子)")
            
            # 立即设置随机种子
            import torch
            import numpy as np
            import random
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            random.seed(args.seed)
            print(f"   已设置所有随机种子为: {args.seed}")
        
        # 验证配置一致性
        if verify_config:
            current_hash = generate_data_config_hash(args)
            saved_hash = save_data.get('config_hash', '')
            
            if current_hash != saved_hash:
                print("⚠️  警告: 当前配置与保存的配置不一致!")
                print(f"   当前配置哈希: {current_hash}")
                print(f"   保存的配置哈希: {saved_hash}")
                
                # 重新计算哈希（因为种子可能已经自动更新）
                updated_hash = generate_data_config_hash(args)
                
                if updated_hash != saved_hash:
                    # 显示具体差异
                    current_config = {
                        'dataset': args.dataset,
                        'num_users': args.num_users,
                        'partition': getattr(args, 'partition', 'noniid-labeldir'),
                        'beta': getattr(args, 'beta', 0.1),
                        'iid': getattr(args, 'iid', False),
                        'use_sampling': getattr(args, 'use_sampling', False),
                        'seed': args.seed
                    }
                    
                    print(f"   更新后配置哈希: {updated_hash}")
                    print("\n   配置差异:")
                    for key in current_config:
                        if key in saved_config and current_config[key] != saved_config[key]:
                            print(f"   - {key}: 当前={current_config[key]}, 保存={saved_config[key]}")
                    
                    response = input("\n   是否继续使用不一致的配置? (y/N): ")
                    if response.lower() != 'y':
                        return None, None
                else:
                    print("✅ 配置验证通过（种子已自动同步）")
        
        # 显示数据统计信息
        if 'metadata' in save_data:
            metadata = save_data['metadata']
            print(f"\n📊 数据统计:")
            print(f"   总样本数: {metadata.get('total_samples', 'Unknown')}")
            print(f"   类别数: {len(metadata.get('unique_classes', []))}")
            
            # 显示客户端类别分布统计
            classes_per_client = metadata.get('classes_per_client', {})
            if classes_per_client:
                class_counts = {}
                for client_id, num_classes in classes_per_client.items():
                    class_counts[num_classes] = class_counts.get(num_classes, 0) + 1
                
                print("   客户端类别分布:")
                for num_classes, count in sorted(class_counts.items()):
                    print(f"     拥有 {num_classes} 个类别的客户端: {count} 个")
        
        return dict_users, client_classes
        
    except Exception as e:
        print(f"❌ 加载客户端数据分配失败: {str(e)}")
        return None, None


def list_saved_data_files(data_save_dir):
    """
    列出所有保存的数据文件
    
    Args:
        data_save_dir: 数据保存目录
    
    Returns:
        list: 数据文件路径列表
    """
    if not os.path.exists(data_save_dir):
        return []
    
    files = []
    for filename in os.listdir(data_save_dir):
        if filename.endswith('.pkl') and 'client_data_' in filename:
            files.append(os.path.join(data_save_dir, filename))
    
    return sorted(files, key=os.path.getmtime, reverse=True)


def print_available_data_files(data_save_dir):
    """
    打印所有可用的数据文件
    
    Args:
        data_save_dir: 数据保存目录
    """
    files = list_saved_data_files(data_save_dir)
    
    if not files:
        print(f"在目录 {data_save_dir} 中没有找到保存的数据文件")
        return
    
    print(f"\n📁 可用的客户端数据文件 ({len(files)} 个):")
    print("-" * 80)
    
    for i, filepath in enumerate(files, 1):
        try:
            # 尝试读取配置信息
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)
            
            config = save_data.get('config', {})
            metadata = save_data.get('metadata', {})
            
            filename = os.path.basename(filepath)
            print(f"{i:2d}. {filename}")
            print(f"    数据集: {config.get('dataset', 'Unknown')}")
            print(f"    客户端数: {config.get('num_users', 'Unknown')}")
            print(f"    分区方法: {config.get('partition', 'Unknown')}")
            print(f"    Beta参数: {config.get('beta', 'Unknown')}")
            print(f"    总样本数: {metadata.get('total_samples', 'Unknown')}")
            print(f"    保存时间: {config.get('save_timestamp', 'Unknown')}")
            print()
            
        except Exception as e:
            filename = os.path.basename(filepath)
            print(f"{i:2d}. {filename} (读取配置失败: {str(e)})")
            print()


if __name__ == '__main__':
    # 测试代码
    print("客户端数据分配保存/加载工具测试")
    
    # 示例用法
    import argparse
    
    # 创建示例参数
    args = argparse.Namespace(
        dataset='mnist',
        num_users=50,
        partition='noniid-labeldir',
        beta=0.1,
        iid=False,
        use_sampling=False,
        seed=1,
        data_save_dir='./test_saved_data/'
    )
    
    # 创建示例数据
    dict_users = {i: set(range(i*100, (i+1)*100)) for i in range(5)}
    client_classes = {i: [i % 3, (i+1) % 3] for i in range(5)}
    
    # 测试保存
    filepath = save_client_data_distribution(dict_users, client_classes, args, "test_data")
    
    if filepath:
        # 测试加载
        loaded_dict_users, loaded_client_classes = load_client_data_distribution(filepath, args)
        
        if loaded_dict_users is not None:
            print("\n✅ 测试成功!")
        else:
            print("\n❌ 测试失败!")