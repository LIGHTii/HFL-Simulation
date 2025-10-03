#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
import random

import numpy as np
from torchvision import datasets, transforms
from utils.data_partition import get_client_datasets
from utils.data_persistence import (
    save_client_data_distribution, 
    load_client_data_distribution,
    print_available_data_files
)


def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)##计算每个用户应该拥有的样本数量
    dict_users = {}
    all_idxs = [i for i in range(len(dataset))] ##包含所有数据索引的列表
    
    for i in range(num_users):  ##给每个用户分配数据的索引
        if len(all_idxs) >= num_items:
            # 如果剩余数据足够，不重复分配
            dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
            all_idxs = list(set(all_idxs) - dict_users[i])
        else:
            # 如果剩余数据不够，允许重复分配
            dict_users[i] = set(np.random.choice(len(dataset), num_items, replace=False))
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300 ##num_shards：将数据划分为的片段数量；num_imgs:每个片段的图像数量
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}  ##每个用户对应的样本索引
    idxs = np.arange(num_shards*num_imgs) ##创建一个包含所有图像索引的数组，就是一个0-的有序数组
    labels = dataset.train_labels.numpy() ##将数据集中样本的标签转换为numpy

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()] ##根据标签对图像进行排序 ，返回为有序数列的原位置的序号例如，如果标签的原始顺序是 [5, 0, 4, 1, 2]，使用 argsort() 后会返回 [1, 4, 3, 2, 0]，表示标签1排在最前，标签5排在最后。
    idxs = idxs_labels[0,:] ##idxs排序后的索引，相当于标签排序得到有序，对应索引跟随改变，得到新位置
       ##其将相同类别的图像集中在一起

    # divide and assign 将数据分配给每个用户 - 允许重叠分配
    shards_per_user = 2  # 每个用户分配的片段数量
    for i in range(num_users):
        # 允许重复选择片段来支持更多用户
        rand_set = set(np.random.choice(num_shards, shards_per_user, replace=False))
        for rand in rand_set: ##将所选片段
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)  ##计算每个用户应该拥有的样本数量
    dict_users = {}
    all_idxs = [i for i in range(len(dataset))]
    
    for i in range(num_users):  ##给每个用户分配数据的索引
        if len(all_idxs) >= num_items:
            # 如果剩余数据足够，不重复分配
            dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
            all_idxs = list(set(all_idxs) - dict_users[i])
        else:
            # 如果剩余数据不够，允许重复分配
            dict_users[i] = set(np.random.choice(len(dataset), num_items, replace=False))
    return dict_users

# new
def cifar_noniid_adapted(dataset, num_users):
    """
    CIFAR数据集的Non-IID划分，基于MNIST Non-IID方法适配
    将10个类别划分为200个分片，每个客户端分配2个分片
    允许片段重叠以支持更多用户
    """
    num_shards, num_imgs = 200, 250  # CIFAR每个分片250张图片
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = np.array(dataset.targets)

    # 按标签排序
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # 为每个客户端随机分配2个分片 - 允许重叠分配
    shards_per_user = 2  # 每个用户分配的片段数量
    for i in range(num_users):
        # 允许重复选择片段来支持更多用户
        rand_set = set(np.random.choice(num_shards, shards_per_user, replace=False))
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)

    return dict_users

def cifar100_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR100 dataset
    """
    num_items = int(len(dataset) / num_users)
    dict_users = {}
    all_idxs = [i for i in range(len(dataset))]
    
    for i in range(num_users):
        if len(all_idxs) >= num_items:
            # 如果剩余数据足够，不重复分配
            dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
            all_idxs = list(set(all_idxs) - dict_users[i])
        else:
            # 如果剩余数据不够，允许重复分配
            dict_users[i] = set(np.random.choice(len(dataset), num_items, replace=False))
    return dict_users

# 新增: CIFAR-100 Non-IID (适配自 mnist_noniid)
def cifar100_noniid_adapted(dataset, num_users):
    """
    CIFAR-100数据集的Non-IID划分，基于MNIST Non-IID方法适配
    将100个类别划分为500个分片，每个客户端分配5个分片 (CIFAR-100 train: 50000图像)
    允许片段重叠以支持更多用户
    """
    num_shards, num_imgs = 500, 100  # 500 shards * 100 imgs = 50000
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = np.array(dataset.targets)

    # 按标签排序
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # 为每个客户端随机分配5个分片 - 允许重叠分配
    shards_per_user = 5  # 每个用户分配的片段数量
    for i in range(num_users):
        # 允许重复选择片段来支持更多用户
        rand_set = set(np.random.choice(num_shards, shards_per_user, replace=False))
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)

    return dict_users


def get_client_classes_from_sampling(dataset, dict_users):
    """
    从sampling.py的数据划分结果中提取客户端类别信息
    """
    client_classes = {}

    for client_id, indices in dict_users.items():
        # 获取该客户端数据的所有标签
        if hasattr(dataset, 'targets'):
            labels = np.array(dataset.targets)[indices.astype(int)]
        else:
            labels = np.array([dataset[int(idx)][1] for idx in indices])

        # 获取唯一的类别
        unique_classes = np.unique(labels).tolist()
        client_classes[client_id] = unique_classes

    return client_classes

def get_data_new(dataset_type, num_clients, data_path, partition_method='homo', noniid_param=0.4):
    """
    使用新的数据划分函数获取数据

    Args:
        dataset_type (str): 数据集类型 ('mnist', 'cifar10', 'cifar100')
        num_clients (int): 客户端数量
        data_path (str): 数据保存路径
        partition_method (str): 数据分区方式
        noniid_param (float): non-iid分布参数

    Returns:
        tuple: (训练数据集, 测试数据集, 客户端数据映射, 客户端类别映射)
    """

    return get_client_datasets(dataset_type, num_clients, data_path, partition_method, noniid_param)

def get_data(args):
    """兼容原有接口的数据获取函数，支持数据保存和加载"""
    
    # 首先检查是否需要从文件加载数据
    if hasattr(args, 'load_data') and args.load_data:
        print(f"\n🔄 尝试从文件加载客户端数据: {args.load_data}")
        
        # 如果指定了相对路径，尝试在数据保存目录中查找
        if not os.path.isabs(args.load_data):
            save_dir = getattr(args, 'data_save_dir', './saved_data/')
            full_path = os.path.join(save_dir, args.load_data)
            if os.path.exists(full_path):
                args.load_data = full_path
        
        # 创建数据集对象（用于兼容性）
        dataset_train, dataset_test = create_dataset_objects(args)
        
        # 尝试加载数据
        dict_users, client_classes = load_client_data_distribution(args.load_data, args)
        
        if dict_users is not None and client_classes is not None:
            print("✅ 成功从文件加载客户端数据分配")
            return dataset_train, dataset_test, dict_users, client_classes
        else:
            print("❌ 从文件加载数据失败，将重新生成数据")
    
    # 如果指定了--load_data但没有提供具体路径，显示可用文件
    if hasattr(args, 'load_data') and args.load_data == '':
        save_dir = getattr(args, 'data_save_dir', './saved_data/')
        print_available_data_files(save_dir)
        exit("请指定要加载的数据文件路径")

    print("\n🔨 生成新的客户端数据分配...")
    
    # 创建数据集对象
    dataset_train, dataset_test = create_dataset_objects(args)
    
    # 确定数据集类型和路径
    if args.dataset == 'mnist':
        dataset_type = 'mnist'
        data_path = os.path.join(args.data_path, 'mnist/')
    elif args.dataset == 'cifar':
        dataset_type = 'cifar10'
        data_path = os.path.join(args.data_path, 'cifar/')
    elif args.dataset == 'cifar100':
        dataset_type = 'cifar100'
        data_path = os.path.join(args.data_path, 'cifar100/')
    else:
        exit('Error: unrecognized dataset')

    # 检查是否使用 sampling.py 中的数据划分方式
    use_sampling_partition = getattr(args, 'use_sampling', False)

    if use_sampling_partition:
        print("使用 sampling.py 中的数据划分方式")
        # 使用 sampling.py 中的数据划分方式
        if args.dataset == 'mnist':
            if hasattr(args, 'iid') and args.iid:
                dict_users = mnist_iid(dataset_train, args.num_users)
                print("使用 MNIST IID 数据划分")
            else:
                dict_users = mnist_noniid(dataset_train, args.num_users)
                print("使用 MNIST Non-IID 数据划分")
        elif args.dataset == 'cifar':
            if hasattr(args, 'iid') and args.iid:
                dict_users = cifar_iid(dataset_train, args.num_users)
                print("使用 CIFAR IID 数据划分")
            else:
                # 对于 CIFAR，如果没有 cifar_noniid 函数，使用修改版的 mnist_noniid
                print("警告: CIFAR 使用修改版的 Non-IID 划分")
                dict_users = cifar_noniid_adapted(dataset_train, args.num_users)
        elif args.dataset == 'cifar100':  # 新增
            if args.iid:
                dict_users = cifar100_iid(dataset_train, args.num_users)
                print("使用 CIFAR100 IID 数据划分")
            else:
                dict_users = cifar100_noniid_adapted(dataset_train, args.num_users)
                print("使用 CIFAR100 Non-IID 数据划分")


        # 计算客户端类别信息（用于FedRS）
        client_classes = get_client_classes_from_sampling(dataset_train, dict_users)

    else:
        # 使用原有的数据划分方法
        # 确定分区方法 - 优先使用新的partition参数
        if hasattr(args, 'partition'):
            partition_method = args.partition
            # 如果还设置了iid参数，覆盖partition设置
            if hasattr(args, 'iid') and args.iid:
                partition_method = 'homo'
        else:
            # 兼容旧版本参数
            if hasattr(args, 'iid') and args.iid:
                partition_method = 'homo'
            else:
                partition_method = 'noniid-labeldir'

        # 确定non-iid参数
        noniid_param = getattr(args, 'beta', 0.4)

        print(f"使用数据划分方法: {partition_method}, non-iid参数: {noniid_param}")

        # 使用新的数据划分方法获取客户端映射
        train_data, test_data, dict_users, client_classes = get_data_new(
            dataset_type, args.num_users, data_path, partition_method, noniid_param
        )
    
    # 检查是否需要保存数据
    if hasattr(args, 'save_data') and args.save_data:
        print("\n💾 保存客户端数据分配...")
        save_client_data_distribution(dict_users, client_classes, args)
    
    return dataset_train, dataset_test, dict_users, client_classes


def create_dataset_objects(args):
    """
    创建数据集对象
    
    Args:
        args: 命令行参数对象
    
    Returns:
        tuple: (dataset_train, dataset_test)
    """
    if args.dataset == 'mnist':
        data_path = os.path.join(args.data_path, 'mnist/')
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST(data_path, train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST(data_path, train=False, download=True, transform=trans_mnist)

    elif args.dataset == 'cifar':
        data_path = os.path.join(args.data_path, 'cifar/')
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10(data_path, train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10(data_path, train=False, download=True, transform=trans_cifar)
    
    elif args.dataset == 'cifar100':

        data_path = os.path.join(args.data_path, 'cifar100/')
        trans_cifar100 = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR100(data_path, train=True, download=True, transform=trans_cifar100)
        dataset_test = datasets.CIFAR100(data_path, train=False, download=True, transform=trans_cifar100)

    else:
        exit('Error: unrecognized dataset')
    
    return dataset_train, dataset_test


def get_data_test(args):
    """
    获取训练和测试数据，支持两种固定的Non-IID数据分布类型
    
    Args:
        args: 参数对象，包含数据集、用户数量等配置
        
    Returns:
        tuple: (dataset_train, dataset_test, dict_users, client_classes)
    """
    import random
    import numpy as np
    
    print(f"\n🧪 使用测试版数据分配 (get_data_test)")
    print(f"📊 Non-IID模式：仅使用两种固定的数据分布类型")
    
    # 首先检查是否需要从文件加载数据
    if hasattr(args, 'load_data') and args.load_data:
        print(f"\n🔄 尝试从文件加载客户端数据: {args.load_data}")
        
        # 如果指定了相对路径，尝试在数据保存目录中查找
        if not os.path.isabs(args.load_data):
            save_dir = getattr(args, 'data_save_dir', './saved_data/')
            full_path = os.path.join(save_dir, args.load_data)
            if os.path.exists(full_path):
                args.load_data = full_path
        
        # 创建数据集对象（用于兼容性）
        dataset_train, dataset_test = create_dataset_objects(args)
        
        # 尝试加载数据
        dict_users, client_classes = load_client_data_distribution(args.load_data, args)
        
        if dict_users is not None and client_classes is not None:
            print("✅ 成功从文件加载客户端数据分配")
            return dataset_train, dataset_test, dict_users, client_classes
        else:
            print("❌ 从文件加载数据失败，将重新生成数据")
    
    print("\n🔨 生成新的测试版客户端数据分配...")
    
    # 创建数据集对象
    dataset_train, dataset_test = create_dataset_objects(args)
    
    # 确定数据集类型
    if args.dataset == 'mnist':
        num_classes = 10
        class_names = list(range(10))
    elif args.dataset == 'cifar':
        num_classes = 10  
        class_names = list(range(10))
    elif args.dataset == 'cifar100':
        num_classes = 100
        class_names = list(range(100))
    else:
        raise ValueError(f'Error: unrecognized dataset {args.dataset}')
    
    # 定义两种固定的数据分布类型
    if args.iid:
        print("🎯 IID模式：所有客户端使用相同的IID分布")
        # IID模式下，所有客户端都使用相同的分布
        if args.dataset == 'mnist':
            dict_users = mnist_iid(dataset_train, args.num_users)
        elif args.dataset == 'cifar':
            dict_users = cifar_iid(dataset_train, args.num_users)
        elif args.dataset == 'cifar100':
            dict_users = cifar100_iid(dataset_train, args.num_users)
        
        # 计算客户端类别信息
        client_classes = get_client_classes_from_sampling(dataset_train, dict_users)
        
    else:
        print("🎯 Non-IID模式：使用两种固定的数据分布类型")
        
        # 定义两种不同的Non-IID分布类型
        if num_classes >= 6:
            # 类型A：偏向前半部分类别 (0, 1, 2, ...)
            type_A_classes = class_names[:num_classes//2]
            # 类型B：偏向后半部分类别 (..., 7, 8, 9)
            type_B_classes = class_names[num_classes//2:]
        else:
            # 如果类别太少，交替分配
            type_A_classes = [class_names[i] for i in range(0, num_classes, 2)]  # 偶数索引
            type_B_classes = [class_names[i] for i in range(1, num_classes, 2)]  # 奇数索引
        
        print(f"📋 类型A分布主要类别: {type_A_classes}")
        print(f"📋 类型B分布主要类别: {type_B_classes}")
        
        # 为每个客户端随机分配分布类型
        distribution_types = []
        for i in range(args.num_users):
            dist_type = random.choice(['A', 'B'])
            distribution_types.append(dist_type)
        
        type_A_count = distribution_types.count('A')
        type_B_count = distribution_types.count('B')
        print(f"📊 分布类型分配：类型A {type_A_count}个客户端，类型B {type_B_count}个客户端")
        
        # 生成每种类型的数据分配
        dict_users = {}
        client_classes = {}
        
        # 获取每个类别的样本索引
        labels = np.array(dataset_train.targets)
        class_indices = {}
        for class_id in range(num_classes):
            class_indices[class_id] = np.where(labels == class_id)[0]
        
        # 为每个客户端分配数据
        samples_per_client = len(dataset_train) // args.num_users
        
        for client_id in range(args.num_users):
            dist_type = distribution_types[client_id]
            
            if dist_type == 'A':
                # 类型A：80%来自type_A_classes，20%来自type_B_classes
                main_classes = type_A_classes
                minor_classes = type_B_classes
            else:
                # 类型B：80%来自type_B_classes，20%来自type_A_classes
                main_classes = type_B_classes  
                minor_classes = type_A_classes
            
            # 分配样本
            client_indices = []
            
            # 80%来自主要类别
            main_samples = int(samples_per_client * 0.8)
            main_samples_per_class = main_samples // len(main_classes)
            
            for class_id in main_classes:
                available_indices = class_indices[class_id]
                if len(available_indices) >= main_samples_per_class:
                    selected = np.random.choice(available_indices, main_samples_per_class, replace=False)
                else:
                    selected = np.random.choice(available_indices, main_samples_per_class, replace=True)
                client_indices.extend(selected)
            
            # 20%来自次要类别
            minor_samples = samples_per_client - len(client_indices)
            if minor_samples > 0 and len(minor_classes) > 0:
                minor_samples_per_class = minor_samples // len(minor_classes)
                for class_id in minor_classes:
                    available_indices = class_indices[class_id]
                    if len(available_indices) >= minor_samples_per_class:
                        selected = np.random.choice(available_indices, minor_samples_per_class, replace=False)
                    else:
                        selected = np.random.choice(available_indices, minor_samples_per_class, replace=True)
                    client_indices.extend(selected)
            
            # 如果还差一些样本，随机补充
            while len(client_indices) < samples_per_client:
                remaining_samples = samples_per_client - len(client_indices)
                all_available = np.concatenate([class_indices[c] for c in (main_classes + minor_classes)])
                additional = np.random.choice(all_available, min(remaining_samples, len(all_available)), replace=False)
                client_indices.extend(additional)
            
            # 确保不超过目标数量
            client_indices = client_indices[:samples_per_client]
            
            dict_users[client_id] = set(client_indices)
            
            # 计算此客户端的类别分布
            client_labels = labels[client_indices]
            unique_classes = np.unique(client_labels).tolist()
            client_classes[client_id] = unique_classes
        
        # 打印分布统计
        print(f"\n📈 数据分布统计:")
        for client_id in range(min(5, args.num_users)):  # 只显示前5个客户端
            dist_type = distribution_types[client_id]
            classes = client_classes[client_id]
            print(f"  客户端{client_id} (类型{dist_type}): {len(classes)}个类别 {classes}")
        
        if args.num_users > 5:
            print(f"  ... 以及其他 {args.num_users - 5} 个客户端")
    
    # 保存数据分配到文件
    if hasattr(args, 'save_data') and args.save_data:
        save_client_data_distribution(dict_users, client_classes, args)
    
    print("✅ 测试版数据分配生成完成")
    return dataset_train, dataset_test, dict_users, client_classes


if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
