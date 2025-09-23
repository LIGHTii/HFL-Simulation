#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import os

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
    dict_users, all_idxs = {}, [i for i in range(len(dataset))] ##dict_users:每个用户的样本索引；all_idxs:包含所有数据索引的列表
    for i in range(num_users):  ##给每个用户不重复地分配数据的索引
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300 ##num_shards：将数据划分为的片段数量；num_imgs:每个片段的图像数量
    idx_shard = [i for i in range(num_shards)]  ##每个片段的索引
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}  ##每个用户对应的样本索引
    idxs = np.arange(num_shards*num_imgs) ##创建一个包含所有图像索引的数组，就是一个0-的有序数组
    labels = dataset.train_labels.numpy() ##将数据集中样本的标签转换为numpy

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()] ##根据标签对图像进行排序 ，返回为有序数列的原位置的序号例如，如果标签的原始顺序是 [5, 0, 4, 1, 2]，使用 argsort() 后会返回 [1, 4, 3, 2, 0]，表示标签1排在最前，标签5排在最后。
    idxs = idxs_labels[0,:] ##idxs排序后的索引，相当于标签排序得到有序，对应索引跟随改变，得到新位置
       ##其将相同类别的图像集中在一起

    # divide and assign 将数据分配给每个用户
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))##随机选择两个片段且不重复
        idx_shard = list(set(idx_shard) - rand_set) ##移除已被选择的片段
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
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):  ##给每个用户不重复地分配数据的索引
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

# new
def cifar_noniid_adapted(dataset, num_users):
    """
    CIFAR数据集的Non-IID划分，基于MNIST Non-IID方法适配
    将10个类别划分为200个分片，每个客户端分配2个分片
    """
    num_shards, num_imgs = 200, 250  # CIFAR每个分片250张图片
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = np.array(dataset.targets)

    # 按标签排序
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # 为每个客户端随机分配2个分片
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
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

    else:
        exit('Error: unrecognized dataset')
    
    return dataset_train, dataset_test


if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
