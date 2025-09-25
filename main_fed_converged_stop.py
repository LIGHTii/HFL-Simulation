#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
import matplotlib

from utils.visualization_tool import create_enhanced_visualizations

matplotlib.use('Agg')
# 设置 matplotlib 使用英文字体，避免中文字体警告
matplotlib.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.unicode_minus': True  # 正确显示负号
})
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from tqdm import tqdm
import torch.multiprocessing as mp
import os
os.environ["OMP_NUM_THREADS"] = "1"

import csv
from datetime import datetime
import pandas as pd

from utils.sampling import get_data
from utils.options import args_parser
from utils.data_partition import get_client_datasets
from utils.visualize_client_data import visualize_client_data_distribution
from utils.eh_test_utils import EHTestsetGenerator, test_eh_model
from utils.bipartite_bandwidth import run_bandwidth_allocation, calculate_distance
from utils.comm_utils import calculate_transmission_time, get_model_size_in_bits, select_eh
from models.Nets import MLP, CNNMnist, CNNCifar, LR, ResNet18, VGG11, MobileNetCifar, LeNet5
from models.Fed import FedAvg, FedAvg_layered
from models.test import test_img
from models.Update import LocalUpdate
from models.ES_cluster import (
    train_initial_models,
    aggregate_es_models, spectral_clustering_es,
    calculate_es_label_distributions,
    visualize_clustering_comparison
)
from utils.conver_check import ConvergenceChecker
import numpy as np

def build_model(args, dataset_train):
    img_size = dataset_train[0][0].shape

    if args.model == 'cnn':
        if args.dataset in ['cifar', 'cifar100']:  # 支持 cifar 和 cifar100
            net_glob = CNNCifar(args=args).to(args.device)  # CNNCifar 需要支持 args.num_classes=100
        elif args.dataset == 'mnist':
            net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        # 计算将图片展平后的输入层维度
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)

    # ======================= 新 =======================
    elif args.model == 'lr' and args.dataset == 'mnist':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = LR(dim_in=len_in, dim_out=args.num_classes).to(args.device)

    elif args.model == 'lenet5' and args.dataset == 'mnist':
        net_glob = LeNet5(args=args).to(args.device)

    elif args.model == 'vgg11' and args.dataset in ['cifar', 'cifar100']:
        net_glob = VGG11(args=args).to(args.device)

    elif args.model == 'resnet18' and args.dataset in ['cifar', 'cifar100']:
        net_glob = ResNet18(args=args).to(args.device)  # ResNet18 需要支持 args.num_classes=100

    else:
        exit('错误：无法识别的模型')

    # print("--- 模型架构 ---")
    # print(net_glob)
    # print("--------------------")
    return net_glob

def get_A_random(num_users, num_ESs):
    A = np.zeros((num_users, num_ESs), dtype=int)

    # 每个 ES 至少要分到的用户数
    base = num_users // num_ESs
    # 多出来的用户数量
    extra = num_users % num_ESs

    # 用户索引
    users = np.arange(num_users)
    np.random.shuffle(users)  # 打乱顺序，保证随机性

    start = 0
    for es in range(num_ESs):
        count = base + (1 if es < extra else 0)
        assigned_users = users[start:start+count]
        for u in assigned_users:
            A[u, es] = 1
        start += count

    return A

def get_B(num_ESs, num_EHs):
    B = np.zeros((num_ESs, num_EHs), dtype=int)

    # 对每一行随机选择一个索引，将该位置设为 1
    for i in range(num_ESs):
        random_index = np.random.randint(0, num_EHs)
        B[i, random_index] = 1

    return B

def get_B_cluster(args, w_locals, A, dict_users, net_glob, client_label_distributions):
    """
    使用谱聚类生成 ES-EH 关联矩阵 B，并可视化聚类结果
    """
    print("开始谱聚类生成B矩阵...")

    # 1. 聚合ES模型
    es_models = aggregate_es_models(w_locals, A, dict_users, net_glob)

    # 2. 使用谱聚类获取ES-EH关联矩阵B
    B, cluster_labels = spectral_clustering_es(
        es_models,
        epsilon=args.epsilon  # 从参数中获取
    )

    # 3. 计算ES的标签分布并可视化
    es_label_distributions = calculate_es_label_distributions(A, client_label_distributions)

    #labels1, labels2, labels3 = run_all_clusterings(es_models, epsilon=args.epsilon)
    # 在完成谱聚类后添加对比可视化
    visualize_clustering_comparison(
        es_label_distributions=es_label_distributions,
        cluster_labels=cluster_labels,
        save_path='./save/clustering_comparison.png'
    )
    return B

# ===== 根据 A、B 构造 C1 和 C2 =====
def build_hierarchy(A, B):
    num_users, num_ESs = A.shape
    _, num_EHs = B.shape

    # client -> ES
    C1 = {j: [] for j in range(num_ESs)}
    for i in range(num_users):
        for j in range(num_ESs):
            if A[i][j] == 1:
                C1[j].append(i)

    # ES -> EH
    C2 = {k: [] for k in range(num_EHs)}
    for j in range(num_ESs):
        for k in range(num_EHs):
            if B[j][k] == 1:
                C2[k].append(j)

    return C1, C2

def train_client(args, user_idx, dataset_train, dict_users, w_input_hfl_random, w_input_hfl_cluster, w_sfl_global, 
                 client_classes=None, train_hfl_random=True, train_hfl_cluster=True, train_sfl=True):
    """
    单个客户端的训练函数，用于被多进程调用。
    现在支持三种模型：SFL、HFL(随机B矩阵)、HFL(聚类B矩阵)

    注意：为了兼容多进程，我们不直接传递大型模型对象，
    而是传递模型权重(state_dict)和模型架构信息(args)，在子进程中重新构建模型。
    
    Args:
        train_hfl_random: 是否训练HFL随机B矩阵模型
        train_hfl_cluster: 是否训练HFL聚类B矩阵模型
        train_sfl: 是否训练SFL模型
    """
    # 在子进程中重新构建模型
    local_net_hfl_random = build_model(args, dataset_train)
    local_net_hfl_cluster = build_model(args, dataset_train)
    local_net_sfl = build_model(args, dataset_train)
    
    # 获取当前客户端的类别信息
    user_classes = client_classes.get(user_idx, None) if client_classes else None
    
    # --- 训练HFL模型 (使用随机B矩阵) - 仅在未收敛时训练 ---
    if train_hfl_random:
        local_random = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[user_idx], user_classes=user_classes)
        local_net_hfl_random.load_state_dict(w_input_hfl_random)
        w_hfl_random, loss_hfl_random = local_random.train(net=local_net_hfl_random.to(args.device))
    else:
        # 如果已收敛，直接返回输入权重和零损失
        w_hfl_random, loss_hfl_random = copy.deepcopy(w_input_hfl_random), 0.0
    
    # --- 训练HFL模型 (使用聚类B矩阵) - 仅在未收敛时训练 ---
    if train_hfl_cluster:
        local_cluster = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[user_idx], user_classes=user_classes)
        local_net_hfl_cluster.load_state_dict(w_input_hfl_cluster)
        w_hfl_cluster, loss_hfl_cluster = local_cluster.train(net=local_net_hfl_cluster.to(args.device))
    else:
        # 如果已收敛，直接返回输入权重和零损失
        w_hfl_cluster, loss_hfl_cluster = copy.deepcopy(w_input_hfl_cluster), 0.0
    
    # --- 训练单层模型 (SFL) - 仅在未收敛时训练 ---
    if train_sfl:
        local_sfl = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[user_idx], user_classes=user_classes)
        local_net_sfl.load_state_dict(w_sfl_global)
        w_sfl, loss_sfl = local_sfl.train(net=local_net_sfl.to(args.device))
    else:
        # 如果已收敛，直接返回输入权重和零损失
        w_sfl, loss_sfl = copy.deepcopy(w_sfl_global), 0.0

    # 返回结果，包括 user_idx 以便后续排序
    return (user_idx, 
            copy.deepcopy(w_hfl_random), loss_hfl_random,
            copy.deepcopy(w_hfl_cluster), loss_hfl_cluster, 
            copy.deepcopy(w_sfl), loss_sfl)

def summarize_results(net_glob_hfl_bipartite, net_glob_hfl_random, net_glob_sfl, dataset_train, dataset_test, args,
                     total_comm_overhead_bipartite_upload, total_comm_overhead_bipartite_download,
                     total_comm_overhead_random_upload, total_comm_overhead_random_download,
                     total_comm_overhead_sfl_upload, total_comm_overhead_sfl_download):
    print("=== Final Communication Overhead Summary ===")
    print(f"SFL Total Overhead: {total_comm_overhead_sfl_upload + total_comm_overhead_sfl_download:.6f}s "
          f"(Upload: {total_comm_overhead_sfl_upload:.6f}s, Download: {total_comm_overhead_sfl_download:.6f}s)")
    print(f"HFL Bipartite Total Overhead: {total_comm_overhead_bipartite_upload + total_comm_overhead_bipartite_download:.6f}s "
          f"(Upload: {total_comm_overhead_bipartite_upload:.6f}s, Download: {total_comm_overhead_bipartite_download:.6f}s)")
    print(f"HFL Random Total Overhead: {total_comm_overhead_random_upload + total_comm_overhead_random_download:.6f}s "
          f"(Upload: {total_comm_overhead_random_upload:.6f}s, Download: {total_comm_overhead_random_download:.6f}s)")

def save_results_to_csv(results, filename):
    """Save results to CSV file for three models, including EH-level testing results"""
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['epoch', 'eh_round', 'es_round', 'train_loss', 'test_loss', 'test_acc', 
                     'model_type', 'level', 'eh_idx']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # 先运行带宽分配算法获取实际的客户端数量
    print("正在分析网络拓扑并确定客户端数量...")
    bipartite_graph, client_nodes, active_es_nodes, A_design, r_client_to_es, r_es, r_es_to_cloud, r_client_to_cloud = run_bandwidth_allocation(
        graphml_file=args.graphml_file, 
        es_ratio=args.es_ratio, 
        max_capacity=args.max_capacity, 
        visualize=True)
    
    if bipartite_graph is None:
        print("Failed to build bipartite graph, exiting.")
        exit(1)
    
    # 根据实际客户端数量更新args.num_users
    actual_num_users = len(client_nodes)
    print(f"网络拓扑分析完成：实际客户端数量为 {actual_num_users}")
    print(f"原始参数设置：args.num_users = {args.num_users}")
    
    # 更新参数以匹配实际客户端数量
    args.num_users = actual_num_users
    print(f"已更新参数：args.num_users = {args.num_users}")

    # 现在使用更新后的参数生成数据分配
    dataset_train, dataset_test, dict_users, client_classes = get_data(args)

    # 打印 FedRS 配置信息
    if args.method == 'fedrs':
        print("\n" + "="*50)
        print("FedRS 算法配置信息")
        print("="*50)
        print(f"联邦学习方法: {args.method}")
        print(f"FedRS Alpha 参数: {args.fedrs_alpha}")
        print(f"最小本地训练轮次: {args.min_le}")
        print(f"最大本地训练轮次: {args.max_le}")
        
        # 统计客户端类别分布
        class_counts = {}
        for client_id, classes in client_classes.items():
            num_classes = len(classes)
            class_counts[num_classes] = class_counts.get(num_classes, 0) + 1
        
        print("\n客户端类别分布统计:")
        for num_classes, count in sorted(class_counts.items()):
            print(f"  拥有 {num_classes} 个类别的客户端数量: {count}")
        print("="*50 + "\n")
    else:
        print(f"\n使用联邦学习方法: {args.method}\n")

    net_glob = build_model(args, dataset_train)
    # 验证数据一致性
    print(f"\n=== 数据一致性验证 ===")
    print(f"客户端节点总数: {len(client_nodes)}")
    print(f"活跃边缘服务器节点总数: {len(active_es_nodes)}")
    print(f"args.num_users: {args.num_users}")
    print(f"dict_users键的数量: {len(dict_users)}")
    print(f"dict_users键的范围: {min(dict_users.keys())} - {max(dict_users.keys())}")
    print("=" * 25)

    net_glob.train()

    # 初始化全局权重
    w_glob = net_glob.state_dict()
    num_users = len(client_nodes)
    num_ESs = len(active_es_nodes)
    k2 = args.ES_k2
    k3 = args.EH_k3
    num_processes = args.num_processes

    A_random = get_A_random(num_users, num_ESs)

    # 使用谱聚类生成B矩阵（替换原来的随机B矩阵）
    print("开始初始训练和谱聚类...")

    # 1. 训练初始本地模型
    w_locals, client_label_distributions = train_initial_models(
        args, dataset_train, dict_users, net_glob, num_users
    )

    # 2. 使用谱聚类生成B矩阵
    B_cluster = get_B_cluster(
        args, w_locals, A_design, dict_users, net_glob, client_label_distributions
    )
    num_EHs = B_cluster.shape[1]
    
    # 3. 同时生成随机B矩阵用于对比
    B_random = get_B(num_ESs, num_EHs)

    # 构建两套层级结构（用于联邦学习聚合）
    C1_random, C2_random = build_hierarchy(A_random, B_random)
    C1_cluster, C2_cluster = build_hierarchy(A_design, B_cluster)

    # 构建通信实际的关联矩阵（用于通信开销计算）
    # 两种A关联矩阵直接可用，B关联矩阵为（es，簇）形式，需转化为es-es，还需生成es-cloud
    model_size = get_model_size_in_bits(w_glob)
    B_random_comm, C_random_comm = select_eh(B_random, r_es, r_es_to_cloud, model_size)
    B_cluster_comm, C_cluster_comm = select_eh(B_cluster, r_es, r_es_to_cloud, model_size)
    print("C1_random (一级->客户端):", C1_random)
    print("C2_random (二级->一级):", C2_random)
    print("C1_cluster (一级->客户端):", C1_cluster)
    print("C2_cluster (二级->一级):", C2_cluster)
    t_client_to_es_random = calculate_transmission_time(model_size, r_client_to_es, A_random)
    t_client_to_es_design = calculate_transmission_time(model_size, r_client_to_es, A_design)
    t_es_to_eh_random = calculate_transmission_time(model_size, r_es, B_random_comm)
    t_es_to_eh_design = calculate_transmission_time(model_size, r_es, B_cluster_comm)
    t_eh_to_cloud_random = calculate_transmission_time(model_size, r_es_to_cloud, C_random_comm)
    t_eh_to_cloud_design = calculate_transmission_time(model_size, r_es_to_cloud, C_cluster_comm)
    t_client_to_cloud_sfl = calculate_transmission_time(model_size, r_client_to_cloud, np.ones((num_users, 1), dtype=int))
    print(f"random:{t_client_to_es_random}, {t_es_to_eh_random}, {t_eh_to_cloud_random}")
    print(f"design:{t_client_to_es_design}, {t_es_to_eh_design}, {t_eh_to_cloud_design}")
    print(f"sfl:{t_client_to_cloud_sfl}")
    t_hfl_random_sig = t_client_to_es_random * k2 + t_es_to_eh_random * k3 + t_eh_to_cloud_random
    t_hfl_design_sig = t_client_to_es_design * k2 + t_es_to_eh_design * k3 + t_eh_to_cloud_design
    t_sfl_sig = t_client_to_cloud_sfl * k2 * k3  # SFL 直接通信到云端，乘以 k2*k3 次
    print(f"hfl_random 预计单轮通信时间: {t_hfl_random_sig:.6f}s")
    print(f"hfl_design 预计单轮通信时间: {t_hfl_design_sig:.6f}s")
    print(f"sfl 预计单轮通信时间: {t_sfl_sig:.6f}s")
    # 生成EH专属测试集
    print("\n--- 生成EH专属测试集 ---")
    print("采用改进的资源分配策略：允许测试样本在多个EH测试集中重复出现")
    print("这确保每个EH都能获得与其下游客户端分布匹配的个性化测试集")
    
    # 为随机B矩阵生成EH专属测试集
    print("\n🎲 为随机B矩阵生成EH专属测试集...")
    eh_testsets_random, eh_label_distributions_random = EHTestsetGenerator.create_eh_testsets(
        dataset_test, A_random, B_random, C1_random, C2_random, dataset_train, dict_users, visualize=True
    )
    
    # 为聚类B矩阵生成EH专属测试集
    print("\n🧩 为聚类B矩阵生成EH专属测试集...")
    eh_testsets_cluster, eh_label_distributions_cluster = EHTestsetGenerator.create_eh_testsets(
        dataset_test, A_design, B_cluster, C1_cluster, C2_cluster, dataset_train, dict_users, visualize=True
    )
    
    print(f"\n✅ 测试集生成完成!")
    print(f"随机B矩阵: 已生成 {len(eh_testsets_random)} 个EH专属测试集")
    print(f"聚类B矩阵: 已生成 {len(eh_testsets_cluster)} 个EH专属测试集")
    
    # 打印每个EH测试集的详细信息
    print(f"\n📊 随机B矩阵 - EH测试集统计:")
    for eh_idx, testset in eh_testsets_random.items():
        unique_samples = len(np.unique(testset))
        total_samples = len(testset)
        print(f"  EH {eh_idx}: 总样本={total_samples}, 唯一样本={unique_samples}, 重复率={1-unique_samples/total_samples:.1%}")
    
    print(f"\n📊 聚类B矩阵 - EH测试集统计:")
    for eh_idx, testset in eh_testsets_cluster.items():
        unique_samples = len(np.unique(testset))
        total_samples = len(testset)
        print(f"  EH {eh_idx}: 总样本={total_samples}, 唯一样本={unique_samples}, 重复率={1-unique_samples/total_samples:.1%}")

    # 打印FedRS配置信息
    print(f"\n--- FedRS Configuration ---")
    print(f"Method: {args.method}")
    if args.method == 'fedrs':
        print(f"FedRS Alpha: {args.fedrs_alpha}")
        print(f"Min Local Epochs: {args.min_le}")
        print(f"Max Local Epochs: {args.max_le}")
        print("FedRS Restricted Softmax: Enabled")
        # 打印前几个客户端的类别信息
        print("Sample Client Class Distributions:")
        for i in range(min(5, len(client_classes))):
            print(f"  Client {i}: Classes {client_classes[i]}")
    else:
        print("FedRS: Disabled (using standard FedAvg)")
    print("-----------------------------\n")

    # 创建结果保存目录
    if not os.path.exists('./results'):
        os.makedirs('./results')

    # 生成唯一的时间戳用于文件名，包含重要参数
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    param_str = f"e{args.epochs}_u{args.num_users}_le{args.local_ep}_{args.dataset}_{args.model}_k2{args.ES_k2}_k3{args.EH_k3}_p{args.num_processes}"
    if not args.iid:
        param_str += f"_beta{args.beta}"
    csv_filename = f'./results/training_results_{param_str}_{timestamp}.csv'

    # 初始化结果记录列表 - 新格式支持三种模型
    results_history = []

    # 训练前测试初始模型
    print("\n--- Testing Initial Global Models ---")
    net_glob.eval()

    # 测试初始模型
    acc_init, loss_init = test_img(net_glob, dataset_test, args)
    print(f"Initial Model - Testing accuracy: {acc_init:.2f}%, Loss: {loss_init:.4f}")

    # 记录初始结果 - 三种模型使用相同的初始权重
    for model_name in ['HFL_Random_B', 'HFL_Cluster_B', 'SFL']:
        results_history.append({
            'epoch': -1,
            'eh_round': 0,
            'es_round': 0,
            'train_loss': 0.0,  # 初始训练损失暂时设为0
            'test_loss': loss_init,
            'test_acc': acc_init,
            'model_type': model_name,
            'level': 'Global',
            'eh_idx': -1
        })

    # 保存初始结果到CSV
    save_results_to_csv(results_history, csv_filename)

    # training
    # --- 初始化三个模型，确保初始权重相同 ---
    # net_glob_hfl_random 是使用随机B矩阵的 HFL 全局模型
    net_glob_hfl_random = copy.deepcopy(net_glob)
    w_glob_hfl_random = net_glob_hfl_random.state_dict()

    # net_glob_hfl_cluster 是使用聚类B矩阵的 HFL 全局模型
    net_glob_hfl_cluster = copy.deepcopy(net_glob)
    w_glob_hfl_cluster = net_glob_hfl_cluster.state_dict()

    # net_glob_sfl 是 SFL 的全局模型
    net_glob_sfl = copy.deepcopy(net_glob)
    w_glob_sfl = net_glob_sfl.state_dict()

    # --- 分别记录三种模型的指标 ---
    loss_train_hfl_random = []
    loss_train_hfl_cluster = []
    loss_train_sfl = []
    loss_test_hfl_random = []
    loss_test_hfl_cluster = []
    loss_test_sfl = []
    acc_test_hfl_random = []
    acc_test_hfl_cluster = []
    acc_test_sfl = []
    t_hfl_random = 0
    t_hfl_design = 0
    t_sfl = 0

    # 记录实际运行的epoch数
    final_epoch = args.epochs

    # --- 初始化收敛检查器 ---
    print("\n=== 初始化收敛检查器 ===")
    # 为每个EH创建收敛检查器 - HFL随机B矩阵
    eh_checkers_random = {}
    for eh_idx in range(num_EHs):
        eh_checkers_random[eh_idx] = ConvergenceChecker(patience = args.EH_k3)
    
    # 为每个EH创建收敛检查器 - HFL聚类B矩阵  
    eh_checkers_cluster = {}
    for eh_idx in range(num_EHs):
        eh_checkers_cluster[eh_idx] = ConvergenceChecker(patience = args.EH_k3)
    
    # 为SFL创建收敛检查器
    sfl_checker = ConvergenceChecker(patience = args.EH_k3)

    # 记录各机制的收敛状态
    converged_hfl_random = False
    converged_hfl_cluster = False
    converged_sfl = False
    
    print(f"已为HFL随机B矩阵创建 {len(eh_checkers_random)} 个EH收敛检查器")
    print(f"已为HFL聚类B矩阵创建 {len(eh_checkers_cluster)} 个EH收敛检查器")
    print(f"已为SFL创建全局收敛检查器")
    print("=" * 30)

    for epoch in range(args.epochs):
        # HFL 随机B矩阵模型权重分发 (Cloud -> EH)
        EHs_ws_hfl_random = [copy.deepcopy(w_glob_hfl_random) for _ in range(num_EHs)]
        
        # HFL 聚类B矩阵模型权重分发 (Cloud -> EH)
        EHs_ws_hfl_cluster = [copy.deepcopy(w_glob_hfl_cluster) for _ in range(num_EHs)]

        # EH 层聚合 k3 轮
        for t3 in range(k3):
            # HFL随机: EH 层 -> ES 层
            ESs_ws_input_hfl_random = [None] * num_ESs
            for EH_idx, ES_indices in C2_random.items():
                for ES_idx in ES_indices:
                    ESs_ws_input_hfl_random[ES_idx] = copy.deepcopy(EHs_ws_hfl_random[EH_idx])
            
            # HFL聚类: EH 层 -> ES 层
            ESs_ws_input_hfl_cluster = [None] * num_ESs
            for EH_idx, ES_indices in C2_cluster.items():
                for ES_idx in ES_indices:
                    ESs_ws_input_hfl_cluster[ES_idx] = copy.deepcopy(EHs_ws_hfl_cluster[EH_idx])

            # ES 层聚合 k2 轮
            for t2 in range(k2):
                # --- 本地训练 (三种模型并行) ---
                # HFL随机: ES 层 -> Client 层
                w_locals_input_hfl_random = [None] * num_users
                for ES_idx, user_indices in C1_random.items():
                    for user_idx in user_indices:
                        w_locals_input_hfl_random[user_idx] = copy.deepcopy(ESs_ws_input_hfl_random[ES_idx])
                
                # HFL聚类: ES 层 -> Client 层
                w_locals_input_hfl_cluster = [None] * num_users
                for ES_idx, user_indices in C1_cluster.items():
                    for user_idx in user_indices:
                        w_locals_input_hfl_cluster[user_idx] = copy.deepcopy(ESs_ws_input_hfl_cluster[ES_idx])

                # 用于存储三种模型本地训练的输出
                w_locals_output_hfl_random = [None] * num_users
                w_locals_output_hfl_cluster = [None] * num_users
                w_locals_output_sfl = [None] * num_users
                loss_locals_hfl_random = []
                loss_locals_hfl_cluster = []
                loss_locals_sfl = []

                # 显示训练状态信息
                active_models = []
                if not converged_hfl_random:
                    active_models.append("HFL_Random")
                if not converged_hfl_cluster:
                    active_models.append("HFL_Cluster")
                if not converged_sfl:
                    active_models.append("SFL")
                
                if not active_models:
                    print(f"\n[Skip Training] 所有模型已收敛，跳过客户端训练")
                    # 如果所有模型都收敛了，跳过训练但仍需要返回结果
                    results = []
                    for user_idx in range(num_users):
                        results.append((user_idx, 
                                      w_locals_input_hfl_random[user_idx], 0.0,
                                      w_locals_input_hfl_cluster[user_idx], 0.0,
                                      w_glob_sfl, 0.0))
                else:
                    print(f"\n[Parallel Training] 为 {len(active_models)} 种活跃模型训练 {args.num_users} 个客户端")
                    print(f"活跃模型: {', '.join(active_models)}")
                    print(f"使用 {num_processes} 个进程并行训练...")

                    # 准备传递给每个子进程的参数
                    tasks = []
                    for user_idx in range(num_users):
                        task_args = (
                            args, user_idx, dataset_train, dict_users,
                            w_locals_input_hfl_random[user_idx], w_locals_input_hfl_cluster[user_idx], 
                            w_glob_sfl, client_classes,
                            not converged_hfl_random,  # train_hfl_random
                            not converged_hfl_cluster,  # train_hfl_cluster  
                            not converged_sfl  # train_sfl
                        )
                        tasks.append(task_args)
                    print("成功创建多线程！")

                    # 创建进程池并分发任务
                    # 使用 with 语句可以自动管理进程池的关闭
                    with mp.Pool(processes=num_processes) as pool:
                        results = pool.starmap(train_client, tqdm(tasks, desc=f"Epoch {epoch}|{t3 + 1}|{t2 + 1} Training Clients"))

                print("训练结束")
                # 收集并整理所有客户端的训练结果
                for result in results:
                    u_idx, w_hr, l_hr, w_hc, l_hc, w_s, l_s = result
                    w_locals_output_hfl_random[u_idx] = w_hr
                    loss_locals_hfl_random.append(l_hr)
                    w_locals_output_hfl_cluster[u_idx] = w_hc
                    loss_locals_hfl_cluster.append(l_hc)
                    w_locals_output_sfl[u_idx] = w_s
                    loss_locals_sfl.append(l_s)
                
                print("排序结束")
                if active_models:
                    print(f"[Parallel Training] 所有 {args.num_users} 个客户端已完成 {len(active_models)} 种模型的训练")
                    print(f"训练的模型: {', '.join(active_models)}")
                else:
                    print(f"[Skip Training] 所有模型已收敛，未进行实际训练")
                print(f'\nEpoch {epoch} | EH_R {t3 + 1}/{k3} | ES_R {t2 + 1}/{k2} | 开始聚合')

                # --- HFL 聚合 (Client -> ES) - 只对未收敛的机制进行聚合 ---
                if not converged_hfl_random:
                    ESs_ws_input_hfl_random = FedAvg_layered(w_locals_output_hfl_random, C1_random)
                    t_hfl_random += t_client_to_es_random
                else:
                    print(f"  [Skip] HFL随机B矩阵已收敛，跳过ES层聚合")
                
                if not converged_hfl_cluster:
                    ESs_ws_input_hfl_cluster = FedAvg_layered(w_locals_output_hfl_cluster, C1_cluster)
                    t_hfl_design += t_client_to_es_design
                else:
                    print(f"  [Skip] HFL聚类B矩阵已收敛，跳过ES层聚合")

                # --- 记录损失 ---
                # 只为实际训练的模型计算平均损失，已收敛的模型损失为0
                if not converged_hfl_random:
                    loss_avg_hfl_random = sum(loss_locals_hfl_random) / len(loss_locals_hfl_random) if loss_locals_hfl_random else 0.0
                else:
                    loss_avg_hfl_random = 0.0  # 已收敛，损失为0
                    
                if not converged_hfl_cluster:
                    loss_avg_hfl_cluster = sum(loss_locals_hfl_cluster) / len(loss_locals_hfl_cluster) if loss_locals_hfl_cluster else 0.0
                else:
                    loss_avg_hfl_cluster = 0.0  # 已收敛，损失为0
                    
                if not converged_sfl:
                    loss_avg_sfl = sum(loss_locals_sfl) / len(loss_locals_sfl) if loss_locals_sfl else 0.0
                else:
                    loss_avg_sfl = 0.0  # 已收敛，损失为0
                
                loss_train_hfl_random.append(loss_avg_hfl_random)
                loss_train_hfl_cluster.append(loss_avg_hfl_cluster)
                loss_train_sfl.append(loss_avg_sfl)

                # 显示损失信息，区分训练和收敛状态
                print(f'\nEpoch {epoch} | EH_R {t3 + 1}/{k3} | ES_R {t2 + 1}/{k2}')
                loss_info = []
                if not converged_hfl_random:
                    loss_info.append(f'HFL_Random Loss: {loss_avg_hfl_random:.4f}')
                else:
                    loss_info.append(f'HFL_Random: 已收敛 ✅')
                    
                if not converged_hfl_cluster:
                    loss_info.append(f'HFL_Cluster Loss: {loss_avg_hfl_cluster:.4f}')
                else:
                    loss_info.append(f'HFL_Cluster: 已收敛 ✅')
                    
                if not converged_sfl:
                    loss_info.append(f'SFL Loss: {loss_avg_sfl:.4f}')
                else:
                    loss_info.append(f'SFL: 已收敛 ✅')
                    
                print(' | '.join(loss_info))

            # HFL 聚合 (ES -> EH) - 只对未收敛的机制进行聚合
            if not converged_hfl_random:
                EHs_ws_hfl_random = FedAvg_layered(ESs_ws_input_hfl_random, C2_random)
                t_hfl_random += t_es_to_eh_random
            else:
                print(f"  [Skip] HFL随机B矩阵已收敛，跳过EH层聚合")
            
            if not converged_hfl_cluster:
                EHs_ws_hfl_cluster = FedAvg_layered(ESs_ws_input_hfl_cluster, C2_cluster)
                t_hfl_design += t_es_to_eh_design
            else:
                print(f"  [Skip] HFL聚类B矩阵已收敛，跳过EH层聚合")
            
            # --- SFL 全局聚合 (Client -> Cloud) - 在EH层聚合时机执行 ---
            if not converged_sfl:
                w_glob_sfl = FedAvg(w_locals_output_sfl)
                net_glob_sfl.load_state_dict(w_glob_sfl)
                t_sfl += t_client_to_cloud_sfl
            else:
                print(f"  [Skip] SFL已收敛，跳过全局聚合")
            
            # --- 在每次EH聚合后测试EH模型在专属测试集上的性能 ---
            print(f"\n[EH Testing] Epoch {epoch} | EH_Round {t3+1}/{k3} - 测试EH模型性能...")
            
            # 创建临时模型对象来加载EH权重并进行测试
            eh_results_random = []
            eh_results_cluster = []
            
            # 测试每个EH的模型性能（在随机B矩阵情况下）- 只测试未收敛的机制
            if not converged_hfl_random:
                for eh_idx, eh_weights in enumerate(EHs_ws_hfl_random):
                    if eh_weights is not None and eh_idx in eh_testsets_random:
                        # 创建临时模型并加载权重
                        temp_model = build_model(args, dataset_train)
                        temp_model.load_state_dict(eh_weights)
                        temp_model.eval()
                        
                        # 在EH专属测试集上测试模型
                        eh_acc, eh_loss = test_eh_model(temp_model, dataset_test, eh_testsets_random[eh_idx], args)
                        
                        # 记录结果
                        eh_results_random.append({
                            'epoch': epoch,
                            'eh_round': t3 + 1,
                            'es_round': k2,  # ES轮次已结束
                            'train_loss': 0.0,  # EH级别没有训练损失
                            'test_loss': eh_loss,
                            'test_acc': eh_acc,
                            'model_type': 'HFL_Random_B',
                            'level': 'EH',
                            'eh_idx': eh_idx
                        })
                        
                        print(f"  [Random] EH {eh_idx}: Acc {eh_acc:.2f}%, Loss {eh_loss:.4f}")
            else:
                print("  [Skip] HFL随机B矩阵已收敛，跳过EH模型测试")
            
            # 测试每个EH的模型性能（在聚类B矩阵情况下）- 只测试未收敛的机制
            if not converged_hfl_cluster:
                for eh_idx, eh_weights in enumerate(EHs_ws_hfl_cluster):
                    if eh_weights is not None and eh_idx in eh_testsets_cluster:
                        # 创建临时模型并加载权重
                        temp_model = build_model(args, dataset_train)
                        temp_model.load_state_dict(eh_weights)
                        temp_model.eval()
                        
                        # 在EH专属测试集上测试模型
                        eh_acc, eh_loss = test_eh_model(temp_model, dataset_test, eh_testsets_cluster[eh_idx], args)
                        
                        # 记录结果
                        eh_results_cluster.append({
                            'epoch': epoch,
                            'eh_round': t3 + 1,
                            'es_round': k2,  # ES轮次已结束
                            'train_loss': 0.0,  # EH级别没有训练损失
                            'test_loss': eh_loss,
                            'test_acc': eh_acc,
                            'model_type': 'HFL_Cluster_B',
                            'level': 'EH',
                            'eh_idx': eh_idx
                        })
                        
                        print(f"  [Cluster] EH {eh_idx}: Acc {eh_acc:.2f}%, Loss {eh_loss:.4f}")
            else:
                print("  [Skip] HFL聚类B矩阵已收敛，跳过EH模型测试")
            
            # 测试SFL全局模型（在全局测试集上）- 只测试未收敛的机制
            if not converged_sfl:
                net_glob_sfl.eval()
                acc_sfl, loss_sfl = test_img(net_glob_sfl, dataset_test, args)
            else:
                print("  [Skip] SFL已收敛，跳过全局模型测试")
                # 使用上一轮的结果作为占位符
                acc_sfl, loss_sfl = acc_test_sfl[-1] if acc_test_sfl else 0.0, loss_test_sfl[-1] if loss_test_sfl else 0.0
            
            # 记录SFL模型结果
            sfl_result = {
                'epoch': epoch,
                'eh_round': t3 + 1,
                'es_round': k2,
                'train_loss': 0.0,  # 使用0.0作为占位符
                'test_loss': loss_sfl,
                'test_acc': acc_sfl,
                'model_type': 'SFL',
                'level': 'Global',
                'eh_idx': -1  # 全局模型没有EH索引
            }
            
            print(f"  [SFL Global]: Acc {acc_sfl:.2f}%, Loss {loss_sfl:.4f}")
            
            # --- 收敛性检查 ---
            print(f"\n[Convergence Check] Epoch {epoch} | EH_Round {t3+1}/{k3}")
            
            # 检查HFL随机B矩阵的收敛性
            if not converged_hfl_random:
                hfl_random_converged_count = 0
                for eh_idx, eh_weights in enumerate(EHs_ws_hfl_random):
                    if eh_weights is not None and eh_idx in eh_testsets_random:
                        # 找到对应的测试结果
                        eh_loss = None
                        eh_acc = None
                        for result in eh_results_random:
                            if result['eh_idx'] == eh_idx:
                                eh_loss = result['test_loss']
                                eh_acc = result['test_acc']
                                break
                        
                        if eh_loss is not None and eh_acc is not None:
                            should_stop, reason = eh_checkers_random[eh_idx].check(eh_loss, eh_acc, epoch)
                            print(f"  [Random] EH {eh_idx}: {reason}")
                            if should_stop:
                                hfl_random_converged_count += 1
                
                # 如果所有EH都收敛，则整个HFL随机B矩阵机制收敛
                active_ehs_random = len([eh for eh in EHs_ws_hfl_random if eh is not None])
                if hfl_random_converged_count == active_ehs_random and active_ehs_random > 0:
                    converged_hfl_random = True
                    print(f"  🎯 [Random] HFL随机B矩阵机制已收敛！所有 {active_ehs_random} 个EH都满足收敛条件")
                else:
                    print(f"  [Random] 收敛进度: {hfl_random_converged_count}/{active_ehs_random} EH已收敛")
            
            # 检查HFL聚类B矩阵的收敛性
            if not converged_hfl_cluster:
                hfl_cluster_converged_count = 0
                for eh_idx, eh_weights in enumerate(EHs_ws_hfl_cluster):
                    if eh_weights is not None and eh_idx in eh_testsets_cluster:
                        # 找到对应的测试结果
                        eh_loss = None
                        eh_acc = None
                        for result in eh_results_cluster:
                            if result['eh_idx'] == eh_idx:
                                eh_loss = result['test_loss']
                                eh_acc = result['test_acc']
                                break
                        
                        if eh_loss is not None and eh_acc is not None:
                            should_stop, reason = eh_checkers_cluster[eh_idx].check(eh_loss, eh_acc, epoch)
                            print(f"  [Cluster] EH {eh_idx}: {reason}")
                            if should_stop:
                                hfl_cluster_converged_count += 1
                
                # 如果所有EH都收敛，则整个HFL聚类B矩阵机制收敛
                active_ehs_cluster = len([eh for eh in EHs_ws_hfl_cluster if eh is not None])
                if hfl_cluster_converged_count == active_ehs_cluster and active_ehs_cluster > 0:
                    converged_hfl_cluster = True
                    print(f"  🎯 [Cluster] HFL聚类B矩阵机制已收敛！所有 {active_ehs_cluster} 个EH都满足收敛条件")
                else:
                    print(f"  [Cluster] 收敛进度: {hfl_cluster_converged_count}/{active_ehs_cluster} EH已收敛")
            
            # 检查SFL的收敛性
            if not converged_sfl:
                should_stop, reason = sfl_checker.check(loss_sfl, acc_sfl, epoch)
                print(f"  [SFL] {reason}")
                if should_stop:
                    converged_sfl = True
                    print(f"  🎯 [SFL] SFL机制已收敛！")
            
            # 将EH测试结果添加到结果历史中
            results_history.extend(eh_results_random)
            results_history.extend(eh_results_cluster)
            results_history.append(sfl_result)
            
            # 每次EH测试后更新CSV文件
            save_results_to_csv(results_history, csv_filename)

        # HFL 全局聚合 (EH -> Cloud) - 只对未收敛的机制进行聚合
        if not converged_hfl_random:
            w_glob_hfl_random = FedAvg(EHs_ws_hfl_random)
            net_glob_hfl_random.load_state_dict(w_glob_hfl_random)
            t_hfl_random += t_eh_to_cloud_random
        else:
            print(f"  [Skip] HFL随机B矩阵已收敛，跳过全局聚合")
        
        if not converged_hfl_cluster:
            w_glob_hfl_cluster = FedAvg(EHs_ws_hfl_cluster)
            net_glob_hfl_cluster.load_state_dict(w_glob_hfl_cluster)
            t_hfl_design += t_eh_to_cloud_design
        else:
            print(f"  [Skip] HFL聚类B矩阵已收敛，跳过全局聚合")

        # --- 在每个 EPOCH 结束时进行测试 - 只测试未收敛的机制 ---
        print(f"\n[End of Epoch {epoch}] 测试全局模型性能...")
        
        # 评估 HFL 随机B模型
        if not converged_hfl_random:
            net_glob_hfl_random.eval()
            acc_hfl_random, loss_hfl_random = test_img(net_glob_hfl_random, dataset_test, args)
            acc_test_hfl_random.append(acc_hfl_random)
            loss_test_hfl_random.append(loss_hfl_random)
        else:
            # 使用上一轮结果作为占位符
            acc_hfl_random = acc_test_hfl_random[-1] if acc_test_hfl_random else 0.0
            loss_hfl_random = loss_test_hfl_random[-1] if loss_test_hfl_random else 0.0
            acc_test_hfl_random.append(acc_hfl_random)
            loss_test_hfl_random.append(loss_hfl_random)
            print(f"  [Skip] HFL随机B矩阵已收敛，使用上一轮结果")

        # 评估 HFL 聚类B模型
        if not converged_hfl_cluster:
            net_glob_hfl_cluster.eval()
            acc_hfl_cluster, loss_hfl_cluster = test_img(net_glob_hfl_cluster, dataset_test, args)
            acc_test_hfl_cluster.append(acc_hfl_cluster)
            loss_test_hfl_cluster.append(loss_hfl_cluster)
        else:
            # 使用上一轮结果作为占位符
            acc_hfl_cluster = acc_test_hfl_cluster[-1] if acc_test_hfl_cluster else 0.0
            loss_hfl_cluster = loss_test_hfl_cluster[-1] if loss_test_hfl_cluster else 0.0
            acc_test_hfl_cluster.append(acc_hfl_cluster)
            loss_test_hfl_cluster.append(loss_hfl_cluster)
            print(f"  [Skip] HFL聚类B矩阵已收敛，使用上一轮结果")

        # 评估 SFL 模型
        if not converged_sfl:
            net_glob_sfl.eval()
            acc_sfl, loss_sfl = test_img(net_glob_sfl, dataset_test, args)
            acc_test_sfl.append(acc_sfl)
            loss_test_sfl.append(loss_sfl)
        else:
            # 使用上一轮结果作为占位符
            acc_sfl = acc_test_sfl[-1] if acc_test_sfl else 0.0
            loss_sfl = loss_test_sfl[-1] if loss_test_sfl else 0.0
            acc_test_sfl.append(acc_sfl)
            loss_test_sfl.append(loss_sfl)
            print(f"  [Skip] SFL已收敛，使用上一轮结果")

        # 记录当前epoch的结果 - 新格式
        current_epoch_results = [
            {
                'epoch': epoch,
                'eh_round': k3,  # 完整的EH轮次
                'es_round': k2,  # 完整的ES轮次
                'train_loss': loss_avg_hfl_random,
                'test_loss': loss_hfl_random,
                'test_acc': acc_hfl_random,
                'model_type': 'HFL_Random_B',
                'level': 'Global',
                'eh_idx': -1
            },
            {
                'epoch': epoch,
                'eh_round': k3,  # 完整的EH轮次
                'es_round': k2,  # 完整的ES轮次
                'train_loss': loss_avg_hfl_cluster,
                'test_loss': loss_hfl_cluster,
                'test_acc': acc_hfl_cluster,
                'model_type': 'HFL_Cluster_B',
                'level': 'Global',
                'eh_idx': -1
            },
            {
                'epoch': epoch,
                'eh_round': k3,  # 完整的EH轮次
                'es_round': k2,  # 完整的ES轮次
                'train_loss': loss_avg_sfl,
                'test_loss': loss_sfl,
                'test_acc': acc_sfl,
                'model_type': 'SFL',
                'level': 'Global',
                'eh_idx': -1
            }
        ]
        
        results_history.extend(current_epoch_results)

        # 保存结果到CSV
        save_results_to_csv(results_history, csv_filename)

        # 打印当前 EPOCH 结束时的测试结果
        print(f'\nEpoch {epoch} [END OF EPOCH TEST]')
        print(f'HFL_Random: Acc {acc_hfl_random:.2f}%, Loss {loss_hfl_random:.4f}')
        print(f'HFL_Cluster: Acc {acc_hfl_cluster:.2f}%, Loss {loss_hfl_cluster:.4f}')
        print(f'SFL: Acc {acc_sfl:.2f}%, Loss {loss_sfl:.4f}')

        # --- 检查是否所有机制都已收敛 ---
        if converged_hfl_random and converged_hfl_cluster and converged_sfl:
            print(f"\n🎉 所有联邦学习机制都已收敛！提前结束训练。")
            print(f"实际训练轮次: {epoch + 1}/{args.epochs}")
            final_epoch = epoch + 1
            break
        else:
            print(f"\n📊 收敛状态: HFL_Random={'✅' if converged_hfl_random else '❌'}, "
                  f"HFL_Cluster={'✅' if converged_hfl_cluster else '❌'}, "
                  f"SFL={'✅' if converged_sfl else '❌'}")

        net_glob_hfl_random.train()  # 切换回训练模式
        net_glob_hfl_cluster.train()  # 切换回训练模式
        net_glob_sfl.train()  # 切换回训练模式

    # =====================================================================================
    # Final Testing - 测试三种模型的最终性能
    # =====================================================================================
    print("\n--- Final Model Evaluation ---")
    print(f"最终收敛状态:")
    print(f"  HFL_Random: {'已收敛 ✅' if converged_hfl_random else '未收敛 ❌'}")
    print(f"  HFL_Cluster: {'已收敛 ✅' if converged_hfl_cluster else '未收敛 ❌'}")
    print(f"  SFL: {'已收敛 ✅' if converged_sfl else '未收敛 ❌'}")
    print(f"实际训练轮次: {final_epoch}/{args.epochs}")
    
    # 测试 HFL 随机B矩阵模型
    net_glob_hfl_random.eval()
    acc_train_hfl_random, loss_train_final_hfl_random = test_img(net_glob_hfl_random, dataset_train, args)
    acc_test_final_hfl_random, loss_test_final_hfl_random = test_img(net_glob_hfl_random, dataset_test, args)
    converged_str = "已收敛" if converged_hfl_random else "未收敛"
    print(f"HFL Model (Random B Matrix) [{converged_str}] - Training accuracy: {acc_train_hfl_random:.2f}%, Testing accuracy: {acc_test_final_hfl_random:.2f}%")
    
    # 测试 HFL 聚类B矩阵模型
    net_glob_hfl_cluster.eval()
    acc_train_hfl_cluster, loss_train_final_hfl_cluster = test_img(net_glob_hfl_cluster, dataset_train, args)
    acc_test_final_hfl_cluster, loss_test_final_hfl_cluster = test_img(net_glob_hfl_cluster, dataset_test, args)
    converged_str = "已收敛" if converged_hfl_cluster else "未收敛"
    print(f"HFL Model (Clustered B Matrix) [{converged_str}] - Training accuracy: {acc_train_hfl_cluster:.2f}%, Testing accuracy: {acc_test_final_hfl_cluster:.2f}%")

    # 测试 SFL 模型
    net_glob_sfl.eval()
    acc_train_sfl, loss_train_final_sfl = test_img(net_glob_sfl, dataset_train, args)
    acc_test_final_sfl, loss_test_final_sfl = test_img(net_glob_sfl, dataset_test, args)
    converged_str = "已收敛" if converged_sfl else "未收敛"
    print(f"SFL Model (Single Layer) [{converged_str}] - Training accuracy: {acc_train_sfl:.2f}%, Testing accuracy: {acc_test_final_sfl:.2f}%")

    # 保存最终结果（添加到结果历史列表中）
    # final_epoch 已在前面定义，此处不需要重复定义
    
    # 添加最终结果到结果历史列表
    final_results = [
        {
            'epoch': final_epoch,
            'eh_round': k3,
            'es_round': k2,
            'train_loss': loss_train_final_hfl_random,
            'test_loss': loss_test_final_hfl_random,
            'test_acc': acc_test_final_hfl_random,
            'model_type': 'HFL_Random_B',
            'level': 'Final',
            'eh_idx': -1
        },
        {
            'epoch': final_epoch,
            'eh_round': k3,
            'es_round': k2,
            'train_loss': loss_train_final_hfl_cluster,
            'test_loss': loss_test_final_hfl_cluster,
            'test_acc': acc_test_final_hfl_cluster,
            'model_type': 'HFL_Cluster_B',
            'level': 'Final',
            'eh_idx': -1
        },
        {
            'epoch': final_epoch,
            'eh_round': k3,
            'es_round': k2,
            'train_loss': loss_train_final_sfl,
            'test_loss': loss_test_final_sfl,
            'test_acc': acc_test_final_sfl,
            'model_type': 'SFL',
            'level': 'Final',
            'eh_idx': -1
        }
    ]
    
    # 将最终结果添加到结果历史中
    results_history.extend(final_results)
    
    # 重新保存整个结果历史到CSV文件
    save_results_to_csv(results_history, csv_filename)
    
    # 将最终结果单独保存到CSV文件（为了兼容性）
    final_summary = [
        {
            'model_type': 'HFL_Random_B',
            'final_train_acc': acc_train_hfl_random,
            'final_train_loss': loss_train_final_hfl_random,
            'final_test_acc': acc_test_final_hfl_random,
            'final_test_loss': loss_test_final_hfl_random
        },
        {
            'model_type': 'HFL_Cluster_B',
            'final_train_acc': acc_train_hfl_cluster,
            'final_train_loss': loss_train_final_hfl_cluster,
            'final_test_acc': acc_test_final_hfl_cluster,
            'final_test_loss': loss_test_final_hfl_cluster
        },
        {
            'model_type': 'SFL',
            'final_train_acc': acc_train_sfl,
            'final_train_loss': loss_train_final_sfl,
            'final_test_acc': acc_test_final_sfl,
            'final_test_loss': loss_test_final_sfl
        }
    ]

    # 将最终汇总结果追加到CSV文件
    with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([])  # 空行分隔
        writer.writerow(['Final Summary'])
        writer.writerow(['model_type', 'final_train_acc', 'final_train_loss', 'final_test_acc', 'final_test_loss'])
        for result in final_summary:
            writer.writerow([result['model_type'], result['final_train_acc'], 
                            result['final_train_loss'], result['final_test_acc'], result['final_test_loss']])

    print(f"\n=== 训练完成 ===")
    print(f"所有结果已保存到: {csv_filename}")
    
    try:
        print("\n正在生成可视化图表...")
        comprehensive_file, summary_file = create_enhanced_visualizations(csv_filename)
        print(f"综合对比图: {comprehensive_file}")
        print(f"性能总结表: {summary_file}")
    except Exception as e:
        print(f"可视化生成失败: {e}")
        print("请检查数据格式或手动运行visualization_tool.py")
    
    # 保存通信时间结果到单独的CSV文件
    communication_filename = f"communication_results_{timestamp}.csv"
    communication_csv_path = f'./results/{communication_filename}'
    
    with open(communication_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['model_type', 'total_communication_time', 'epochs', 'avg_communication_per_epoch'])
        writer.writerow(['HFL_Random_B', f"{t_hfl_random:.6f}", final_epoch, f"{t_hfl_random/final_epoch:.6f}"])
        writer.writerow(['HFL_Cluster_B', f"{t_hfl_design:.6f}", final_epoch, f"{t_hfl_design/final_epoch:.6f}"])
        writer.writerow(['SFL', f"{t_sfl:.6f}", final_epoch, f"{t_sfl/final_epoch:.6f}"])
    
    print(f"通信时间结果已保存到: {communication_csv_path}")

    print(f"\n=== 实验总结 ===")
    print("本次实验对比了三种联邦学习方法:")
    print("1. HFL (Random B Matrix) - 使用随机生成的ES-EH关联矩阵")
    print("2. HFL (Clustered B Matrix) - 使用谱聚类生成的ES-EH关联矩阵") 
    print("3. SFL (Single Layer) - 传统单层联邦学习")
    print(f"训练参数: 设定epochs={args.epochs}, 实际epochs={final_epoch}, clients={args.num_users}, local_epochs={args.local_ep}")
    print(f"层级参数: k2={args.ES_k2} (ES层聚合轮数), k3={args.EH_k3} (EH层聚合轮数)")
    print(f"并行参数: num_processes={args.num_processes}")
    print(f"数据集: {args.dataset}, 模型: {args.model}, IID: {args.iid}")
    if not args.iid:
        print(f"非IID参数: beta={args.beta}")
    
    print(f"\n=== 通信开销分析 ===")
    print(f"总通信时间对比:")
    print(f"  • HFL_Random: {t_hfl_random:.6f}s (平均每轮: {t_hfl_random/final_epoch:.6f}s)")
    print(f"  • HFL_Cluster: {t_hfl_design:.6f}s (平均每轮: {t_hfl_design/final_epoch:.6f}s)")
    print(f"  • SFL: {t_sfl:.6f}s (平均每轮: {t_sfl/final_epoch:.6f}s)")
    
    print(f"\n=== 收敛性分析 ===")
    print(f"收敛检查器参数: patience=5, min_delta=0.001")
    print(f"最终收敛状态:")
    print(f"  • HFL_Random: {'✅ 已收敛' if converged_hfl_random else '❌ 未收敛'}")
    print(f"  • HFL_Cluster: {'✅ 已收敛' if converged_hfl_cluster else '❌ 未收敛'}")  
    print(f"  • SFL: {'✅ 已收敛' if converged_sfl else '❌ 未收敛'}")
    
    if converged_hfl_random and converged_hfl_cluster and converged_sfl:
        print(f"🎉 所有机制均收敛，训练在第{final_epoch}轮提前结束")
    elif final_epoch < args.epochs:
        print(f"⚠️ 部分机制收敛，训练在第{final_epoch}轮提前结束")
    else:
        print(f"⏰ 训练完成设定的{args.epochs}轮，部分机制可能未完全收敛")
    
    # 显示最终性能对比
    try:
        df = pd.read_csv(csv_filename, encoding='utf-8')
        final_results = df[df['epoch'] == df['epoch'].max()]
        print(f"\n最终测试准确率对比:")
        for _, row in final_results.iterrows():
            model_name = row['model_type'].replace('_', ' ')
            print(f"  {model_name}: {row['test_acc']:.2f}%")
    except:
        pass
