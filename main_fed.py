#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from tqdm import tqdm
import multiprocessing
import os
import csv
from datetime import datetime

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from utils.data_partition import get_client_datasets
from utils.visualize_client_data import visualize_client_data_distribution
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg, FedAvg_layered
from models.test import test_img
from models.cluster import (
    train_initial_models,
    aggregate_es_models, spectral_clustering_es,
    visualize_es_clustering_result,
    calculate_es_label_distributions
)

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
        tuple: (训练数据集, 测试数据集, 客户端数据映射)
    """

    return get_client_datasets(dataset_type, num_clients, data_path, partition_method, noniid_param)


def get_data(args):
    """兼容原有接口的数据获取函数"""

    # 确定数据集类型和路径
    if args.dataset == 'mnist':
        dataset_type = 'mnist'
        data_path = os.path.join(args.data_path, 'mnist/')
        # 创建兼容的数据集对象
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST(data_path, train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST(data_path, train=False, download=True, transform=trans_mnist)

    elif args.dataset == 'cifar':
        dataset_type = 'cifar10'
        data_path = os.path.join(args.data_path, 'cifar/')
        # 创建兼容的数据集对象
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10(data_path, train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10(data_path, train=False, download=True, transform=trans_cifar)

    else:
        exit('Error: unrecognized dataset')

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
    train_data, test_data, dict_users = get_data_new(
        dataset_type, args.num_users, data_path, partition_method, noniid_param
    )
    # visualize_client_data_distribution(dict_users, dataset_train, args)
    return dataset_train, dataset_test, dict_users


def build_model(args, dataset_train):
    img_size = dataset_train[0][0].shape

    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        # 计算将图片展平后的输入层维度
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('错误：无法识别的模型')

    print("--- 模型架构 ---")
    print(net_glob)
    print("--------------------")
    return net_glob


def get_A(num_users, num_ESs):
    A = np.zeros((num_users, num_ESs), dtype=int)

    # 对每一行随机选择一个索引，将该位置设为 1
    for i in range(num_users):
        random_index = np.random.randint(0, num_ESs)
        A[i, random_index] = 1

    return A


'''def get_B(num_ESs, num_EHs):
    B = np.zeros((num_ESs, num_EHs), dtype=int)

    # 对每一行随机选择一个索引，将该位置设为 1
    for i in range(num_ESs):
        random_index = np.random.randint(0, num_EHs)
        B[i, random_index] = 1

    return B'''


def get_B_cluster(args, w_locals, A, dict_users, net_glob, client_label_distributions, num_EHs=None):
    """
    使用谱聚类生成 ES-EH 关联矩阵 B，并可视化聚类结果
    """
    print("开始谱聚类生成B矩阵...")

    # 1. 聚合ES模型
    es_models = aggregate_es_models(w_locals, A, dict_users, net_glob)

    # 2. 使用谱聚类获取ES-EH关联矩阵B
    B, cluster_labels = spectral_clustering_es(
        es_models,
        num_EHs=num_EHs,  # 可以指定或设置为None自动确定
        sigma=args.sigma,  # 从参数中获取
        epsilon=args.epsilon  # 从参数中获取
    )

    # 3. 计算每个ES的标签分布
    es_label_distributions = calculate_es_label_distributions(A, client_label_distributions)

    # 4. 可视化ES聚类结果
    visualize_es_clustering_result(
        es_label_distributions=es_label_distributions,
        cluster_labels=cluster_labels,
        save_path='./save/es_clustering_result.png'
    )

    return B  # 只返回B矩阵


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


def train_client(args, user_idx, dataset_train, dict_users, w_input_hfl, w_sfl_global):
    """
    单个客户端的训练函数，用于被多进程调用。

    注意：为了兼容多进程，我们不直接传递大型模型对象，
    而是传递模型权重(state_dict)和模型架构信息(args)，在子进程中重新构建模型。
    """
    # 在子进程中重新构建模型
    # 这种方式可以避免在进程间传递复杂的、不可序列化的对象
    # print(f"CLIENT_{user_idx} START")
    if args.model == 'cnn' and args.dataset == 'cifar':
        local_net_hfl = CNNCifar(args=args)
        local_net_sfl = CNNCifar(args=args)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        local_net_hfl = CNNMnist(args=args)
        local_net_sfl = CNNMnist(args=args)
    # 你可以根据需要添加 MLP 等其他模型的构建逻辑
    else:
        # 退出或抛出错误
        exit('Error: unrecognized model in train_client')
    # print(f"CLIENT_{user_idx} TRAIN")
    # --- 训练分层模型 (HFL) ---
    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[user_idx])
    local_net_hfl.load_state_dict(w_input_hfl)
    w_hfl, loss_hfl = local.train(net=local_net_hfl.to(args.device))
    # print(f"CLIENT_{user_idx} TRAINing")
    # --- 训练单层模型 (SFL) ---
    local_net_sfl.load_state_dict(w_sfl_global)
    w_sfl, loss_sfl = local.train(net=local_net_sfl.to(args.device))
    print(f"CLIENT_{user_idx} END")

    # 返回结果，包括 user_idx 以便后续排序
    return user_idx, copy.deepcopy(w_hfl), loss_hfl, copy.deepcopy(w_sfl), loss_sfl

def save_results_to_csv(results, filename):
    """Save results to CSV file"""
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['epoch', 'hfl_test_acc', 'hfl_test_loss', 'sfl_test_acc', 'sfl_test_loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    dataset_train, dataset_test, dict_users = get_data(args)

    net_glob = build_model(args, dataset_train)

    net_glob.train()

    # 初始化全局权重
    w_glob = net_glob.state_dict()
    num_users = args.num_users
    num_ESs = num_users // 5
    num_EHs = num_ESs // 3
    k2 = 2
    k3 = 2
    num_processes = 8  # min(args.num_users//3, (os.cpu_count())//3)

    A = get_A(num_users, num_ESs)

    # 使用谱聚类生成B矩阵（替换原来的随机B矩阵）
    print("开始初始训练和谱聚类...")

    # 1. 训练初始本地模型
    w_locals, client_label_distributions = train_initial_models(
        args, dataset_train, dict_users, net_glob, num_users
    )

    # 2. 使用谱聚类生成B矩阵
    B = get_B_cluster(
        args, w_locals, A, dict_users, net_glob, client_label_distributions, num_EHs
    )

    C1, C2 = build_hierarchy(A, B)
    print("C1 (一级->客户端):", C1)
    print("C2 (二级->一级):", C2)

    # 创建结果保存目录
    if not os.path.exists('./results'):
        os.makedirs('./results')

    # 生成唯一的时间戳用于文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f'./results/training_results_{timestamp}.csv'

    # 初始化结果记录列表
    results_history = []

    # 训练前测试初始模型
    print("\n--- Testing Initial Global Models ---")
    net_glob.eval()

    # 测试初始 HFL 模型
    acc_hfl_init, loss_hfl_init = test_img(net_glob, dataset_test, args)
    print(f"Initial HFL Model - Testing accuracy: {acc_hfl_init:.2f}%, Loss: {loss_hfl_init:.4f}")

    # 测试初始 SFL 模型
    acc_sfl_init, loss_sfl_init = test_img(net_glob, dataset_test, args)
    print(f"Initial SFL Model - Testing accuracy: {acc_sfl_init:.2f}%, Loss: {loss_sfl_init:.4f}")

    # 记录初始结果
    results_history.append({
        'epoch': -1,
        'hfl_test_acc': acc_hfl_init,
        'hfl_test_loss': loss_hfl_init,
        'sfl_test_acc': acc_sfl_init,
        'sfl_test_loss': loss_sfl_init
    })

    # 保存初始结果到CSV
    save_results_to_csv(results_history, csv_filename)

    # training
    # --- 初始化两个模型，确保初始权重相同 ---
    # net_glob_hfl 是 HFL 的全局模型
    net_glob_hfl = net_glob
    w_glob_hfl = net_glob_hfl.state_dict()

    # net_glob_sfl 是 SFL 的全局模型
    net_glob_sfl = copy.deepcopy(net_glob)
    w_glob_sfl = net_glob_sfl.state_dict()

    # --- 分别记录两种模型的指标 ---
    loss_train_hfl = []
    loss_train_sfl = []
    loss_test_hfl = []
    loss_test_sfl = []
    acc_test_hfl = []
    acc_test_sfl = []

    # 添加提前停止标志
    early_stop = False

    for epoch in range(args.epochs):
        if early_stop:
            print(f"HFL accuracy reached 95% at epoch {epoch - 1}. Stopping training early.")
            break

        # HFL 模型权重分发 (Cloud -> EH)
        EHs_ws_hfl = [copy.deepcopy(w_glob_hfl) for _ in range(num_EHs)]

        # EH 层聚合 k3 轮
        for t3 in range(k3):
            # HFL: EH 层 -> ES 层
            ESs_ws_input_hfl = [None] * num_ESs
            for EH_idx, ES_indices in C2.items():
                for ES_idx in ES_indices:
                    ESs_ws_input_hfl[ES_idx] = copy.deepcopy(EHs_ws_hfl[EH_idx])

            # ES 层聚合 k2 轮
            for t2 in range(k2):
                # --- 本地训练 (HFL 和 SFL 并行) ---
                # HFL: ES 层 -> Client 层
                w_locals_input_hfl = [None] * num_users
                for ES_idx, user_indices in C1.items():
                    for user_idx in user_indices:
                        w_locals_input_hfl[user_idx] = copy.deepcopy(ESs_ws_input_hfl[ES_idx])

                # 用于存储两种模型本地训练的输出
                w_locals_output_hfl = [None] * num_users
                w_locals_output_sfl = [None] * num_users
                loss_locals_hfl = []
                loss_locals_sfl = []

                print(
                    f"\n[Parallel Training] Starting training for {args.num_users} clients using {num_processes} processes...")

                # 准备传递给每个子进程的参数
                tasks = []
                for user_idx in range(num_users):
                    # 注意：这里我们不传递完整的 net_glob_hfl 和 net_glob_sfl 对象
                    # 因为它们可能不可序列化。我们已经在工作函数内部重新构建它们。
                    task_args = (
                        args, user_idx, dataset_train, dict_users,
                        w_locals_input_hfl[user_idx], w_glob_sfl,
                    )
                    tasks.append(task_args)
                print("成功创建多线程！")

                # 创建进程池并分发任务
                # 使用 with 语句可以自动管理进程池的关闭
                with multiprocessing.Pool(processes=num_processes) as pool:
                    # ===================== 主要修改点 =====================
                    # 使用 tqdm 包装 tasks 列表，以在多进程训练期间显示进度条。
                    # desc 参数为进度条提供一个描述性标签。
                    # 随着每个客户端训练任务的完成，进度条会自动更新。
                    # ====================================================
                    results = pool.starmap(train_client,
                                           tqdm(tasks, desc=f"Epoch {epoch}|{t3 + 1}|{t2 + 1} Training Clients"))

                print("训练结束")
                # 3. 收集并整理所有客户端的训练结果
                # 因为 starmap 保证了顺序，我们可以直接处理
                # 如果使用 imap_unordered，则需要根据返回的 user_idx 来排序
                for result in results:
                    u_idx, w_h, l_h, w_s, l_s = result
                    w_locals_output_hfl[u_idx] = w_h
                    loss_locals_hfl.append(l_h)
                    w_locals_output_sfl[u_idx] = w_s
                    loss_locals_sfl.append(l_s)
                print("排序结束")
                print(f"[Parallel Training] All {args.num_users} clients have finished training.")
                print(f'\nEpoch {epoch} | EH_R {t3 + 1}/{k3} | ES_R {t2 + 1}/{k2} | 开始聚合')

                # --- HFL 聚合 (Client -> ES) ---
                ESs_ws_input_hfl = FedAvg_layered(w_locals_output_hfl, C1)

                # --- SFL 全局聚合 (Client -> Cloud) ---
                w_glob_sfl = FedAvg(w_locals_output_sfl)
                net_glob_sfl.load_state_dict(w_glob_sfl)

                # --- 记录损失 ---
                loss_avg_hfl = sum(loss_locals_hfl) / len(loss_locals_hfl)
                loss_avg_sfl = sum(loss_locals_sfl) / len(loss_locals_sfl)
                loss_train_hfl.append(loss_avg_hfl)
                loss_train_sfl.append(loss_avg_sfl)

                print(
                    f'\nEpoch {epoch} | EH_R {t3 + 1}/{k3} | ES_R {t2 + 1}/{k2} | HFL Loss {loss_avg_hfl:.4f} | SFL Loss {loss_avg_sfl:.4f}')

            # HFL 聚合 (ES -> EH)
            EHs_ws_hfl = FedAvg_layered(ESs_ws_input_hfl, C2)

        # HFL 全局聚合 (EH -> Cloud)
        # HFL 的全局模型只在 k3 轮结束后才更新一次
        w_glob_hfl = FedAvg(EHs_ws_hfl)
        net_glob_hfl.load_state_dict(w_glob_hfl)

        # --- 在每个 EPOCH 结束时进行测试 ---
        net_glob_hfl.eval()
        net_glob_sfl.eval()

        # 评估 HFL 模型
        acc_hfl, loss_hfl = test_img(net_glob_hfl, dataset_test, args)
        acc_test_hfl.append(acc_hfl)
        loss_test_hfl.append(loss_hfl)

        # 评估 SFL 模型
        acc_sfl, loss_sfl = test_img(net_glob_sfl, dataset_test, args)
        acc_test_sfl.append(acc_sfl)
        loss_test_sfl.append(loss_sfl)

        # 记录当前epoch的结果
        results_history.append({
            'epoch': epoch,
            'hfl_test_acc': acc_hfl,
            'hfl_test_loss': loss_hfl,
            'sfl_test_acc': acc_sfl,
            'sfl_test_loss': loss_sfl
        })

        # 保存结果到CSV
        save_results_to_csv(results_history, csv_filename)

        # 打印当前 EPOCH 结束时的测试结果
        print(
            f'\nEpoch {epoch} [END OF EPOCH TEST] | HFL Acc: {acc_hfl:.2f}%, Loss: {loss_hfl:.4f} | SFL Acc: {acc_sfl:.2f}%, Loss: {loss_sfl:.4f}')

        # 检查是否达到停止条件
        if acc_hfl >= 97.0:
            print(f"HFL accuracy reached {acc_hfl:.2f}% at epoch {epoch}. Stopping training early.")
            early_stop = True

        net_glob_hfl.train()  # 切换回训练模式
        net_glob_sfl.train()  # 切换回训练模式

    # =====================================================================================
    # Plot loss curve
    # =====================================================================================
    plt.figure()
    plt.plot(range(len(loss_train_hfl)), loss_train_hfl, label='HFL (3-layer)')
    plt.plot(range(len(loss_train_sfl)), loss_train_sfl, label='SFL (Frequent Update)')
    plt.ylabel('Train Loss')
    plt.xlabel('Communication Rounds (ES-Client Level)')
    plt.legend()
    plt.title('HFL vs SFL Training Loss')
    plt.savefig(
        './save/fed_compare_freq_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac,
                                                                args.iid))
    # =====================================================================================
    # Plot test loss curve
    # =====================================================================================
    plt.figure()
    plt.plot(range(len(loss_test_hfl)), loss_test_hfl, label='HFL (Test Loss)')
    plt.plot(range(len(loss_test_sfl)), loss_test_sfl, label='SFL (Test Loss)')
    plt.ylabel('Test Loss')
    plt.xlabel('Communication Rounds (ES-Client Level)')
    plt.legend()
    plt.title('HFL vs SFL Test Loss')
    plt.savefig(
        './save/fed_test_loss_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac,
                                                             args.iid))
    plt.show()

    # =====================================================================================
    # Testing
    # =====================================================================================
    print("\n--- Final Model Evaluation ---")
    # 测试 HFL 模型
    net_glob_hfl.eval()
    acc_train_hfl, loss_train_hfl = test_img(net_glob_hfl, dataset_train, args)
    acc_test_hfl, loss_test_hfl = test_img(net_glob_hfl, dataset_test, args)
    print(f"HFL Model (Slower Global Update) - Training accuracy: {acc_train_hfl:.2f}%")
    print(f"HFL Model (Slower Global Update) - Testing accuracy: {acc_test_hfl:.2f}%")

    # 测试 SFL 模型
    net_glob_sfl.eval()
    acc_train_sfl, loss_train_sfl = test_img(net_glob_sfl, dataset_train, args)
    acc_test_sfl, loss_test_sfl = test_img(net_glob_sfl, dataset_test, args)
    print(f"SFL Model (Frequent Global Update) - Training accuracy: {acc_train_sfl:.2f}%")
    print(f"SFL Model (Frequent Global Update) - Testing accuracy: {acc_test_sfl:.2f}%")

    # 保存最终结果
    final_results = {
        'hfl_train_acc': acc_train_hfl,
        'hfl_train_loss': loss_train_hfl,
        'hfl_test_acc': acc_test_hfl,
        'hfl_test_loss': loss_test_hfl,
        'sfl_train_acc': acc_train_sfl,
        'sfl_train_loss': loss_train_sfl,
        'sfl_test_acc': acc_test_sfl,
        'sfl_test_loss': loss_test_sfl
    }

    # 将最终结果追加到CSV文件
    with open(csv_filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([])  # 空行分隔
        writer.writerow(['Final Results'])
        for key, value in final_results.items():
            writer.writerow([key, value])

    print(f"\nAll results saved to {csv_filename}")