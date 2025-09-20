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
import csv
from datetime import datetime

# 修正导入语句
from utils.options import args_parser  # 替换 import args
from bipartite_bandwidth import run_bandwidth_allocation
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg, FedAvg_layered
from models.test import test_img


def calculate_comm_overhead(args, assignments, r, B_n, C1, k2, k3, model_size):
    """
    计算逐层通信开销（延迟，秒）。
    - Client-ES: 每个ES下 max( model_size / r_mn ) * k2
    - ES-EH/Cloud: max( model_size / B_n[ES] ) * k3
    - 返回每epoch开销、客户端到ES延迟列表、ES到云的最大延迟
    """
    num_ESs = len(C1)
    # Client-ES层：每个ES的最大客户端延迟（上传/下载对称）
    es_client_delays = []
    for es_idx in range(num_ESs):
        clients = C1[es_idx]
        if not clients:
            es_client_delays.append(0.0)
            continue
        max_client_delay = 0.0
        for client_idx in clients:
            for m, n in assignments:
                if m == client_idx and n == es_idx:
                    r_mn = r[m, n] / 8  # bit/s to bytes/s (除以8)
                    if r_mn > 0:
                        client_delay = model_size / r_mn  # 秒
                        max_client_delay = max(max_client_delay, client_delay)
                    break
        es_client_delays.append(max_client_delay * k2)  # 累加k2轮
    client_es_total = sum(es_client_delays)  # 所有ES累加

    # ES-EH/Cloud层：所有ES的最大延迟
    es_cloud_delays = []
    for es_idx in range(num_ESs):
        if B_n[es_idx] > 0:
            es_delay = model_size / (B_n[es_idx] / 8)  # bit/s to bytes/s
            es_cloud_delays.append(es_delay)
    max_es_cloud_delay = max(es_cloud_delays) if es_cloud_delays else 0.0
    es_cloud_total = max_es_cloud_delay * k3  # 以最长ES为准，累加k3轮

    epoch_overhead = client_es_total + es_cloud_total  # 逐层累加
    return epoch_overhead, es_client_delays, max_es_cloud_delay

def get_data(args):
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
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


def get_A_from_assignments(assignments, num_users, num_ESs):
    A = np.zeros((num_users, num_ESs), dtype=int)
    print(f"Creating A matrix with shape: ({num_users}, {num_ESs})")
    print(f"Assignments: {assignments}")

    for m, n in assignments:
        if n < 0 or n >= num_ESs:
            print(f"Warning: Invalid edge server index {n} for client {m}, skipping (valid range: 0 to {num_ESs - 1})")
            continue
        if m < 0 or m >= num_users:
            print(f"Warning: Invalid client index {m}, skipping (valid range: 0 to {num_users - 1})")
            continue
        A[m, n] = 1
    return A


def get_B(num_ESs, num_EHs):
    B = np.zeros((num_ESs, num_EHs), dtype=int)

    # 对每一行随机选择一个索引，将该位置设为 1
    for i in range(num_ESs):
        random_index = np.random.randint(0, num_EHs)
        B[i, random_index] = 1

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

def save_results_to_csv(results, final_results, filename):
    """Save results to CSV file"""
    import csv
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['epoch', 'hfl_test_acc', 'hfl_test_loss', 'sfl_test_acc', 'sfl_test_loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
        # 只有当 final_results 不为空时才写入最终结果
        if final_results:
            writer.writerow({key: '' for key in fieldnames})  # 空行分隔符
            writer.writerow({'epoch': 'Final Results'})
            for key, value in final_results.items():
                writer.writerow({'epoch': key, 'hfl_test_acc': value if key == 'hfl_test_acc' else '',
                                'hfl_test_loss': value if key == 'hfl_test_loss' else '',
                                'sfl_test_acc': value if key == 'sfl_test_acc' else '',
                                'sfl_test_loss': value if key == 'sfl_test_loss' else ''})


if __name__ == '__main__':

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 临时保留
    test_mode = True  # 或通过 args.test_mode 设置
    num_processes = min(4, os.cpu_count() or 1)  # 优化进程数
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # 强制设置测试模式的 num_users
    if test_mode:
        args.num_users = 10

    # 调用带宽分配
    bipartite_graph, client_nodes, es_nodes, distance_matrix, r, assignments, loads, B_n = run_bandwidth_allocation(
        num_users=args.num_users
    )
    if bipartite_graph is None:
        exit("错误：无法构建二部图，请检查 graph-example/Ulaknet.graphml 文件。")

    # 调试：验证 run_bandwidth_allocation 输出
    num_ESs = len(es_nodes)
    print(f"Number of edge servers (num_ESs): {num_ESs}")
    print(f"Edge server nodes: {es_nodes}")
    print(f"Assignments before processing: {assignments}")
    invalid_assignments = [(m, n) for m, n in assignments if n >= num_ESs or n < 0]
    if invalid_assignments:
        print(f"Error: Invalid assignments detected: {invalid_assignments}")
        exit("错误：分配中包含无效的边缘服务器索引")

    args.num_users = min(args.num_users, len(client_nodes))

    if test_mode:
        print("进入测试模式：减少参数以进行快速验证。")
        args.epochs = 2
        args.num_users = min(10, args.num_users)
        client_nodes = client_nodes[:args.num_users]
        assignments = [(i, j) for i, j in assignments if i < args.num_users]
        print(f"Assignments after test mode filtering: {assignments}")
        num_processes = min(2, num_processes)
        k2 = 1
        k3 = 1
        num_EHs = max(1, num_ESs // 5)
        dataset_train, dataset_test, dict_users = get_data(args)
    else:
        dataset_train, dataset_test, dict_users = get_data(args)
        k2 = 2
        k3 = 2
        num_EHs = num_ESs // 3

    # 检查数据集
    for idx in dict_users:
        if len(dict_users[idx]) == 0:
            print(f"Warning: Client {idx} has empty dataset")

    A = get_A_from_assignments(assignments, args.num_users, num_ESs)
    B = get_B(num_ESs, num_EHs)
    C1, C2 = build_hierarchy(A, B)
    print("C1 (一级->客户端):", C1)
    print("C2 (二级->一级):", C2)

    net_glob = build_model(args, dataset_train)
    net_glob.train()
    w_glob = net_glob.state_dict()

    # 创建结果保存目录
    if not os.path.exists('./results'):
        os.makedirs('./results')

    # 生成唯一时间戳用于文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f'./results/training_results_{timestamp}.csv'

    # 初始化结果记录列表
    results_history = []
    final_results = {}  # 初始化 final_results 为空字典，避免未定义问题

    # 新增：初始化通信开销
    total_comm_overhead = 0.0  # 总通信开销 (秒)

    # 训练前测试初始模型
    print("\n--- 测试初始全局模型 ---")
    net_glob.eval()
    acc_hfl_init, loss_hfl_init = test_img(net_glob, dataset_test, args)
    print(f"初始 HFL 模型 - 测试准确率: {acc_hfl_init:.2f}%, 损失: {loss_hfl_init:.4f}")
    acc_sfl_init, loss_sfl_init = test_img(net_glob, dataset_test, args)
    print(f"初始 SFL 模型 - 测试准确率: {acc_sfl_init:.2f}%, 损失: {loss_sfl_init:.4f}")

    # 记录初始结果
    results_history.append({
        'epoch': -1,
        'hfl_test_acc': acc_hfl_init,
        'hfl_test_loss': loss_hfl_init,
        'sfl_test_acc': acc_sfl_init,
        'sfl_test_loss': loss_sfl_init
    })
    save_results_to_csv(results_history, {}, csv_filename)  # 使用空字典

    # --- 初始化两个模型 ---
    net_glob_hfl = net_glob
    w_glob_hfl = net_glob_hfl.state_dict()
    net_glob_sfl = copy.deepcopy(net_glob)
    w_glob_sfl = net_glob_sfl.state_dict()

    # --- 记录指标 ---
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
                # HFL: ES 层 -> Client 层
                w_locals_input_hfl = [None] * args.num_users
                for ES_idx, user_indices in C1.items():
                    for user_idx in user_indices:
                        w_locals_input_hfl[user_idx] = copy.deepcopy(ESs_ws_input_hfl[ES_idx])

                # 用于存储训练输出
                w_locals_output_hfl = [None] * args.num_users
                w_locals_output_sfl = [None] * args.num_users
                loss_locals_hfl = []
                loss_locals_sfl = []

                print(
                    f"\n[Parallel Training] Starting training for {args.num_users} clients using {num_processes} processes...")
                tasks = []
                for user_idx in range(args.num_users):
                    task_args = (
                        args, user_idx, dataset_train, dict_users,
                        w_locals_input_hfl[user_idx], w_glob_sfl,
                    )
                    tasks.append(task_args)
                print("成功创建多线程！")

                with multiprocessing.Pool(processes=num_processes) as pool:
                    results = pool.starmap(train_client,
                                           tqdm(tasks, desc=f"Epoch {epoch}|{t3 + 1}|{t2 + 1} Training Clients"))

                print("训练结束")
                for result in results:
                    u_idx, w_h, l_h, w_s, l_s = result
                    if w_h is None or not w_h:
                        print(f"Warning: Client {u_idx} returned invalid HFL weights")
                        w_h = {}
                    if w_s is None or not w_s:
                        print(f"Warning: Client {u_idx} returned invalid SFL weights")
                        w_s = {}
                    w_locals_output_hfl[u_idx] = w_h
                    loss_locals_hfl.append(l_h)
                    w_locals_output_sfl[u_idx] = w_s
                    loss_locals_sfl.append(l_s)
                print("排序结束")
                print(f"[Parallel Training] All {args.num_users} clients have finished training.")

                print(f'\nEpoch {epoch} | EH_R {t3 + 1}/{k3} | ES_R {t2 + 1}/{k2} | 开始聚合')
                ESs_ws_input_hfl = FedAvg_layered(w_locals_output_hfl, C1)
                if ESs_ws_input_hfl is None or all(w is None for w in ESs_ws_input_hfl):
                    print("错误：HFL ES层聚合失败，无有效模型参数")
                    continue
                w_glob_sfl = FedAvg(w_locals_output_sfl)
                if w_glob_sfl is None:
                    print("错误：SFL 全局模型聚合失败")
                    continue
                net_glob_sfl.load_state_dict(w_glob_sfl)

                loss_avg_hfl = sum([l for l in loss_locals_hfl if l != float('inf')]) / max(1, len([l for l in
                                                                                                    loss_locals_hfl if
                                                                                                    l != float('inf')]))
                loss_avg_sfl = sum([l for l in loss_locals_sfl if l != float('inf')]) / max(1, len([l for l in
                                                                                                    loss_locals_sfl if
                                                                                                    l != float('inf')]))
                loss_train_hfl.append(loss_avg_hfl)
                loss_train_sfl.append(loss_avg_sfl)
                print(
                    f'Epoch {epoch} | EH_R {t3 + 1}/{k3} | ES_R {t2 + 1}/{k2} | HFL Loss {loss_avg_hfl:.4f} | SFL Loss {loss_avg_sfl:.4f}')

            # HFL 聚合 (ES -> EH)
            EHs_ws_hfl = FedAvg_layered(ESs_ws_input_hfl, C2)
            if EHs_ws_hfl is None or all(w is None for w in EHs_ws_hfl):
                print("错误：HFL EH层聚合失败，无有效模型参数")
                continue

        # HFL 全局聚合 (EH -> Cloud)
        w_glob_hfl = FedAvg(EHs_ws_hfl)
        if w_glob_hfl is None:
            print("错误：HFL 全局模型聚合失败")
            continue
        net_glob_hfl.load_state_dict(w_glob_hfl)

        # 计算通信开销
        epoch_overhead, es_client_delays, max_es_cloud_delay = calculate_comm_overhead(args, assignments, r, B_n, C1,
                                                                                       k2, k3, args.model_size)
        total_comm_overhead += epoch_overhead
        print(
            f'Epoch {epoch} Communication Overhead: {epoch_overhead:.4f}s (Client-ES: {max(es_client_delays):.4f}s max, ES-Cloud: {max_es_cloud_delay:.4f}s max)')
        # 测试
        net_glob_hfl.eval()
        net_glob_sfl.eval()
        acc_hfl, loss_hfl = test_img(net_glob_hfl, dataset_test, args)
        acc_sfl, loss_sfl = test_img(net_glob_sfl, dataset_test, args)
        acc_test_hfl.append(acc_hfl)
        loss_test_hfl.append(loss_hfl)
        acc_test_sfl.append(acc_sfl)
        loss_test_sfl.append(loss_sfl)

        results_history.append({
            'epoch': epoch,
            'hfl_test_acc': acc_hfl,
            'hfl_test_loss': loss_hfl,
            'sfl_test_acc': acc_sfl,
            'sfl_test_loss': loss_sfl
        })
        save_results_to_csv(results_history, {}, csv_filename)  # 使用空字典

        print(
            f'Epoch {epoch} [END OF EPOCH TEST] | HFL Acc: {acc_hfl:.2f}%, Loss: {loss_hfl:.4f} | SFL Acc: {acc_sfl:.2f}%, Loss: {loss_sfl:.4f}')

        if acc_hfl >= 95.0:
            print(f"HFL accuracy reached {acc_hfl:.2f}% at epoch {epoch}. Stopping training early.")
            early_stop = True

        net_glob_hfl.train()
        net_glob_sfl.train()

    # 绘制损失曲线
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
    plt.close()

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
    plt.close()

    # 最终评估
    print("\n--- Final Model Evaluation ---")
    net_glob_hfl.eval()
    acc_train_hfl, loss_train_hfl = test_img(net_glob_hfl, dataset_train, args)
    acc_test_hfl, loss_test_hfl = test_img(net_glob_hfl, dataset_test, args)
    print(f"HFL Model - Training accuracy: {acc_train_hfl:.2f}%")
    print(f"HFL Model - Testing accuracy: {acc_test_hfl:.2f}%")

    net_glob_sfl.eval()
    acc_train_sfl, loss_train_sfl = test_img(net_glob_sfl, dataset_train, args)
    acc_test_sfl, loss_test_sfl = test_img(net_glob_sfl, dataset_test, args)
    print(f"SFL Model - Training accuracy: {acc_train_sfl:.2f}%")
    print(f"SFL Model - Testing accuracy: {acc_test_sfl:.2f}%")

    final_results = {
        'hfl_train_acc': acc_train_hfl,
        'hfl_train_loss': loss_train_hfl,
        'hfl_test_acc': acc_test_hfl,
        'hfl_test_loss': loss_test_hfl,
        'sfl_train_acc': acc_train_sfl,
        'sfl_train_loss': loss_train_sfl,
        'sfl_test_acc': acc_test_sfl,
        'sfl_test_loss': loss_test_sfl,
        'total_comm_overhead': total_comm_overhead
    }

    save_results_to_csv(results_history, final_results, csv_filename)

    print(f"\nAll results saved to {csv_filename}")
    print(f"\nTotal Communication Overhead over {args.epochs} epochs: {total_comm_overhead:.4f}s")