#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.7 (updated for compatibility)
import os
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from tqdm import tqdm
import multiprocessing as mp  # Added alias for multiprocessing
import csv
from datetime import datetime

from utils.options import args_parser
from bipartite_bandwidth import run_bandwidth_allocation, plot_assigned_graph
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg, FedAvg_layered
from models.test import test_img

def select_cluster_heads(C2, es_nodes, r_es, loads, B_n):
    cluster_heads = {}
    for ch_idx, es_indices in C2.items():
        if not es_indices:
            print(f"Warning: Cluster {ch_idx} is empty, skipping cluster head selection")
            cluster_heads[ch_idx] = -1
            continue
        if len(es_indices) == 1:
            cluster_heads[ch_idx] = es_indices[0]
            print(f"Cluster {ch_idx} has only one ES {es_nodes[es_indices[0]]}, selected as cluster head")
            continue
        scores = []
        for es_idx in es_indices:
            avg_distance = np.mean([r_es[es_idx, other_idx] for other_idx in es_indices if other_idx != es_idx])
            if avg_distance == 0:
                avg_distance = 1e-10
            score = 0.3 * (1 / avg_distance) + 0.2 * (1 / (loads[es_idx] + 1)) + 0.5 * (B_n[es_idx] / 5e7)
            scores.append((es_idx, score))
        best_es_idx = max(scores, key=lambda x: x[1])[0]
        cluster_heads[ch_idx] = best_es_idx
        print(f"Cluster {ch_idx}: Selected ES {es_nodes[best_es_idx]} as cluster head (score: {max(scores, key=lambda x: x[1])[1]:.4f})")
    return cluster_heads

def calculate_comm_overhead_hfl(args, assignments, r, r_es, B_n, C1, C2, cluster_heads, k2, k3, model_size, es_nodes):
    num_ESs = len(C1)
    # Upload delays
    es_client_delays_upload = []
    for es_idx in range(num_ESs):
        clients = C1[es_idx]
        if not clients:
            print(f"ES {es_nodes[es_idx]}: 0 clients, max_client_delay_upload=0.0s")
            continue
        max_client_delay = 0.0
        min_r_in_es = float('inf')
        for client_idx in clients:
            for m, n in assignments:
                if m == client_idx and n == es_idx:
                    r_mn = r[m, n] / 8
                    if r_mn > 0:
                        client_delay = model_size / r_mn
                        max_client_delay = max(max_client_delay, client_delay)
                        min_r_in_es = min(min_r_in_es, r[m, n])
                    break
        if max_client_delay > 0:
            es_client_delays_upload.append(max_client_delay)
        print(f"ES {es_nodes[es_idx]}: {len(clients)} clients, max_client_delay_upload={max_client_delay:.6f}s, min_r={min_r_in_es:.2e}")
    # Take max over all ES max delays, then multiply by k2
    client_es_total_upload = max(es_client_delays_upload) * k2 if es_client_delays_upload else 0.0

    es_ch_delays_upload = []
    for ch_idx, es_indices in C2.items():
        if not es_indices or cluster_heads[ch_idx] == -1:
            continue
        ch_idx_es = cluster_heads[ch_idx]
        max_es_ch_delay = 0.0
        for es_idx in es_indices:
            if es_idx != ch_idx_es:
                r_es_ch = r_es[es_idx, ch_idx_es] / 8
                if r_es_ch > 0:
                    es_ch_delay = model_size / r_es_ch
                    max_es_ch_delay = max(max_es_ch_delay, es_ch_delay)
        if max_es_ch_delay > 0:
            es_ch_delays_upload.append(max_es_ch_delay)
    # Take max over all cluster max delays
    es_ch_total_upload = max(es_ch_delays_upload) if es_ch_delays_upload else 0.0

    ch_cloud_delays_upload = []
    for ch_idx, ch_es_idx in cluster_heads.items():
        if ch_es_idx != -1 and B_n[ch_es_idx] > 0:
            B_n_adjusted = B_n[ch_es_idx] * 20
            ch_delay = model_size / (B_n_adjusted / 8)
            ch_cloud_delays_upload.append(ch_delay)
    max_ch_cloud_delay_upload = max(ch_cloud_delays_upload) if ch_cloud_delays_upload else 0.0
    ch_cloud_total_upload = max_ch_cloud_delay_upload * k3

    # Download delays
    es_client_delays_download = []
    for es_idx in range(num_ESs):
        clients = C1[es_idx]
        if not clients:
            print(f"ES {es_nodes[es_idx]}: 0 clients, max_client_delay_download=0.0s")
            continue
        max_client_delay = 0.0
        min_r_in_es = float('inf')
        for client_idx in clients:
            for m, n in assignments:
                if m == client_idx and n == es_idx:
                    r_mn = r[m, n] / 8
                    if r_mn > 0:
                        client_delay = (model_size / r_mn) * args.download_factor
                        max_client_delay = max(max_client_delay, client_delay)
                        min_r_in_es = min(min_r_in_es, r[m, n])
                    break
        if max_client_delay > 0:
            es_client_delays_download.append(max_client_delay)
        print(f"ES {es_nodes[es_idx]}: {len(clients)} clients, max_client_delay_download={max_client_delay:.6f}s, min_r={min_r_in_es:.2e}")
    # Take max over all ES max delays, then multiply by k2
    client_es_total_download = max(es_client_delays_download) * k2 if es_client_delays_download else 0.0

    es_ch_delays_download = []
    for ch_idx, es_indices in C2.items():
        if not es_indices or cluster_heads[ch_idx] == -1:
            continue
        ch_idx_es = cluster_heads[ch_idx]
        max_es_ch_delay = 0.0
        for es_idx in es_indices:
            if es_idx != ch_idx_es:
                r_es_ch = r_es[es_idx, ch_idx_es] / 8
                if r_es_ch > 0:
                    es_ch_delay = (model_size / r_es_ch) * args.download_factor
                    max_es_ch_delay = max(max_es_ch_delay, es_ch_delay)
        if max_es_ch_delay > 0:
            es_ch_delays_download.append(max_es_ch_delay)
    # Take max over all cluster max delays
    es_ch_total_download = max(es_ch_delays_download) if es_ch_delays_download else 0.0

    ch_cloud_delays_download = []
    for ch_idx, ch_es_idx in cluster_heads.items():
        if ch_es_idx != -1 and B_n[ch_es_idx] > 0:
            B_n_adjusted = B_n[ch_es_idx] * 20
            ch_delay = (model_size / (B_n_adjusted / 8)) * args.download_factor
            ch_cloud_delays_download.append(ch_delay)
    max_ch_cloud_delay_download = max(ch_cloud_delays_download) if ch_cloud_delays_download else 0.0
    ch_cloud_total_download = max_ch_cloud_delay_download * k3

    epoch_overhead_upload = client_es_total_upload + es_ch_total_upload + ch_cloud_total_upload
    epoch_overhead_download = client_es_total_download + es_ch_total_download + ch_cloud_total_download
    epoch_overhead = epoch_overhead_upload + epoch_overhead_download
    print(f"HFL Delays: Upload (Client-ES={client_es_total_upload:.4f}s, ES-CH={es_ch_total_upload:.4f}s, CH-Cloud={ch_cloud_total_upload:.4f}s), "
          f"Download (Client-ES={client_es_total_download:.4f}s, ES-CH={es_ch_total_download:.4f}s, CH-Cloud={ch_cloud_total_download:.4f}s)")
    return epoch_overhead, es_client_delays_upload, es_ch_delays_upload, max_ch_cloud_delay_upload, \
           es_client_delays_download, es_ch_delays_download, max_ch_cloud_delay_download

def calculate_comm_overhead_sfl(args, model_size, num_users, distance_matrix, client_nodes, W=1e6, P_tx=0.1, N_0=1e-9, alpha=3.5):
    client_cloud_delays_upload = []
    client_cloud_delays_download = []
    bandwidths = []
    for client_idx in range(num_users):
        avg_client_es_distance = np.mean([distance_matrix[client_idx, es_idx] for es_idx in range(distance_matrix.shape[1])])
        d_client_cloud = avg_client_es_distance * 10
        path_loss = (d_client_cloud / 1000) ** alpha
        SNR = P_tx / (N_0 * path_loss)
        B_client_cloud = W * np.log2(1 + SNR)
        if B_client_cloud <= 0:
            print(f"Warning: Client {client_idx} has zero or negative bandwidth, setting to 1e6")
            B_client_cloud = 1e6
        bandwidths.append(B_client_cloud)
        client_delay_upload = model_size / (B_client_cloud / 8)
        client_delay_download = (model_size / (B_client_cloud / 8)) * args.download_factor
        client_cloud_delays_upload.append(client_delay_upload)
        client_cloud_delays_download.append(client_delay_download)
        print(f"Client {client_nodes[client_idx]}: Distance={d_client_cloud:.2f}m, Bandwidth={B_client_cloud:.2e} bit/s, "
              f"Delay Upload={client_delay_upload:.6f}s, Delay Download={client_delay_download:.6f}s")
    epoch_overhead_upload = sum(client_cloud_delays_upload)
    epoch_overhead_download = sum(client_cloud_delays_download)
    epoch_overhead = epoch_overhead_upload + epoch_overhead_download
    max_client_delay_upload = max(client_cloud_delays_upload) if client_cloud_delays_upload else 0.0
    max_client_delay_download = max(client_cloud_delays_download) if client_cloud_delays_download else 0.0
    print(f"SFL Bandwidth stats: mean={np.mean(bandwidths):.2e}, max={np.max(bandwidths):.2e}, min={np.min(bandwidths):.2e}")
    print(f"SFL: {num_users} clients, Total delay (Upload={epoch_overhead_upload:.6f}s, Download={epoch_overhead_download:.6f}s)")
    return epoch_overhead, client_cloud_delays_upload, max_client_delay_upload, client_cloud_delays_download, max_client_delay_download

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
    for idx in dict_users:
        print(f"Client {idx} data size: {len(dict_users[idx])}")
    return dataset_train, dataset_test, dict_users

def build_model(args, dataset_train):
    img_size = dataset_train[0][0].shape
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
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

def get_A_from_assignments(assignments, num_users, num_ESs, es_nodes, active_es_nodes):
    A = np.zeros((num_users, num_ESs), dtype=int)
    print(f"Creating A matrix with shape: ({num_users}, {num_ESs}) from assignments")
    print(f"Assignments: {assignments}")
    es_index_map = {es_nodes[i]: i for i in range(len(es_nodes))}
    for m, n in assignments:
        if es_nodes[n] not in active_es_nodes:
            print(f"Warning: Assignment to inactive ES {es_nodes[n]} (index {n}), skipping")
            continue
        if n < 0 or n >= num_ESs:
            print(f"Warning: Invalid edge server index {n} for client {m}, skipping")
            continue
        if m < 0 or m >= num_users:
            print(f"Warning: Invalid client index {m}, skipping")
            continue
        A[m, n] = 1
    return A

def get_A(num_users, num_ESs, es_nodes, active_es_nodes, r, max_capacity=4):
    A = np.zeros((num_users, num_ESs), dtype=int)
    active_indices = [es_nodes.index(es) for es in active_es_nodes]
    loads = [0] * num_ESs  # Track load for each ES
    for i in range(num_users):
        if active_indices:
            # Filter ES with load < max_capacity
            available_indices = [idx for idx in active_indices if loads[idx] < max_capacity]
            if available_indices:
                # Prioritize low load ES for balance
                available_loads = [loads[idx] for idx in available_indices]
                min_load = min(available_loads)
                low_load_indices = [available_indices[j] for j in range(len(available_indices)) if available_loads[j] == min_load]
                random_index = random.choice(low_load_indices)
                A[i, random_index] = 1
                loads[random_index] += 1
                print(f"Client {i} assigned to ES {es_nodes[random_index]} (index {random_index}, r={r[i, random_index]:.2e})")
    print(f"HFL Random Loads: {loads}")
    print(f"HFL Random Load std: {np.std([loads[idx] for idx in active_indices]):.2f}")
    return A

def get_B(num_ESs, num_EHs, es_nodes, active_es_nodes):
    B = np.zeros((num_ESs, num_EHs), dtype=int)
    active_indices = [es_nodes.index(es) for es in active_es_nodes]
    for i in active_indices:
        random_index = np.random.randint(0, num_EHs)
        B[i, random_index] = 1
    return B

def build_hierarchy(A, B):
    num_users, num_ESs = A.shape
    _, num_EHs = B.shape
    C1 = {j: [] for j in range(num_ESs)}
    for i in range(num_users):
        for j in range(num_ESs):
            if A[i][j] == 1:
                C1[j].append(i)
    C2 = {k: [] for k in range(num_EHs)}
    for j in range(num_ESs):
        for k in range(num_EHs):
            if B[j][k] == 1:
                C2[k].append(j)
    return C1, C2

def train_client(args, user_idx, dataset_train, dict_users, w_input):
    if len(dict_users[user_idx]) == 0:
        print(f"Warning: Client {user_idx} has empty dataset, skipping")
        return user_idx, None, float('inf')

    if args.model == 'cnn' and args.dataset == 'cifar':
        local_net = CNNCifar(args=args)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        local_net = CNNMnist(args=args)
    else:
        exit('Error: unrecognized model in train_client')

    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[user_idx])
    if w_input is None:
        print(f"Warning: Client {user_idx} weights are None, using default weights")
        w_input = copy.deepcopy(local_net.state_dict())
    try:
        local_net.load_state_dict(w_input)
        w, loss = local.train(net=local_net.to(args.device))
    except Exception as e:
        print(f"Error training for client {user_idx}: {e}")
        w, loss = None, float('inf')

    print(f"CLIENT_{user_idx} END")
    return user_idx, copy.deepcopy(w), loss

def save_results_to_csv(results, final_results, filename):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['epoch', 'hfl_bipartite_test_acc', 'hfl_bipartite_test_loss',
                      'hfl_random_test_acc', 'hfl_random_test_loss',
                      'sfl_test_acc', 'sfl_test_loss',
                      'hfl_bipartite_overhead_upload', 'hfl_bipartite_overhead_download',
                      'hfl_random_overhead_upload', 'hfl_random_overhead_download',
                      'sfl_overhead_upload', 'sfl_overhead_download']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
        if final_results:
            writer.writerow({key: '' for key in fieldnames})
            writer.writerow({'epoch': 'Final Results'})
            for key, value in final_results.items():
                writer.writerow(
                    {'epoch': key,
                     'hfl_bipartite_test_acc': value if key == 'hfl_bipartite_test_acc' else '',
                     'hfl_bipartite_test_loss': value if key == 'hfl_bipartite_test_loss' else '',
                     'hfl_random_test_acc': value if key == 'hfl_random_test_acc' else '',
                     'hfl_random_test_loss': value if key == 'hfl_random_test_loss' else '',
                     'sfl_test_acc': value if key == 'sfl_test_acc' else '',
                     'sfl_test_loss': value if key == 'sfl_test_loss' else '',
                     'hfl_bipartite_overhead_upload': value if key == 'hfl_bipartite_overhead_upload' else '',
                     'hfl_bipartite_overhead_download': value if key == 'hfl_bipartite_overhead_download' else '',
                     'hfl_random_overhead_upload': value if key == 'hfl_random_overhead_upload' else '',
                     'hfl_random_overhead_download': value if key == 'hfl_random_overhead_download' else '',
                     'sfl_overhead_upload': value if key == 'sfl_overhead_upload' else '',
                     'sfl_overhead_download': value if key == 'sfl_overhead_download' else ''})

if __name__ == '__main__':
    mp.set_start_method('spawn')  # Set multiprocessing start method to 'spawn' for CUDA compatibility
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    num_processes = min(4, os.cpu_count() or 1)
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    args.epochs = 5
    args.local_ep = 5
    args.download_factor = 0.2  # Default download delay factor (download delay = upload delay * download_factor)

    bipartite_graph, client_nodes, es_nodes, distance_matrix, r, r_random, assignments, loads, B_n, r_es, pos, active_es_nodes = run_bandwidth_allocation()
    if bipartite_graph is None:
        exit("错误：无法构建二部图，请检查 graph-example/Ulaknet.graphml 文件。")

    num_ESs = len(active_es_nodes)
    print(f"Number of active edge servers (num_ESs): {num_ESs}")
    print(f"Active edge server nodes: {active_es_nodes}")
    print(f"Assignments: {assignments}")
    print(f"Loads: {loads}")

    args.num_users = len(client_nodes)
    dataset_train, dataset_test, dict_users = get_data(args)
    k2 = 2
    k3 = 2
    num_EHs = max(1, num_ESs // 3)

    max_capacity = max(1, int(len(client_nodes) / len(es_nodes)) + 1)
    print(f"Max capacity per ES: {max_capacity}")

    A_bipartite = get_A_from_assignments(assignments, args.num_users, len(es_nodes), es_nodes, active_es_nodes)
    A_random = get_A(args.num_users, len(es_nodes), es_nodes, active_es_nodes, r, max_capacity)

    B_bipartite = get_B(len(es_nodes), num_EHs, es_nodes, active_es_nodes)
    B_random = get_B(len(es_nodes), num_EHs, es_nodes, active_es_nodes)

    C1_bipartite, C2_bipartite = build_hierarchy(A_bipartite, B_bipartite)
    C1_random, C2_random = build_hierarchy(A_random, B_random)

    cluster_heads_bipartite = select_cluster_heads(C2_bipartite, es_nodes, r_es, loads, B_n)
    cluster_heads_random = select_cluster_heads(C2_random, es_nodes, r_es, loads, B_n)

    print("C1_bipartite (Client->ES):", C1_bipartite)
    print("C2_bipartite (ES->CH):", C2_bipartite)
    print("Cluster Heads Bipartite:", {k: es_nodes[v] if v != -1 else -1 for k, v in cluster_heads_bipartite.items()})
    print("C1_random (Client->ES):", C1_random)
    print("C2_random (ES->CH):", C2_random)
    print("Cluster Heads Random:", {k: es_nodes[v] if v != -1 else -1 for k, v in cluster_heads_random.items()})

    active_es_indices = [es_nodes.index(es) for es in active_es_nodes]
    plot_assigned_graph(bipartite_graph, client_nodes, es_nodes, assignments, cluster_heads_bipartite, C2_bipartite, active_es_indices)

    net_glob = build_model(args, dataset_train)
    net_glob.train()
    w_glob = net_glob.state_dict()

    args.model_size = sum(p.numel() * p.element_size() for p in net_glob.parameters())
    print(f"Model size: {args.model_size} bytes")

    if not os.path.exists('./results'):
        os.makedirs('./results')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f'./results/training_results_{timestamp}.csv'

    results_history = []
    final_results = {}
    total_comm_overhead_bipartite_upload = 0.0
    total_comm_overhead_bipartite_download = 0.0
    total_comm_overhead_random_upload = 0.0
    total_comm_overhead_random_download = 0.0
    total_comm_overhead_sfl_upload = 0.0
    total_comm_overhead_sfl_download = 0.0

    print("\n--- Testing initial global model ---")
    net_glob.eval()
    acc_init, loss_init = test_img(net_glob, dataset_test, args)
    print(f"Initial model - Test accuracy: {acc_init:.2f}%, Loss: {loss_init:.4f}")

    results_history.append({
        'epoch': -1,
        'hfl_bipartite_test_acc': acc_init,
        'hfl_bipartite_test_loss': loss_init,
        'hfl_random_test_acc': acc_init,
        'hfl_random_test_loss': loss_init,
        'sfl_test_acc': acc_init,
        'sfl_test_loss': loss_init,
        'hfl_bipartite_overhead_upload': 0.0,
        'hfl_bipartite_overhead_download': 0.0,
        'hfl_random_overhead_upload': 0.0,
        'hfl_random_overhead_download': 0.0,
        'sfl_overhead_upload': 0.0,
        'sfl_overhead_download': 0.0
    })
    save_results_to_csv(results_history, {}, csv_filename)

    net_glob_hfl_bipartite = copy.deepcopy(net_glob)
    w_glob_hfl_bipartite = net_glob_hfl_bipartite.state_dict()

    net_glob_hfl_random = copy.deepcopy(net_glob)
    w_glob_hfl_random = net_glob_hfl_random.state_dict()

    net_glob_sfl = copy.deepcopy(net_glob)
    w_glob_sfl = net_glob_sfl.state_dict()

    loss_train_hfl_bipartite = []
    loss_train_hfl_random = []
    loss_train_sfl = []
    loss_test_hfl_bipartite = []
    loss_test_hfl_random = []
    loss_test_sfl = []
    acc_test_hfl_bipartite = []
    acc_test_hfl_random = []
    acc_test_sfl = []

    early_stop = False

    for epoch in range(args.epochs):
        if early_stop:
            print(f"Accuracy reached 95% at epoch {epoch - 1}. Stopping training early.")
            break

        # HFL Bipartite
        EHs_ws_hfl_bipartite = [copy.deepcopy(w_glob_hfl_bipartite) for _ in range(num_EHs)]
        for t3 in range(k3):
            ESs_ws_input_hfl_bipartite = [None] * len(es_nodes)
            for EH_idx, ES_indices in C2_bipartite.items():
                if cluster_heads_bipartite[EH_idx] != -1:
                    ch_idx = cluster_heads_bipartite[EH_idx]
                    ESs_ws_input_hfl_bipartite[ch_idx] = copy.deepcopy(EHs_ws_hfl_bipartite[EH_idx])

            for t2 in range(k2):
                w_locals_input_hfl_bipartite = [copy.deepcopy(w_glob_hfl_bipartite) for _ in range(args.num_users)]
                for ES_idx, user_indices in C1_bipartite.items():
                    if ESs_ws_input_hfl_bipartite[ES_idx] is not None:
                        for user_idx in user_indices:
                            w_locals_input_hfl_bipartite[user_idx] = copy.deepcopy(ESs_ws_input_hfl_bipartite[ES_idx])

                w_locals_output_hfl_bipartite = [None] * args.num_users
                loss_locals_hfl_bipartite = []

                print(f"\n[Parallel Training HFL Bipartite] Starting training for {args.num_users} clients...")
                tasks_bipartite = []
                for user_idx in range(args.num_users):
                    if w_locals_input_hfl_bipartite[user_idx] is None:
                        print(f"Warning: Client {user_idx} has no HFL bipartite weights, skipping")
                        continue
                    task_args = (args, user_idx, dataset_train, dict_users, w_locals_input_hfl_bipartite[user_idx])
                    tasks_bipartite.append(task_args)
                print(f"Created {len(tasks_bipartite)} training tasks for HFL bipartite")

                with mp.Pool(processes=num_processes) as pool:
                    results_bipartite = pool.starmap(train_client, tqdm(tasks_bipartite,
                                                                        desc=f"Epoch {epoch}|{t3 + 1}|{t2 + 1} Training Clients HFL Bipartite"))

                print("Training HFL bipartite completed")
                for result in results_bipartite:
                    u_idx, w_h, l_h = result
                    if w_h is None or not isinstance(w_h, dict):
                        print(f"Warning: Client {u_idx} returned invalid HFL bipartite weights")
                        w_h = {}
                    w_locals_output_hfl_bipartite[u_idx] = w_h
                    loss_locals_hfl_bipartite.append(l_h)
                print(f"[Parallel Training HFL Bipartite] All {len(tasks_bipartite)} clients have finished training.")

                print(f'\nEpoch {epoch} | EH_R {t3 + 1}/{k3} | ES_R {t2 + 1}/{k2} | Starting aggregation HFL Bipartite')
                ESs_ws_input_hfl_bipartite = FedAvg_layered(w_locals_output_hfl_bipartite, C1_bipartite, dict_users)
                if ESs_ws_input_hfl_bipartite is None or all(w is None for w in ESs_ws_input_hfl_bipartite):
                    print("Error: HFL ES-layer aggregation failed for bipartite, no valid weights")
                    continue

                loss_avg_hfl_bipartite = sum([l for l in loss_locals_hfl_bipartite if l != float('inf')]) / max(1,len([l for l in loss_locals_hfl_bipartite if l != float('inf')]))
                loss_train_hfl_bipartite.append(loss_avg_hfl_bipartite)
                print(
                    f'Epoch {epoch} | EH_R {t3 + 1}/{k3} | ES_R {t2 + 1}/{k2} | HFL Bipartite Loss {loss_avg_hfl_bipartite:.4f}')

            EHs_ws_hfl_bipartite = FedAvg_layered(ESs_ws_input_hfl_bipartite, C2_bipartite)
            if EHs_ws_hfl_bipartite is None or all(w is None for w in EHs_ws_hfl_bipartite):
                print("Error: HFL ES-layer to EH-layer aggregation failed for bipartite")
                continue

        w_glob_hfl_bipartite = FedAvg([w for w in EHs_ws_hfl_bipartite if w is not None])
        if w_glob_hfl_bipartite is None:
            print("Error: HFL global model aggregation failed for bipartite")
            continue
        net_glob_hfl_bipartite.load_state_dict(w_glob_hfl_bipartite)

        epoch_overhead_bipartite, es_client_delays_bipartite_upload, es_ch_delays_bipartite_upload, max_ch_cloud_delay_bipartite_upload, \
        es_client_delays_bipartite_download, es_ch_delays_bipartite_download, max_ch_cloud_delay_bipartite_download = calculate_comm_overhead_hfl(
            args, assignments, r, r_es, B_n, C1_bipartite, C2_bipartite, cluster_heads_bipartite, k2, k3, args.model_size, es_nodes)
        total_comm_overhead_bipartite_upload += (epoch_overhead_bipartite - sum(es_client_delays_bipartite_download) - sum(es_ch_delays_bipartite_download) - max_ch_cloud_delay_bipartite_download)
        total_comm_overhead_bipartite_download += (sum(es_client_delays_bipartite_download) + sum(es_ch_delays_bipartite_download) + max_ch_cloud_delay_bipartite_download)
        print(f'Epoch {epoch} Communication Overhead HFL Bipartite: Total={epoch_overhead_bipartite:.4f}s '
              f'(Upload: Client-ES={sum(es_client_delays_bipartite_upload):.4f}s, ES-CH={sum(es_ch_delays_bipartite_upload):.4f}s, CH-Cloud={max_ch_cloud_delay_bipartite_upload:.4f}s; '
              f'Download: Client-ES={sum(es_client_delays_bipartite_download):.4f}s, ES-CH={sum(es_ch_delays_bipartite_download):.4f}s, CH-Cloud={max_ch_cloud_delay_bipartite_download:.4f}s)')

        net_glob_hfl_bipartite.eval()
        acc_hfl_bipartite, loss_hfl_bipartite = test_img(net_glob_hfl_bipartite, dataset_test, args)
        acc_test_hfl_bipartite.append(acc_hfl_bipartite)
        loss_test_hfl_bipartite.append(loss_hfl_bipartite)

        # HFL Random
        EHs_ws_hfl_random = [copy.deepcopy(w_glob_hfl_random) for _ in range(num_EHs)]
        random_assignments = [(i, np.argmax(A_random[i])) for i in range(args.num_users) if np.sum(A_random[i]) > 0]

        for t3 in range(k3):
            ESs_ws_input_hfl_random = [None] * len(es_nodes)
            for EH_idx, ES_indices in C2_random.items():
                if cluster_heads_random[EH_idx] != -1:
                    ch_idx = cluster_heads_random[EH_idx]
                    ESs_ws_input_hfl_random[ch_idx] = copy.deepcopy(EHs_ws_hfl_random[EH_idx])

            for t2 in range(k2):
                w_locals_input_hfl_random = [copy.deepcopy(w_glob_hfl_random) for _ in range(args.num_users)]
                for ES_idx, user_indices in C1_random.items():
                    if ESs_ws_input_hfl_random[ES_idx] is not None:
                        for user_idx in user_indices:
                            w_locals_input_hfl_random[user_idx] = copy.deepcopy(ESs_ws_input_hfl_random[ES_idx])

                w_locals_output_hfl_random = [None] * args.num_users
                loss_locals_hfl_random = []

                print(f"\n[Parallel Training HFL Random] Starting training for {args.num_users} clients...")
                tasks_random = []
                for user_idx in range(args.num_users):
                    if w_locals_input_hfl_random[user_idx] is None:
                        print(f"Warning: Client {user_idx} has no HFL random weights, skipping")
                        continue
                    task_args = (args, user_idx, dataset_train, dict_users, w_locals_input_hfl_random[user_idx])
                    tasks_random.append(task_args)
                print(f"Created {len(tasks_random)} training tasks for HFL random")

                with mp.Pool(processes=num_processes) as pool:
                    results_random = pool.starmap(train_client, tqdm(tasks_random, desc=f"Epoch {epoch}|{t3 + 1}|{t2 + 1} Training Clients HFL Random"))

                print("Training HFL random completed")
                for result in results_random:
                    u_idx, w_h, l_h = result
                    if w_h is None or not isinstance(w_h, dict):
                        print(f"Warning: Client {u_idx} returned invalid HFL random weights")
                        w_h = {}
                    w_locals_output_hfl_random[u_idx] = w_h
                    loss_locals_hfl_random.append(l_h)
                print(f"[Parallel Training HFL Random] All {len(tasks_random)} clients have finished training.")

                print(f'\nEpoch {epoch} | EH_R {t3 + 1}/{k3} | ES_R {t2 + 1}/{k2} | Starting aggregation HFL Random')
                ESs_ws_input_hfl_random = FedAvg_layered(w_locals_output_hfl_random, C1_random, dict_users)
                if ESs_ws_input_hfl_random is None or all(w is None for w in ESs_ws_input_hfl_random):
                    print("Error: HFL ES-layer aggregation failed for random, no valid weights")
                    continue

                loss_avg_hfl_random = sum([l for l in loss_locals_hfl_random if l != float('inf')]) / max(1, len([l for l in loss_locals_hfl_random if l != float('inf')]))
                loss_train_hfl_random.append(loss_avg_hfl_random)
                print(f'Epoch {epoch} | EH_R {t3 + 1}/{k3} | ES_R {t2 + 1}/{k2} | HFL Random Loss {loss_avg_hfl_random:.4f}')

            EHs_ws_hfl_random = FedAvg_layered(ESs_ws_input_hfl_random, C2_random)
            if EHs_ws_hfl_random is None or all(w is None for w in EHs_ws_hfl_random):
                print("Error: HFL ES-layer to EH-layer aggregation failed for random")
                continue

        w_glob_hfl_random = FedAvg([w for w in EHs_ws_hfl_random if w is not None])
        if w_glob_hfl_random is None:
            print("Error: HFL global model aggregation failed for random")
            continue
        net_glob_hfl_random.load_state_dict(w_glob_hfl_random)

        epoch_overhead_random, es_client_delays_random_upload, es_ch_delays_random_upload, max_ch_cloud_delay_random_upload, \
        es_client_delays_random_download, es_ch_delays_random_download, max_ch_cloud_delay_random_download = calculate_comm_overhead_hfl(
            args, random_assignments, r, r_es, B_n, C1_random, C2_random, cluster_heads_random, k2, k3, args.model_size, es_nodes)
        total_comm_overhead_random_upload += (epoch_overhead_random - sum(es_client_delays_random_download) - sum(es_ch_delays_random_download) - max_ch_cloud_delay_random_download)
        total_comm_overhead_random_download += (sum(es_client_delays_random_download) + sum(es_ch_delays_random_download) + max_ch_cloud_delay_random_download)
        print(f'Epoch {epoch} Communication Overhead HFL Random: Total={epoch_overhead_random:.4f}s '
              f'(Upload: Client-ES={sum(es_client_delays_random_upload):.4f}s, ES-CH={sum(es_ch_delays_random_upload):.4f}s, CH-Cloud={max_ch_cloud_delay_random_upload:.4f}s; '
              f'Download: Client-ES={sum(es_client_delays_random_download):.4f}s, ES-CH={sum(es_ch_delays_random_download):.4f}s, CH-Cloud={max_ch_cloud_delay_random_download:.4f}s)')

        net_glob_hfl_random.eval()
        acc_hfl_random, loss_hfl_random = test_img(net_glob_hfl_random, dataset_test, args)
        acc_test_hfl_random.append(acc_hfl_random)
        loss_test_hfl_random.append(loss_hfl_random)

        # SFL
        w_locals_sfl = [None] * args.num_users
        loss_locals_sfl = []

        print(f"\n[Parallel Training SFL] Starting training for {args.num_users} clients...")
        tasks_sfl = []
        for user_idx in range(args.num_users):
            task_args = (args, user_idx, dataset_train, dict_users, copy.deepcopy(w_glob_sfl))
            tasks_sfl.append(task_args)
        print(f"Created {len(tasks_sfl)} training tasks for SFL")

        with mp.Pool(processes=num_processes) as pool:
            results_sfl = pool.starmap(train_client, tqdm(tasks_sfl, desc=f"Epoch {epoch} Training Clients SFL"))

        print("Training SFL completed")
        for result in results_sfl:
            u_idx, w, l = result
            if w is None or not isinstance(w, dict):
                print(f"Warning: Client {u_idx} returned invalid SFL weights")
                w = {}
            w_locals_sfl[u_idx] = w
            loss_locals_sfl.append(l)
        print(f"[Parallel Training SFL] All {len(tasks_sfl)} clients have finished training.")

        print(f'\nEpoch {epoch} | Starting aggregation SFL')
        w_glob_sfl = FedAvg(w_locals_sfl)
        if w_glob_sfl is None:
            print("Error: SFL global model aggregation failed")
            continue
        net_glob_sfl.load_state_dict(w_glob_sfl)

        epoch_overhead_sfl, client_delays_sfl_upload, client_cloud_delay_sfl_upload, client_delays_sfl_download, client_cloud_delay_sfl_download = calculate_comm_overhead_sfl(
            args, args.model_size, args.num_users, distance_matrix, client_nodes)
        total_comm_overhead_sfl_upload += (epoch_overhead_sfl - sum(client_delays_sfl_download))
        total_comm_overhead_sfl_download += sum(client_delays_sfl_download)
        print(f'Epoch {epoch} Communication Overhead SFL: Total={epoch_overhead_sfl:.4f}s '
              f'(Upload: Max Client-Cloud={client_cloud_delay_sfl_upload:.4f}s; Download: Max Client-Cloud={client_cloud_delay_sfl_download:.4f}s)')

        net_glob_sfl.eval()
        acc_sfl, loss_sfl = test_img(net_glob_sfl, dataset_test, args)
        acc_test_sfl.append(acc_sfl)
        loss_test_sfl.append(loss_sfl)

        loss_avg_sfl = sum([l for l in loss_locals_sfl if l != float('inf')]) / max(1, len([l for l in loss_locals_sfl if l != float('inf')]))
        loss_train_sfl.append(loss_avg_sfl)
        print(f'Epoch {epoch} | SFL Loss {loss_avg_sfl:.4f}')

        results_history.append({
            'epoch': epoch,
            'hfl_bipartite_test_acc': acc_hfl_bipartite,
            'hfl_bipartite_test_loss': loss_hfl_bipartite,
            'hfl_random_test_acc': acc_hfl_random,
            'hfl_random_test_loss': loss_hfl_random,
            'sfl_test_acc': acc_sfl,
            'sfl_test_loss': loss_sfl,
            'hfl_bipartite_overhead_upload': total_comm_overhead_bipartite_upload,
            'hfl_bipartite_overhead_download': total_comm_overhead_bipartite_download,
            'hfl_random_overhead_upload': total_comm_overhead_random_upload,
            'hfl_random_overhead_download': total_comm_overhead_random_download,
            'sfl_overhead_upload': total_comm_overhead_sfl_upload,
            'sfl_overhead_download': total_comm_overhead_sfl_download
        })
        save_results_to_csv(results_history, {}, csv_filename)

        print(f'Epoch {epoch} [END OF EPOCH TEST] | HFL Bipartite Acc: {acc_hfl_bipartite:.2f}%, Loss: {loss_hfl_bipartite:.4f} | '
              f'HFL Random Acc: {acc_hfl_random:.2f}%, Loss: {loss_hfl_random:.4f} | '
              f'SFL Acc: {acc_sfl:.2f}%, Loss: {loss_sfl:.4f}')

        if acc_hfl_bipartite >= 95.0 or acc_hfl_random >= 95.0 or acc_sfl >= 95.0:
            print(f"Accuracy reached 95% at epoch {epoch}. Stopping training early.")
            early_stop = True

        net_glob_hfl_bipartite.train()
        net_glob_hfl_random.train()
        net_glob_sfl.train()

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(loss_train_hfl_bipartite)), loss_train_hfl_bipartite, 'b-', label='HFL Bipartite Train Loss', linewidth=2)
    plt.plot(range(len(loss_train_hfl_random)), loss_train_hfl_random, 'r--', label='HFL Random Train Loss', linewidth=2)
    plt.plot(range(len(loss_train_sfl)), loss_train_sfl, 'g:', label='SFL Train Loss', linewidth=2)
    plt.ylabel('Train Loss', fontsize=12)
    plt.xlabel('Communication Rounds', fontsize=12)
    plt.legend(fontsize=10)
    plt.title('Train Loss Comparison', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('./save/compare_train_loss.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(loss_test_hfl_bipartite)), loss_test_hfl_bipartite, 'b-', label='HFL Bipartite Test Loss', linewidth=2)
    plt.plot(range(len(loss_test_hfl_random)), loss_test_hfl_random, 'r--', label='HFL Random Test Loss', linewidth=2)
    plt.plot(range(len(loss_test_sfl)), loss_test_sfl, 'g:', label='SFL Test Loss', linewidth=2)
    plt.ylabel('Test Loss', fontsize=12)
    plt.xlabel('Communication Rounds', fontsize=12)
    plt.legend(fontsize=10)
    plt.title('Test Loss Comparison', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('./save/compare_test_loss.png')
    plt.close()

    print("\n--- Final Model Evaluation ---")
    net_glob_hfl_bipartite.eval()
    acc_train_hfl_bipartite, loss_train_hfl_bipartite = test_img(net_glob_hfl_bipartite, dataset_train, args)
    acc_test_hfl_bipartite, loss_test_hfl_bipartite = test_img(net_glob_hfl_bipartite, dataset_test, args)
    print(f"HFL Bipartite Model - Training accuracy: {acc_train_hfl_bipartite:.2f}%")
    print(f"HFL Bipartite Model - Testing accuracy: {acc_test_hfl_bipartite:.2f}%")

    net_glob_hfl_random.eval()
    acc_train_hfl_random, loss_train_hfl_random = test_img(net_glob_hfl_random, dataset_train, args)
    acc_test_hfl_random, loss_test_hfl_random = test_img(net_glob_hfl_random, dataset_test, args)
    print(f"HFL Random Model - Training accuracy: {acc_train_hfl_random:.2f}%")
    print(f"HFL Random Model - Testing accuracy: {acc_test_hfl_random:.2f}%")

    net_glob_sfl.eval()
    acc_train_sfl, loss_train_sfl = test_img(net_glob_sfl, dataset_train, args)
    acc_test_sfl, loss_test_sfl = test_img(net_glob_sfl, dataset_test, args)
    print(f"SFL Model - Training accuracy: {acc_train_sfl:.2f}%")
    print(f"SFL Model - Testing accuracy: {acc_test_sfl:.2f}%")

    final_results = {
        'hfl_bipartite_train_acc': acc_train_hfl_bipartite,
        'hfl_bipartite_train_loss': loss_train_hfl_bipartite,
        'hfl_bipartite_test_acc': acc_test_hfl_bipartite,
        'hfl_bipartite_test_loss': loss_test_hfl_bipartite,
        'hfl_random_train_acc': acc_train_hfl_random,
        'hfl_random_train_loss': loss_train_hfl_random,
        'hfl_random_test_acc': acc_test_hfl_random,
        'hfl_random_test_loss': loss_test_hfl_random,
        'sfl_train_acc': acc_train_sfl,
        'sfl_train_loss': loss_train_sfl,
        'sfl_test_acc': acc_test_sfl,
        'sfl_test_loss': loss_test_sfl,
        'hfl_bipartite_overhead_upload': total_comm_overhead_bipartite_upload,
        'hfl_bipartite_overhead_download': total_comm_overhead_bipartite_download,
        'hfl_random_overhead_upload': total_comm_overhead_random_upload,
        'hfl_random_overhead_download': total_comm_overhead_random_download,
        'sfl_overhead_upload': total_comm_overhead_sfl_upload,
        'sfl_overhead_download': total_comm_overhead_sfl_download
    }

    save_results_to_csv(results_history, final_results, csv_filename)
    print(f"\nAll results saved to {csv_filename}")
    print(f"\nTotal Communication Overhead HFL Bipartite over {args.epochs} epochs: "
          f"Upload={total_comm_overhead_bipartite_upload:.4f}s, Download={total_comm_overhead_bipartite_download:.4f}s")
    print(f"\nTotal Communication Overhead HFL Random over {args.epochs} epochs: "
          f"Upload={total_comm_overhead_random_upload:.4f}s, Download={total_comm_overhead_random_download:.4f}s")
    print(f"\nTotal Communication Overhead SFL over {args.epochs} epochs: "
          f"Upload={total_comm_overhead_sfl_upload:.4f}s, Download={total_comm_overhead_sfl_download:.4f}s")