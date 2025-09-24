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
import multiprocessing as mp
import csv
from datetime import datetime
import networkx as nx
import argparse
from bipartite_bandwidth import run_bandwidth_allocation, plot_assigned_graph

def get_dataset(args):
    if args.dataset == 'mnist':
        dataset_train = None
        dataset_test = None
    else:
        dataset_train = None
        dataset_test = None
    return dataset_train, dataset_test

def get_model(args):
    if args.model == 'cnn' and args.dataset == 'mnist':
        net_glob_sfl = torch.nn.Module()
        net_glob_hfl_bipartite = torch.nn.Module()
        net_glob_hfl_random = torch.nn.Module()
    else:
        net_glob_sfl = torch.nn.Module()
        net_glob_hfl_bipartite = torch.nn.Module()
        net_glob_hfl_random = torch.nn.Module()
    return net_glob_sfl, net_glob_hfl_bipartite, net_glob_hfl_random

def calculate_comm_overhead_sfl(args, model_size, num_users, distance_matrix, client_nodes, W=1e6, P_tx=0.1, N_0=1e-9, alpha=3.5):
    client_cloud_delays_upload = []
    client_cloud_delays_download = []
    bandwidths = []
    for client_idx in range(num_users):
        max_client_es_distance = np.max(distance_matrix[client_idx, :])
        d_client_cloud = (max_client_es_distance / 1000) * 15
        path_loss = (d_client_cloud / 1000) ** alpha
        SNR = P_tx / (N_0 * path_loss * (1 + np.random.uniform(0, 0.2)))
        B_client_cloud = W * np.log2(1 + SNR)
        if B_client_cloud <= 0:
            B_client_cloud = 1e6
        bandwidths.append(B_client_cloud)
        client_delay_upload = model_size / (B_client_cloud / 8)
        client_delay_download = client_delay_upload * args.download_factor
        client_cloud_delays_upload.append(client_delay_upload)
        client_cloud_delays_download.append(client_delay_download)
        print(f"Client {client_nodes[client_idx]}: Distance={d_client_cloud:.2f}km, Bandwidth={B_client_cloud:.2e} bit/s, "
              f"Delay Upload={client_delay_upload:.6f}s, Delay Download={client_delay_download:.6f}s")
    epoch_overhead_upload = sum(client_cloud_delays_upload)
    epoch_overhead_download = sum(client_cloud_delays_download)
    max_client_delay_upload = max(client_cloud_delays_upload) if client_cloud_delays_upload else 0.0
    max_client_delay_download = max(client_cloud_delays_download) if client_cloud_delays_download else 0.0
    print(f"SFL Bandwidth stats: mean={np.mean(bandwidths):.2e}, max={np.max(bandwidths):.2e}, min={np.min(bandwidths):.2e}")
    print(f"SFL: {num_users} clients, Total delay (Upload={epoch_overhead_upload:.6f}s, Download={epoch_overhead_download:.6f}s), "
          f"Max delay (Upload={max_client_delay_upload:.6f}s, Download={max_client_delay_download:.6f}s)")
    return epoch_overhead_upload + epoch_overhead_download, client_cloud_delays_upload, max_client_delay_upload, client_cloud_delays_download, max_client_delay_download

def calculate_comm_overhead_hfl_bipartite(args, assignments, r, r_es, B_n, C1, C2, cluster_heads, k2, k3, model_size, es_nodes):
    num_ESs = len(C1)
    es_client_delays_upload = []
    for es_idx in range(num_ESs):
        clients = C1[es_idx]
        if not clients:
            print(f"ES {es_nodes[es_idx]}: 0 clients, max_client_delay_upload=0.0s")
            continue
        max_client_delay = 0.0
        for client_idx in clients:
            for m, n in assignments:
                if m == client_idx and n == es_idx:
                    r_mn = r[m, n] / 8
                    if r_mn > 0:
                        client_delay = model_size / r_mn
                        max_client_delay = max(max_client_delay, client_delay)
                        print(f"Client {m} to ES {n}: r={r[m, n]:.2e}, delay={client_delay:.6f}s")
                    break
        if max_client_delay > 0:
            es_client_delays_upload.append(max_client_delay)
        print(f"ES {es_nodes[es_idx]}: {len(clients)} clients, max_client_delay_upload={max_client_delay:.6f}s")
    client_es_total_upload = max(es_client_delays_upload) * k2 if es_client_delays_upload else 0.0
    es_ch_total_upload = sum(model_size / (r_es[es_idx, cluster_heads[es_idx]] / 8) for es_idx in range(num_ESs) if cluster_heads[es_idx] != -1 and r_es[es_idx, cluster_heads[es_idx]] > 0) * k3
    ch_cloud_total_upload = sum(model_size / (B_n[cluster_heads[es_idx]] / 8) for es_idx in range(num_ESs) if cluster_heads[es_idx] != -1 and B_n[cluster_heads[es_idx]] > 0) if any(cluster_heads[es_idx] != -1 for es_idx in range(num_ESs)) else 0.0
    client_es_total_download = client_es_total_upload * args.download_factor
    es_ch_total_download = es_ch_total_upload * args.download_factor
    ch_cloud_total_download = ch_cloud_total_upload * args.download_factor
    total_overhead = (client_es_total_upload + es_ch_total_upload + ch_cloud_total_upload +
                     client_es_total_download + es_ch_total_download + ch_cloud_total_download)
    print(f"HFL Bipartite: Total delay={total_overhead:.6f}s (Upload: Client-ES={client_es_total_upload:.6f}s, "
          f"ES-CH={es_ch_total_upload:.6f}s, CH-Cloud={ch_cloud_total_upload:.6f}s; Download: Client-ES={client_es_total_download:.6f}s, "
          f"ES-CH={es_ch_total_download:.6f}s, CH-Cloud={ch_cloud_total_download:.6f}s)")
    return total_overhead, client_es_total_upload, es_ch_total_upload, ch_cloud_total_upload, client_es_total_download, es_ch_total_download, ch_cloud_total_download

def calculate_comm_overhead_hfl_random(args, assignments, r_initial, r_es, B_n, C1, C2, cluster_heads, k2, k3, model_size, es_nodes, client_nodes):
    num_ESs = len(es_nodes)
    num_users = len(client_nodes)
    random_assignments = [(m, random.randint(0, num_ESs - 1)) for m in range(num_users)]
    random_C1 = [[] for _ in range(num_ESs)]
    for m, n in random_assignments:
        random_C1[n].append(m)
    es_client_delays_upload = []
    for es_idx in range(num_ESs):
        clients = random_C1[es_idx]
        if not clients:
            print(f"ES {es_nodes[es_idx]}: 0 clients, max_client_delay_upload=0.0s")
            continue
        max_client_delay = 0.0
        for client_idx in clients:
            for m, n in random_assignments:
                if m == client_idx and n == es_idx:
                    r_mn = r_initial[m, n] / 8
                    if r_mn > 0:
                        client_delay = model_size / r_mn
                        max_client_delay = max(max_client_delay, client_delay)
                        print(f"Client {m} to ES {n}: r={r_initial[m, n]:.2e}, delay={client_delay:.6f}s")
                    break
        if max_client_delay > 0:
            es_client_delays_upload.append(max_client_delay)
        print(f"ES {es_nodes[es_idx]}: {len(clients)} clients, max_client_delay_upload={max_client_delay:.6f}s")
    client_es_total_upload = max(es_client_delays_upload) * k2 if es_client_delays_upload else 0.0
    es_ch_total_upload = sum(model_size / (r_es[es_idx, cluster_heads[es_idx]] / 8) for es_idx in range(num_ESs) if cluster_heads[es_idx] != -1 and r_es[es_idx, cluster_heads[es_idx]] > 0) * k3
    ch_cloud_total_upload = sum(model_size / (B_n[cluster_heads[es_idx]] / 8) for es_idx in range(num_ESs) if cluster_heads[es_idx] != -1 and B_n[cluster_heads[es_idx]] > 0) if any(cluster_heads[es_idx] != -1 for es_idx in range(num_ESs)) else 0.0
    client_es_total_download = client_es_total_upload * args.download_factor
    es_ch_total_download = es_ch_total_upload * args.download_factor
    ch_cloud_total_download = ch_cloud_total_upload * args.download_factor
    total_overhead = (client_es_total_upload + es_ch_total_upload + ch_cloud_total_upload +
                     client_es_total_download + es_ch_total_download + ch_cloud_total_download)
    print(f"HFL Random: Total delay={total_overhead:.6f}s (Upload: Client-ES={client_es_total_upload:.6f}s, "
          f"ES-CH={es_ch_total_upload:.6f}s, CH-Cloud={ch_cloud_total_upload:.6f}s; Download: Client-ES={client_es_total_download:.6f}s, "
          f"ES-CH={es_ch_total_download:.6f}s, CH-Cloud={ch_cloud_total_download:.6f}s)")
    return total_overhead, client_es_total_upload, es_ch_total_upload, ch_cloud_total_upload, client_es_total_download, es_ch_total_download, ch_cloud_total_download

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist', help='dataset name')
    parser.add_argument('--epoch', type=int, default=0, help='number of epochs')
    parser.add_argument('--num_channel', type=int, default=1, help='number of channels')
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--all_clients', action='store_true', help='use all clients')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--download_factor', type=float, default=0.2, help='download factor')
    args = parser.parse_args()

    dataset_train, dataset_test = get_dataset(args)
    net_glob_sfl, net_glob_hfl_bipartite, net_glob_hfl_random = get_model(args)

    bipartite_graph, client_nodes, es_nodes, distance_matrix, r, r_random, assignments, loads, B_n, r_es, pos, active_es_nodes, r_initial, association_matrix, active_es_distance_matrix, cloud_pos, r_es_to_cloud = run_bandwidth_allocation()
    if bipartite_graph is None:
        print("Failed to build bipartite graph, exiting.")
        return

    num_users = len(client_nodes)

    C1 = [[] for _ in range(len(es_nodes))]
    C2 = {}
    cluster_heads = {}
    for m, n in assignments:
        C1[n].append(m)
    for i in range(len(es_nodes)):
        cluster_heads[i] = i if i % 2 == 0 else i - 1 if i > 0 else 0
        if cluster_heads[i] not in C2:
            C2[cluster_heads[i]] = []
        C2[cluster_heads[i]].append(i)

    model_size = 1e7
    k2, k3 = 2.0, 1.5

    total_comm_overhead_sfl, client_cloud_delays_upload, _, client_cloud_delays_download, _ = calculate_comm_overhead_sfl(
        args, model_size, num_users, distance_matrix, client_nodes)
    total_comm_overhead_sfl_upload = sum(client_cloud_delays_upload)
    total_comm_overhead_sfl_download = sum(client_cloud_delays_download)

    total_comm_overhead_bipartite, client_es_total_upload, es_ch_total_upload, ch_cloud_total_upload, client_es_total_download, es_ch_total_download, ch_cloud_total_download = calculate_comm_overhead_hfl_bipartite(
        args, assignments, r, r_es, B_n, C1, C2, cluster_heads, k2, k3, model_size, es_nodes)
    total_comm_overhead_bipartite_upload = client_es_total_upload + es_ch_total_upload + ch_cloud_total_upload
    total_comm_overhead_bipartite_download = client_es_total_download + es_ch_total_download + ch_cloud_total_download

    total_comm_overhead_random, client_es_total_upload, es_ch_total_upload, ch_cloud_total_upload, client_es_total_download, es_ch_total_download, ch_cloud_total_download = calculate_comm_overhead_hfl_random(
        args, assignments, r_initial, r_es, B_n, C1, C2, cluster_heads, k2, k3, model_size, es_nodes, client_nodes)
    total_comm_overhead_random_upload = client_es_total_upload + es_ch_total_upload + ch_cloud_total_upload
    total_comm_overhead_random_download = client_es_total_download + es_ch_total_download + ch_cloud_total_download

    summarize_results(net_glob_hfl_bipartite, net_glob_hfl_random, net_glob_sfl, dataset_train, dataset_test, args,
                     total_comm_overhead_bipartite_upload, total_comm_overhead_bipartite_download,
                     total_comm_overhead_random_upload, total_comm_overhead_random_download,
                     total_comm_overhead_sfl_upload, total_comm_overhead_sfl_download)

    print("=== Detailed Allocation Information ===")
    print(f"Client Nodes: {client_nodes}")
    print(f"Edge Server Nodes: {es_nodes}")
    print(f"Distance Matrix:\n{distance_matrix}")
    print(f"Transmission Rates (r):\n{r}")
    print(f"Random Transmission Rates (r_random):\n{r_random}")
    print(f"ES-to-ES Transmission Rates (r_es):\n{r_es}")
    print(f"Edge-to-Cloud Bandwidth (B_n): {B_n}")
    print(f"Assignments: {assignments}")
    print(f"Loads: {loads}")
    print(f"Cluster Heads: {cluster_heads}")
    print(f"C1 (Client to ES): {C1}")
    print(f"C2 (ES to CH): {C2}")
    print(f"Active ES Nodes: {active_es_nodes}")
    print("=== End of Detailed Output ===")

if __name__ == '__main__':
    main()