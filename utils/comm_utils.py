#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import numpy as np
from scipy.optimize import linear_sum_assignment
import random

def build_bipartite_graph(M, N, max_client_es_distance=1000, sigma=1.0):
    """
    Build initial bipartite graph with random distances and bandwidths.
    
    Args:
        M (int): Number of clients
        N (int): Number of edge servers (ES)
        max_client_es_distance (float): Maximum distance between client and ES (meters)
        sigma (float): Standard deviation for bandwidth variation
    
    Returns:
        tuple: (r_initial, distances)
            - r_initial: Initial bandwidth matrix (M x N)
            - distances: Distance matrix (M x N)
    """
    np.random.seed(42)
    random.seed(42)
    
    # Generate random distances between clients and ES
    distances = np.random.uniform(100, max_client_es_distance, size=(M, N))
    
    # Calculate initial bandwidths based on distances
    r_initial = np.zeros((M, N))
    for m in range(M):
        for n in range(N):
            d = distances[m, n] / 1000  # Convert to km
            path_loss = (d ** 3.5)  # Path loss with alpha=3.5
            SNR = 1.0 / (1e-9 * path_loss * (1 + np.random.normal(0, sigma)))
            r_initial[m, n] = 1e6 * np.log2(1 + SNR)  # Bandwidth in bits/s
    
    return r_initial, distances

def establish_communication_channels(r_initial, max_capacity=None):
    """
    Allocate bandwidth using bipartite optimization with load balancing.
    
    Args:
        r_initial (np.ndarray): Initial bandwidth matrix (M x N)
        max_capacity (int, optional): Maximum number of clients per ES
    
    Returns:
        tuple: (r, assignments, loads)
            - r: Optimized bandwidth matrix
            - assignments: List of (client, ES) assignments
            - loads: Number of clients per ES
    """
    M, N = r_initial.shape
    if max_capacity is None:
        max_capacity = M // N + 1
    
    # Step 1: Hungarian algorithm to maximize total bandwidth
    row_ind, col_ind = linear_sum_assignment(-r_initial)  # Maximize by negating
    assignments = list(zip(row_ind, col_ind))
    r = np.zeros_like(r_initial)
    for m, n in assignments:
        r[m, n] = r_initial[m, n]
    
    # Step 2: Simulated Annealing to maximize minimum bandwidth
    temp = 1000
    cooling_rate = 0.995
    iterations = 5000
    
    best_r = r.copy()
    best_min_rate = np.min(r[r > 0]) if np.any(r > 0) else 0
    
    for _ in range(iterations):
        r_new = r.copy()
        m1, m2 = np.random.choice(M, 2, replace=False)
        n1 = np.argmax(r[m1])
        n2 = np.argmax(r[m2])
        if n1 != n2:
            r_new[m1, n1], r_new[m1, n2] = 0, r_initial[m1, n2]
            r_new[m2, n2], r_new[m2, n1] = 0, r_initial[m2, n1]
            
            loads = np.sum(r_new > 0, axis=0)
            if np.all(loads <= max_capacity):
                min_rate = np.min(r_new[r_new > 0]) if np.any(r_new > 0) else 0
                if min_rate > best_min_rate or np.random.rand() < np.exp((min_rate - best_min_rate) / temp):
                    r = r_new
                    if min_rate > best_min_rate:
                        best_min_rate = min_rate
                        best_r = r_new.copy()
        
        temp *= cooling_rate
    
    # Step 3: Reallocate if any ES is overloaded
    loads = np.sum(best_r > 0, axis=0)
    while np.any(loads > max_capacity):
        for n in range(N):
            if loads[n] > max_capacity:
                clients = [m for m in range(M) if best_r[m, n] > 0]
                m = random.choice(clients)
                best_r[m, n] = 0
                available_es = [i for i in range(N) if loads[i] < max_capacity]
                if available_es:
                    new_n = random.choice(available_es)
                    best_r[m, new_n] = r_initial[m, new_n]
        loads = np.sum(best_r > 0, axis=0)
    
    assignments = [(m, n) for m in range(M) for n in range(N) if best_r[m, n] > 0]
    return best_r, assignments, loads

def run_bandwidth_allocation(M, N, max_client_es_distance=1000, sigma=1.0):
    """
    Run the full bandwidth allocation process.
    
    Args:
        M (int): Number of clients
        N (int): Number of edge servers
        max_client_es_distance (float): Maximum distance (meters)
        sigma (float): Standard deviation for bandwidth variation
    
    Returns:
        tuple: (C1, r, assignments, loads)
            - C1: Dictionary mapping ES to list of clients
            - r: Optimized bandwidth matrix
            - assignments: List of (client, ES) assignments
            - loads: Number of clients per ES
    """
    r_initial, distances = build_bipartite_graph(M, N, max_client_es_distance, sigma)
    r, assignments, loads = establish_communication_channels(r_initial, max_capacity=M // N + 1)
    
    C1 = {n: [] for n in range(N)}
    for m, n in assignments:
        C1[n].append(m)
    
    return C1, r, assignments, loads

def calculate_comm_overhead_sfl(M, max_client_es_distance=1000, model_size=1e7, download_factor=0.2, rounds=1, distance_factor=15):
    """
    Calculate communication overhead for SFL (Server-based FL).
    Uses maximum client-to-cloud delay multiplied by rounds.
    
    Args:
        M (int): Number of clients
        max_client_es_distance (float): Maximum client-ES distance (meters)
        model_size (float): Model size in bits
        download_factor (float): Download delay as fraction of upload
        rounds (int): Number of communication rounds
        distance_factor (float): Factor to amplify client-cloud distance
    
    Returns:
        float: Total communication overhead (seconds)
    """
    np.random.seed(42)
    alpha = 3.5
    P_tx = 1.0
    N_0 = 1e-9
    W = 1e6
    
    delays = []
    for i in range(M):
        d_client_cloud = (max_client_es_distance / 1000) * distance_factor  # km
        path_loss = (d_client_cloud ** alpha)
        noise_factor = 1 + np.random.normal(0, 1.0)
        SNR = P_tx / (N_0 * path_loss * noise_factor)
        B_client_cloud = W * np.log2(1 + SNR)  # bits/s
        upload_delay = model_size / (B_client_cloud / 8)  # seconds
        delays.append(upload_delay)
    
    max_delay = max(delays) if delays else 0
    download_delay = max_delay * download_factor
    total_overhead = (max_delay + download_delay) * rounds
    return total_overhead

def calculate_comm_overhead_hfl_random(C1, C2, cluster_heads, num_users, num_ESs, r_mn, B_n, model_size=1e7, download_factor=0.2, k2=2.0, k3=1.5):
    """
    Calculate communication overhead for HFL with random allocation.
    
    Args:
        C1 (dict): ES to client assignments
        C2 (dict): CH to ES assignments
        cluster_heads (list): Cluster head index for each ES
        num_users (int): Number of clients
        num_ESs (int): Number of edge servers
        r_mn (np.ndarray): Client-ES bandwidth matrix
        B_n (np.ndarray): ES-cloud bandwidth array
        model_size (float): Model size in bits
        download_factor (float): Download delay as fraction of upload
        k2 (float): ES aggregation rounds
        k3 (float): CH aggregation rounds
    
    Returns:
        float: Total communication overhead (seconds)
    """
    # Client to ES
    max_delays = []
    for n in range(num_ESs):
        clients = C1.get(n, [])
        if clients:
            delays = [model_size / (r_mn[m, n] / 8) for m in clients if r_mn[m, n] > 0]
            max_delays.append(max(delays) if delays else 0)
    
    client_es_overhead = max(max_delays) * k2 if max_delays else 0
    
    # ES to CH
    es_ch_overhead = 0
    for n in range(num_ESs):
        ch_idx = cluster_heads[n]
        if ch_idx is not None and ch_idx < len(B_n):
            es_ch_overhead += model_size / (B_n[ch_idx] / 8)
    
    es_ch_overhead *= k3
    
    # CH to Cloud
    ch_cloud_overhead = sum(model_size / (B_n[n] / 8) for n in range(len(B_n)) if B_n[n] > 0)
    
    total_upload_overhead = client_es_overhead + es_ch_overhead + ch_cloud_overhead
    total_download_overhead = total_upload_overhead * download_factor
    return total_upload_overhead + total_download_overhead

def calculate_comm_overhead_hfl_bipartite(C1, C2, cluster_heads, num_users, num_ESs, r, B_n, model_size=1e7, download_factor=0.2, k2=2.0, k3=1.5):
    """
    Calculate communication overhead for HFL with bipartite allocation.
    
    Args:
        C1 (dict): ES to client assignments
        C2 (dict): CH to ES assignments
        cluster_heads (list): Cluster head index for each ES
        num_users (int): Number of clients
        num_ESs (int): Number of edge servers
        r (np.ndarray): Optimized client-ES bandwidth matrix
        B_n (np.ndarray): ES-cloud bandwidth array
        model_size (float): Model size in bits
        download_factor (float): Download delay as fraction of upload
        k2 (float): ES aggregation rounds
        k3 (float): CH aggregation rounds
    
    Returns:
        float: Total communication overhead (seconds)
    """
    # Client to ES
    max_delays = []
    for n in range(num_ESs):
        clients = C1.get(n, [])
        if clients:
            delays = [model_size / (r[m, n] / 8) for m in clients if r[m, n] > 0]
            max_delays.append(max(delays) if delays else 0)
    
    client_es_overhead = max(max_delays) * k2 if max_delays else 0
    
    # ES to CH
    es_ch_overhead = 0
    for n in range(num_ESs):
        ch_idx = cluster_heads[n]
        if ch_idx is not None and ch_idx < len(B_n):
            es_ch_overhead += model_size / (B_n[ch_idx] / 8)
    
    es_ch_overhead *= k3
    
    # CH to Cloud
    ch_cloud_overhead = sum(model_size / (B_n[n] / 8) for n in range(len(B_n)) if B_n[n] > 0)
    
    total_upload_overhead = client_es_overhead + es_ch_overhead + ch_cloud_overhead
    comm_utils = total_upload_overhead * download_factor
    return total_upload_overhead + comm_utils