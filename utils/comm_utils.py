#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import numpy as np
from scipy.optimize import linear_sum_assignment
import random


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


def calculate_transmission_time(model_size, rate_matrix, association_matrix, sender_power=None):
    """
    计算传输时间和通信开销，返回所有关联设备之间传输时间的最大值和通信开销
    
    Args:
        model_size (float): 模型大小（比特）
        rate_matrix (np.ndarray): 速率矩阵，形状为 (发送设备数, 接收设备数)，单位为 bits/s
        association_matrix (np.ndarray): 关联矩阵，形状与速率矩阵相同 (发送设备数, 接收设备数)
                                       矩阵中的1表示存在关联，0表示无关联
        sender_power (float, optional): 所有发送设备的发送功率（数值型），单位为W
                                      如果为None，则不计算通信开销
    
    Returns:
        tuple: (max_transmission_time, communication_overhead)
            - max_transmission_time (float): 最大传输时间（秒）
            - communication_overhead (float): 通信开销（W·s），如果sender_power为None则返回0
    
    Example:
        # 假设有3个发送设备和2个接收设备
        model_size = 1e6  # 1Mb
        rate_matrix = np.array([[1e6, 2e6],    # 发送设备0到接收设备0,1的速率
                               [1.5e6, 1.8e6], # 发送设备1到接收设备0,1的速率  
                               [2.2e6, 1.2e6]]) # 发送设备2到接收设备0,1的速率
        association_matrix = np.array([[1, 0],    # 发送设备0关联到接收设备0
                                      [1, 0],    # 发送设备1关联到接收设备0  
                                      [0, 1]])   # 发送设备2关联到接收设备1
        sender_power = 1.5  # 所有发送设备使用相同功率1.5W
        max_time, comm_overhead = calculate_transmission_time(model_size, rate_matrix, association_matrix, sender_power)
    """
    if association_matrix is None or association_matrix.size == 0:
        print("警告: 关联矩阵为空，返回传输时间为0")
        return 0.0
    
    if rate_matrix is None or rate_matrix.size == 0:
        print("警告: 速率矩阵为空，返回传输时间为0")
        return 0.0
    
    # 检查两个矩阵的形状是否一致
    if rate_matrix.shape != association_matrix.shape:
        print(f"警告: 速率矩阵形状 {rate_matrix.shape} 与关联矩阵形状 {association_matrix.shape} 不一致")
        return 0.0
    
    transmission_times = []
    communication_overhead = 0.0
    
    # 检查发送功率参数（现在是数值型）
    if sender_power is not None:
        try:
            sender_power = float(sender_power)
            if sender_power <= 0:
                print(f"警告: 发送功率 {sender_power} 应为正数，设置为None")
                sender_power = None
        except (TypeError, ValueError):
            print(f"警告: 发送功率参数类型错误，应为数值型，设置为None")
            sender_power = None
    
    # 遍历关联矩阵中的每个位置
    for sender_device in range(association_matrix.shape[0]):
        for target_device in range(association_matrix.shape[1]):
            # 检查是否存在关联（关联矩阵中该位置为1）
            if association_matrix[sender_device, target_device] == 1:
                transmission_rate = rate_matrix[sender_device, target_device]
                
                # 检查传输速率是否有效（大于0）
                if transmission_rate > 0:
                    # 传输时间 = 模型大小（比特） / 传输速率（比特/秒）
                    transmission_time = model_size / transmission_rate
                    transmission_times.append(transmission_time)
                    
                    # 计算通信开销 = 传输时间 * 发送功率（所有设备使用相同功率）
                    if sender_power is not None:
                        overhead = transmission_time * sender_power
                        communication_overhead += overhead
                        
                        # 调试信息（可选，在实际使用时可以注释掉）
                        # print(f"发送设备 {sender_device} -> 目标设备 {target_device}: "
                        #       f"速率 {transmission_rate:.2e} bits/s, "
                        #       f"传输时间 {transmission_time:.4f} 秒, "
                        #       f"功率 {sender_power:.2f} W, 开销 {overhead:.4f} W·s")
                    else:
                        # 调试信息（可选，在实际使用时可以注释掉）
                        # print(f"发送设备 {sender_device} -> 目标设备 {target_device}: "
                        #       f"速率 {transmission_rate:.2e} bits/s, "
                        #       f"传输时间 {transmission_time:.4f} 秒")
                        pass
                else:
                    print(f"警告: 发送设备 {sender_device} 到目标设备 {target_device} 的传输速率为 {transmission_rate}，跳过")
    
    # 返回最大传输时间和通信开销
    if transmission_times:
        max_transmission_time = max(transmission_times)
        # print(f"计算得到的传输时间列表: {[f'{t:.4f}' for t in transmission_times]} 秒")
        print(f"最大传输时间: {max_transmission_time:.4f} 秒")
        if sender_power is not None:
            print(f"总通信开销: {communication_overhead:.4f} W·s")
        return max_transmission_time, communication_overhead
    else:
        print("警告: 没有有效的传输时间计算，返回0")
        return 0.0, 0.0


def get_model_size_in_bits(model):
    """
    计算模型的大小（以比特为单位）
    
    Args:
        model: PyTorch 模型对象
        
    Returns:
        float: 模型大小（比特）
    """
    try:
        import torch
        
        if model is None:
            print("警告: 模型为空，返回默认大小")
            return 1e6  # 默认1Mb
        
        total_params = 0
        for param in model.parameters():
            if param.requires_grad:
                total_params += param.numel()
        
        # 假设每个参数为32位浮点数（4字节 = 32比特）
        model_size_bits = total_params * 32
        
        print(f"模型参数数量: {total_params:,}")
        print(f"模型大小: {model_size_bits / 8 / 1024 / 1024:.2f} MB ({model_size_bits:.0f} bits)")
        
        return float(model_size_bits)
        
    except ImportError:
        print("警告: 无法导入torch，返回默认模型大小")
        return 1e6  # 默认1Mb
    except Exception as e:
        print(f"警告: 计算模型大小时出错: {e}，返回默认大小")
        return 1e6  # 默认1Mb


def association_matrix_to_dict(association_matrix):
    """
    将关联矩阵转换为关联字典
    
    Args:
        association_matrix (np.ndarray): 关联矩阵，形状为 (发送设备数, 目标设备数)
                                       矩阵中的1表示存在关联，0表示无关联
    
    Returns:
        dict: 关联字典，格式为 {目标设备索引: [发送设备索引列表]}
    
    Example:
        # 关联矩阵示例：3个发送设备，2个目标设备
        association_matrix = np.array([[1, 0],    # 发送设备0关联到目标设备0
                                      [1, 0],    # 发送设备1关联到目标设备0  
                                      [0, 1]])   # 发送设备2关联到目标设备1
        # 转换后的字典: {0: [0, 1], 1: [2]}
    """
    if association_matrix is None or association_matrix.size == 0:
        print("警告: 关联矩阵为空，返回空字典")
        return {}
    
    association_dict = {}
    
    # 遍历每个目标设备（列）
    for target_device in range(association_matrix.shape[1]):
        # 找到与该目标设备关联的所有发送设备（行）
        sender_devices = []
        for sender_device in range(association_matrix.shape[0]):
            if association_matrix[sender_device, target_device] == 1:
                sender_devices.append(sender_device)
        
        # 只有当存在关联的发送设备时才添加到字典中
        if sender_devices:
            association_dict[target_device] = sender_devices
    
    print(f"关联矩阵形状: {association_matrix.shape}")
    print(f"转换后的关联字典: {association_dict}")
    
    return association_dict

def select_eh(B_matrix, es_es_rate_matrix, es_cloud_rate_matrix, model_size):
    """
    在每一个簇中, 挑选一个es作为eh, 要求使得max(簇内es传输到eh)+eh-cloud传输时间最小

    Args:
        B_matrix (np.ndarray): 关联矩阵B，形状为(num_es, num_clusters)
        es_es_rate_matrix (np.ndarray): ES-ES传输速率矩阵，形状为(num_es, num_es)，单位为 bits/s
        es_cloud_rate_matrix (np.ndarray): ES-Cloud传输速率矩阵，形状为(num_es,)，单位为 bits/s
        model_size (float): 模型大小（比特）
    
    Returns:
        np.ndarray: ES-EH关联矩阵，形状为(num_es, num_es)，其中每列对应一个被选为EH的ES
    
    Example:
        # 假设有4个ES，2个簇
        B_matrix = np.array([[1, 0],    # ES0 属于簇0
                            [1, 0],    # ES1 属于簇0
                            [0, 1],    # ES2 属于簇1  
                            [0, 1]])   # ES3 属于簇1
        # 函数会在簇0中选择ES0或ES1作为EH，在簇1中选择ES2或ES3作为EH
    """
    if B_matrix is None or B_matrix.size == 0:
        print("警告: B矩阵为空")
        return np.array([])
    
    if es_es_rate_matrix is None or es_es_rate_matrix.size == 0:
        print("警告: ES-ES速率矩阵为空")
        return np.array([])
    
    if es_cloud_rate_matrix is None or es_cloud_rate_matrix.size == 0:
        print("警告: ES-Cloud速率矩阵为空")
        return np.array([])
    
    num_es = B_matrix.shape[0]
    num_clusters = B_matrix.shape[1]
    
    # 检查矩阵维度一致性
    if es_es_rate_matrix.shape != (num_es, num_es):
        print(f"警告: ES-ES速率矩阵形状 {es_es_rate_matrix.shape} 与预期 ({num_es}, {num_es}) 不一致")
        return np.array([])
    
    if len(es_cloud_rate_matrix) != num_es:
        print(f"警告: ES-Cloud速率矩阵长度 {len(es_cloud_rate_matrix)} 与ES数量 {num_es} 不一致")
        return np.array([])
    
    selected_ehs = []  # 存储每个簇选中的EH索引
    
    print(f"开始为 {num_clusters} 个簇选择EH...")
    
    # 遍历每个簇
    for cluster_id in range(num_clusters):
        # 找到属于当前簇的所有ES
        cluster_es_indices = []
        for es_id in range(num_es):
            if B_matrix[es_id, cluster_id] == 1:
                cluster_es_indices.append(es_id)
        if not cluster_es_indices:
            print(f"警告: 簇 {cluster_id} 中没有ES")
            continue
        
        print(f"簇 {cluster_id} 包含ES: {cluster_es_indices}")
        
        best_eh = None
        min_total_time = float('inf')
        
        # 尝试每个ES作为该簇的EH
        for candidate_eh in cluster_es_indices:
            # 计算簇内其他ES到候选EH的传输时间
            intra_cluster_times = []
            
            for es_id in cluster_es_indices:
                if es_id != candidate_eh:  # 排除EH自己
                    # ES到EH的传输速率
                    es_to_eh_rate = es_es_rate_matrix[es_id, candidate_eh]
                    
                    if es_to_eh_rate > 0:
                        # 传输时间 = 模型大小 / 传输速率
                        transmission_time = model_size / es_to_eh_rate
                        intra_cluster_times.append(transmission_time)
                    else:
                        # 如果速率为0，设置为无穷大（表示无法传输）
                        intra_cluster_times.append(float('inf'))
            
            # 计算簇内最大传输时间
            max_intra_time = max(intra_cluster_times) if intra_cluster_times else 0.0
            
            # 计算EH到云端的传输时间
            eh_to_cloud_rate = es_cloud_rate_matrix[candidate_eh]
            # 确保 eh_to_cloud_rate 是标量
            if isinstance(eh_to_cloud_rate, np.ndarray):
                eh_to_cloud_rate = float(eh_to_cloud_rate.item())
            else:
                eh_to_cloud_rate = float(eh_to_cloud_rate)
                
            if eh_to_cloud_rate > 0:
                eh_to_cloud_time = model_size / eh_to_cloud_rate
            else:
                eh_to_cloud_time = float('inf')
            
            # 总传输时间 = max(簇内ES到EH时间) + EH到云端时间
            total_time = max_intra_time + eh_to_cloud_time
            
            print(f"  候选EH {candidate_eh}: 簇内最大时间={max_intra_time:.4f}s, "
                  f"到云端时间={eh_to_cloud_time:.4f}s, 总时间={total_time:.4f}s")
            
            # 选择总时间最小的ES作为EH
            if total_time < min_total_time:
                min_total_time = total_time
                best_eh = candidate_eh
        
        if best_eh is not None:
            selected_ehs.append(best_eh)
            print(f"簇 {cluster_id} 选择ES {best_eh} 作为EH，总传输时间: {min_total_time:.4f}s")
        else:
            print(f"警告: 簇 {cluster_id} 未能选择出合适的EH")
    
    # 构建ES-EH关联矩阵
    # 矩阵形状为(num_es, num_es)，每一列对应一个被选为EH的ES
    num_selected_ehs = len(selected_ehs)
    if num_selected_ehs == 0:
        print("警告: 没有选择出任何EH")
        return np.zeros((num_es, num_es), dtype=int)
    
    # 创建ES-EH关联矩阵，形状为(num_es, num_es)
    es_eh_matrix = np.zeros((num_es, num_es), dtype=int)
    
    # 根据原始B矩阵和选中的EH来填充关联矩阵
    for cluster_id in range(num_clusters):
        if cluster_id < len(selected_ehs):
            selected_eh = selected_ehs[cluster_id]
            
            # 将属于该簇的所有ES关联到选中的EH
            # 在矩阵中，第selected_eh列表示该EH，所有属于该簇的ES在该列置1
            for es_id in range(num_es):
                if B_matrix[es_id, cluster_id] == 1:
                    es_eh_matrix[es_id, selected_eh] = 1

    
    # 构建EH-Cloud关联矩阵
    # 矩阵形状为(num_es, 1)，表示每个ES与云服务器的连接关系
    # 只有被选为EH的ES才与云服务器直接连接
    eh_cloud_matrix = np.zeros((num_es, 1), dtype=int)
    
    # 将选中的EH标记为与云服务器连接
    for selected_eh in selected_ehs:
        eh_cloud_matrix[selected_eh, 0] = 1
    
    print(f"最终选择的EH列表: {selected_ehs}")
    print(f"ES-EH关联矩阵形状: {es_eh_matrix.shape}")
    print(f"ES-EH关联矩阵:\n{es_eh_matrix}")
    print(f"EH-Cloud关联矩阵形状: {eh_cloud_matrix.shape}")
    print(f"EH-Cloud关联矩阵:\n{eh_cloud_matrix.flatten()}")

    return es_eh_matrix, eh_cloud_matrix


def select_eh_random(B_matrix):
    """
    在每一个簇中随机挑选一个ES作为EH
    
    Args:
        B_matrix (np.ndarray): 关联矩阵B，形状为(num_es, num_clusters)
    
    Returns:
        tuple: (es_eh_matrix, eh_cloud_matrix)
            - es_eh_matrix (np.ndarray): ES-EH关联矩阵，形状为(num_es, num_es)
            - eh_cloud_matrix (np.ndarray): EH-Cloud关联矩阵，形状为(num_es, 1)
    
    Example:
        # 假设有4个ES，2个簇
        B_matrix = np.array([[1, 0],    # ES0 属于簇0
                            [1, 0],    # ES1 属于簇0
                            [0, 1],    # ES2 属于簇1  
                            [0, 1]])   # ES3 属于簇1
        # 函数会在簇0中随机选择ES0或ES1作为EH，在簇1中随机选择ES2或ES3作为EH
    """
    if B_matrix is None or B_matrix.size == 0:
        print("警告: B矩阵为空")
        return np.array([]), np.array([])
    
    num_es = B_matrix.shape[0]
    num_clusters = B_matrix.shape[1]
    
    selected_ehs = []  # 存储每个簇选中的EH索引
    
    print(f"开始为 {num_clusters} 个簇随机选择EH...")
    
    # 遍历每个簇
    for cluster_id in range(num_clusters):
        # 找到属于当前簇的所有ES
        cluster_es_indices = []
        for es_id in range(num_es):
            if B_matrix[es_id, cluster_id] == 1:
                cluster_es_indices.append(es_id)
        
        if not cluster_es_indices:
            print(f"警告: 簇 {cluster_id} 中没有ES")
            continue
        
        # 随机选择一个ES作为EH
        selected_eh = random.choice(cluster_es_indices)
        selected_ehs.append(selected_eh)
        
        print(f"簇 {cluster_id} 包含ES: {cluster_es_indices}")
        print(f"簇 {cluster_id} 随机选择ES {selected_eh} 作为EH")
    
    # 构建ES-EH关联矩阵
    num_selected_ehs = len(selected_ehs)
    if num_selected_ehs == 0:
        print("警告: 没有选择出任何EH")
        return np.zeros((num_es, num_es), dtype=int), np.zeros((num_es, 1), dtype=int)
    
    # 创建ES-EH关联矩阵，形状为(num_es, num_es)
    es_eh_matrix = np.zeros((num_es, num_es), dtype=int)
    
    # 根据原始B矩阵和选中的EH来填充关联矩阵
    for cluster_id in range(num_clusters):
        if cluster_id < len(selected_ehs):
            selected_eh = selected_ehs[cluster_id]
            
            # 将属于该簇的所有ES关联到选中的EH
            # 在矩阵中，第selected_eh列表示该EH，所有属于该簇的ES在该列置1
            for es_id in range(num_es):
                if B_matrix[es_id, cluster_id] == 1:
                    es_eh_matrix[es_id, selected_eh] = 1
    
    # 构建EH-Cloud关联矩阵
    # 矩阵形状为(num_es, 1)，表示每个ES与云服务器的连接关系
    # 只有被选为EH的ES才与云服务器直接连接
    eh_cloud_matrix = np.zeros((num_es, 1), dtype=int)
    
    # 将选中的EH标记为与云服务器连接
    for selected_eh in selected_ehs:
        eh_cloud_matrix[selected_eh, 0] = 1
    
    print(f"最终随机选择的EH列表: {selected_ehs}")
    print(f"ES-EH关联矩阵形状: {es_eh_matrix.shape}")
    print(f"ES-EH关联矩阵:\n{es_eh_matrix}")
    print(f"EH-Cloud关联矩阵形状: {eh_cloud_matrix.shape}")
    print(f"EH-Cloud关联矩阵:\n{eh_cloud_matrix.flatten()}")
    
    return es_eh_matrix, eh_cloud_matrix

