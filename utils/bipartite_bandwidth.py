import random
import math
import networkx as nx
import numpy as np
# import cupy as cp
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from options import args_parser

def calculate_transmission_rates(distance_matrix, transmit_power, center_bandwidth, bandwidth_sigma, 
                               noise_density=10**(-20.4), path_loss_exponent=3.5, g0_at_1m=1e-4, 
                               bandwidth_range=(1e6, 5e7), is_diagonal_zero=False, node_indices=None):
    """
    计算基于距离矩阵的传输速率
    
    Args:
        distance_matrix: 距离矩阵 (M, N)
        transmit_power: 发射功率向量或标量
        center_bandwidth: 中心带宽值（对数正态分布的中心）
        bandwidth_sigma: 带宽标准差（对数正态分布）
        noise_density: 噪声功率谱密度，默认10**(-20.4)
        path_loss_exponent: 路径损耗指数，默认3.5
        g0_at_1m: 1米处的参考信道增益，默认1e-4
        bandwidth_range: 带宽范围元组 (min, max)，默认(1e6, 5e7)
        is_diagonal_zero: 是否将对角线元素置零，默认False
        node_indices: 如果提供，将使用这些索引从transmit_power中选择值

    Returns:
        tuple: (传输速率矩阵, 信道增益矩阵, 带宽矩阵)
    """
    # 复制距离矩阵以避免修改原始矩阵
    dist_m = np.copy(distance_matrix)
    
    # 避免距离为0导致无穷大信道增益
    dist_m[dist_m == 0] = 1.0
    
    # 计算信道增益 (基于路径损耗模型)
    channel_gain = g0_at_1m * (dist_m ** -path_loss_exponent)
    
    # 如果需要，将对角线元素置零
    if is_diagonal_zero:
        np.fill_diagonal(channel_gain, 0)
    
    # 生成带宽矩阵 (对数正态分布)
    mu = np.log(center_bandwidth)
    bandwidth = np.random.lognormal(mean=mu, sigma=bandwidth_sigma, size=dist_m.shape)
    bandwidth = np.clip(bandwidth, bandwidth_range[0], bandwidth_range[1])
    
    # 如果需要，将对角线带宽置零
    if is_diagonal_zero:
        np.fill_diagonal(bandwidth, 0)
    
    # 计算接收信号功率
    # 如果提供了node_indices，则使用对应的发射功率
    if node_indices is not None:
        power = transmit_power[node_indices]
        received_power = power[:, None] * channel_gain
    else:
        # 否则，直接使用发射功率（可以是标量或向量）
        if isinstance(transmit_power, (int, float)):
            received_power = transmit_power * channel_gain
        else:
            received_power = transmit_power[:, None] * channel_gain
    
    # 计算噪声功率
    noise_power = noise_density * bandwidth
    
    # 计算信噪比 (SNR)
    snr = received_power / (noise_power + 1e-30)  # 添加小值避免除零
    
    # 计算传输速率 (香农公式)
    rate = bandwidth * np.log2(1 + snr)
    
    # 如果需要，确保对角线速率为0
    if is_diagonal_zero:
        np.fill_diagonal(rate, 0)
    
    return rate, channel_gain, bandwidth

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    # cp.random.seed(seed)

def select_edge_servers_uniformly(node_ids, pos, es_ratio):
    """
    使用K-Means聚类算法选择地理分布均匀的边缘服务器
    
    Args:
        node_ids: 所有有效节点ID列表
        pos: 节点位置字典 {node_id: (lon, lat)}
        es_ratio: 边缘服务器占比
    
    Returns:
        tuple: (边缘服务器节点列表, 客户端节点列表)
    """
    num_es = max(1, int(len(node_ids) * es_ratio))
    print(f"Selecting {num_es} edge servers from {len(node_ids)} nodes (ratio: {es_ratio:.2f})")
    
    # 提取节点位置用于聚类
    positions = []
    node_positions_dict = {}
    for node_id in node_ids:
        if node_id in pos:
            lon, lat = pos[node_id]
            positions.append([lon, lat])
            node_positions_dict[node_id] = [lon, lat]
    
    positions = np.array(positions)
    
    # 使用K-Means聚类确保边缘服务器地理分布均匀
    kmeans = KMeans(n_clusters=num_es, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(positions)
    cluster_centers = kmeans.cluster_centers_
    
    # 为每个聚类选择最接近聚类中心的节点作为边缘服务器
    es_nodes = []
    for i in range(num_es):
        cluster_nodes = [node_ids[j] for j in range(len(node_ids)) if cluster_labels[j] == i]
        cluster_center = cluster_centers[i]
        
        # 找到距离聚类中心最近的节点
        min_distance = float('inf')
        best_node = None
        for node_id in cluster_nodes:
            if node_id in node_positions_dict:
                node_pos = node_positions_dict[node_id]
                distance = np.sqrt((node_pos[0] - cluster_center[0])**2 + 
                                 (node_pos[1] - cluster_center[1])**2)
                if distance < min_distance:
                    min_distance = distance
                    best_node = node_id
        
        if best_node:
            es_nodes.append(best_node)
    
    # 确保边缘服务器数量正确
    if len(es_nodes) < num_es:
        remaining_nodes = [n for n in node_ids if n not in es_nodes]
        while len(es_nodes) < num_es and remaining_nodes:
            es_nodes.append(remaining_nodes.pop(0))
    
    client_nodes = [node for node in node_ids if node not in es_nodes]
    
    print(f"Selected edge servers: {es_nodes}")
    print(f"Client nodes count: {len(client_nodes)}, Edge servers count: {len(es_nodes)}")
    
    return es_nodes, client_nodes


def visualize_node_distribution(client_nodes, es_nodes, pos, es_ratio, save_path=None, cloud_pos=None, association_matrix=None):
    """
    生成节点地理分布可视化图
    
    Args:
        client_nodes: 客户端节点列表
        es_nodes: 边缘服务器节点列表
        pos: 节点位置字典 {node_id: (lon, lat)}
        es_ratio: 边缘服务器占比
        save_path: 保存路径（可选）
        cloud_pos: 云服务器位置 (lon, lat)，如果提供则显示
        association_matrix: 客户端到边缘服务器的关联矩阵，如果提供则绘制关联边
    
    Returns:
        str: 保存的文件名
    """
    plt.figure(figsize=(12, 8))
    
    # 如果提供了关联矩阵，先绘制连接边
    if association_matrix is not None and len(client_nodes) > 0 and len(es_nodes) > 0:
        print("绘制客户端与边缘服务器之间的关联边...")
        # 绘制客户端和边缘服务器之间的关联边
        for i, client in enumerate(client_nodes):
            if client in pos:
                client_pos = pos[client]
                # 找出此客户端关联的边缘服务器
                for j, es in enumerate(es_nodes):
                    if es in pos and association_matrix[i, j] == 1:
                        es_pos = pos[es]
                        # 绘制边，使用灰色，z-order设为较低值，确保点在边上面
                        plt.plot([client_pos[0], es_pos[0]], [client_pos[1], es_pos[1]], 
                                color='#909291', linewidth=0.8, alpha=0.6, zorder=1)
    
    # 绘制客户端节点
    client_positions = np.array([pos[node] for node in client_nodes if node in pos])
    if len(client_positions) > 0:
        plt.scatter(client_positions[:, 0], client_positions[:, 1], 
                   c='#719aac', marker='o', s=100, alpha=0.7, label=f'Clients ({len(client_nodes)})', zorder=2)
    
    # 绘制边缘服务器节点
    es_positions = np.array([pos[node] for node in es_nodes if node in pos])
    if len(es_positions) > 0:
        plt.scatter(es_positions[:, 0], es_positions[:, 1], 
                   c='#e29135', marker='^', s=100, 
                   label=f'Edge Servers ({len(es_nodes)})', zorder=3)
    
    # 添加边缘服务器编号标注
    for i, node in enumerate(es_nodes):
        if node in pos:
            lon, lat = pos[node]
            plt.annotate(f'ES{i}', (lon, lat), xytext=(5, 5), 
                       textcoords='offset points', fontsize=8, fontweight='bold', zorder=4)
                       
    # 如果提供了云服务器位置，则绘制云服务器
    if cloud_pos is not None:
        plt.scatter([cloud_pos[0]], [cloud_pos[1]], 
                   c='#d83a3a', marker='*', s=300,
                   label='Cloud Server')
        plt.annotate('Cloud', (cloud_pos[0], cloud_pos[1]), xytext=(5, 5),
                   textcoords='offset points', fontsize=10, fontweight='bold', color='#d83a3a')
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    total_nodes = len(client_nodes) + len(es_nodes)
    
    # 根据是否包含关联边和云服务器修改标题
    title = f'Geographic Distribution of Nodes\n(ES Ratio: {es_ratio:.2f}, Total: {total_nodes} nodes)'
    if association_matrix is not None:
        title += ' with Associations'
    if cloud_pos is not None:
        title += ' and Cloud Server'
    plt.title(title)
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存可视化图，文件名根据内容区分
    if save_path:
        viz_filename = save_path
    else:
        viz_filename = f"./save/node_distribution_es{len(es_nodes)}_ratio{es_ratio:.2f}"
        if association_matrix is not None:
            viz_filename += "_with_associations"
        if cloud_pos is not None:
            viz_filename += "_with_cloud"
        viz_filename += ".png"
    
    # 确保保存目录存在
    import os
    os.makedirs(os.path.dirname(viz_filename), exist_ok=True)
    
    plt.savefig(viz_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Geographic distribution saved as: {viz_filename}")
    
    return viz_filename


def calculate_distance(lat1, lon1, lat2, lon2):
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    R = 6371.0
    return R * c

def build_bipartite_graph(graphml_file="./graph-example/Ulaknet.graphml", es_ratio=None, visualize=True):
    """
    从GraphML文件构建客户端-边缘服务器二部图网络拓扑
    
    Args:
        graphml_file: GraphML网络拓扑文件路径
        es_ratio: 边缘服务器在总节点中的占比
        visualize: 是否生成可视化图
    
    Returns:
        tuple: (二部图, 客户端节点列表, 边缘服务器节点列表, 客户端-ES距离矩阵, ES-ES距离矩阵, 节点位置字典)
    """

    # 读取GraphML文件
    try:
        G = nx.read_graphml(graphml_file, node_type=str)
    except Exception as e:
        print(f"Error reading GraphML file: {e}")
        return None, [], [], None, None, None

    # 过滤有效节点（必须有经纬度信息）
    node_ids = []
    removed_nodes = []
    for node in G.nodes(data=True):
        node_id, node_data = node
        # 无经纬度的节点移除
        if 'Latitude' not in node_data or 'Longitude' not in node_data:
            removed_nodes.append(node_id)
            print(f"Node {node_id} removed: Missing Latitude or Longitude")
            continue
        try:
            lat = float(node_data['Latitude'])
            lon = float(node_data['Longitude'])
            # 经纬度为0的节点移除
            if lat == 0 and lon == 0:
                removed_nodes.append(node_id)
                print(f"Node {node_id} removed: Invalid coordinates (0, 0)")
                continue
            # 经纬度正常节点添加位置属性
            G.nodes[node_id]['pos'] = (lon, lat)
            node_ids.append(node_id)
        except (ValueError, TypeError) as e:
            # 经纬度解析错误的节点移除
            removed_nodes.append(node_id)
            print(f"Node {node_id} removed: Invalid Latitude/Longitude ({e})")
            continue

    # 清理无效节点
    G.remove_nodes_from(removed_nodes)
    if removed_nodes:
        print(f"Removed {len(removed_nodes)} nodes missing lat/lon data: {removed_nodes}")

    pos = nx.get_node_attributes(G, 'pos')

    # 进一步清理仍缺少pos属性的节点
    invalid_nodes = [n for n in G.nodes() if n not in pos]
    if invalid_nodes:
        print(f"Error: Nodes {invalid_nodes} still lack pos attribute, removing")
        G.remove_nodes_from(invalid_nodes)
        node_ids = [n for n in node_ids if n in pos]

    if not node_ids:
        print("Error: No valid nodes after filtering")
        return None, [], [], None, None, None
        
    # 如果es_ratio为None，设置默认值
    if es_ratio is None:
        args = args_parser()
        es_ratio = args.es_ratio
        print(f"Using default es_ratio from args: {es_ratio}")

    # 使用封装的函数选择边缘服务器
    es_nodes, client_nodes = select_edge_servers_uniformly(node_ids, pos, es_ratio)

    # 构建完全二部图（每个客户端连接所有边缘服务器）
    bipartite_graph = nx.Graph()
    bipartite_graph.add_nodes_from(client_nodes, bipartite=0)  # 客户端标记为0
    bipartite_graph.add_nodes_from(es_nodes, bipartite=1)      # 边缘服务器标记为1
    for c in client_nodes:
        for e in es_nodes:
            bipartite_graph.add_edge(c, e)

    # 为二部图节点添加位置信息
    for node in bipartite_graph.nodes():
        if node in pos:
            bipartite_graph.nodes[node]['pos'] = pos[node]
        else:
            print(f"Warning: Node {node} in bipartite graph lacks pos, removing")
            bipartite_graph.remove_node(node)
            if node in client_nodes:
                client_nodes.remove(node)
            if node in es_nodes:
                es_nodes.remove(node)

    # 最终位置检查
    pos = nx.get_node_attributes(bipartite_graph, 'pos')
    if not all(node in pos for node in client_nodes + es_nodes):
        print("Error: Some nodes still lack pos attributes after final check")
        return None, [], [], None, None, None

    # 计算客户端-边缘服务器距离矩阵
    distance_matrix = np.zeros((len(client_nodes), len(es_nodes)))
    for i, c in enumerate(client_nodes):
        for j, e in enumerate(es_nodes):
            if c in pos and e in pos:
                c_pos = pos[c]
                e_pos = pos[e]
                distance_matrix[i, j] = calculate_distance(c_pos[1], c_pos[0], e_pos[1], e_pos[0]) * 1000
            else:
                print(f"Warning: No position for client {c} or edge {e}, setting distance to infinity")
                distance_matrix[i, j] = float('inf')

    # 打印节点样例信息
    if client_nodes:
        sample_node = G.nodes[client_nodes[0]]
        print(f"Node data sample: {sample_node}")

    # 生成地理分布可视化（如果需要）
    if visualize:
        # 在初始阶段，还没有关联矩阵，所以这里不传递association_matrix
        visualize_node_distribution(client_nodes, es_nodes, pos, es_ratio)

    return bipartite_graph, client_nodes, es_nodes, distance_matrix, pos

def plot_graph(bipartite_graph, client_nodes, es_nodes):
    pos = nx.get_node_attributes(bipartite_graph, 'pos')
    missing_pos = [n for n in client_nodes + es_nodes if n not in pos]
    if missing_pos:
        print(f"Error: Nodes {missing_pos} missing pos attributes in plot_graph")
        return
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(bipartite_graph, pos, nodelist=client_nodes, node_color='lightblue', node_shape='o',
                           node_size=300, label='Clients')
    nx.draw_networkx_nodes(bipartite_graph, pos, nodelist=es_nodes, node_color='lightcoral', node_shape='s',
                           node_size=300, label='Edge Servers')
    plt.legend()
    plt.title("Bipartite Graph (Clients and Edge Servers)")
    plt.savefig("bipartite_graph.png")
    plt.close()

def create_association_based_on_rate(rate_matrix):
    """
    基于传输速率矩阵为每个客户端选择最佳的边缘服务器
    
    Args:
        rate_matrix: 传输速率矩阵 (M, N)，M为客户端数量，N为边缘服务器数量
    
    Returns:
        tuple: (分配列表[(client_idx, es_idx), ...], 关联矩阵(M, N))
    """
    M, N = rate_matrix.shape  # M: 客户端数量, N: 边缘服务器数量
    
    # 为每个客户端选择传输速率最大的边缘服务器
    best_es_for_client = np.argmax(rate_matrix, axis=1)  # 每个客户端最佳的边缘服务器索引
    
    # 创建分配列表 [(client_idx, es_idx), ...]
    assignments = [(m, best_es_for_client[m]) for m in range(M)]
    
    # 创建关联矩阵 (二进制矩阵，表示客户端和边缘服务器之间的关联)
    association_matrix = np.zeros((M, N), dtype=int)
    for m, n in assignments:
        association_matrix[m, n] = 1
    
    return assignments, association_matrix

def plot_assigned_graph(bipartite_graph, client_nodes, es_nodes, assignments, cluster_heads, C2, es_nodes_indices):
    pos = nx.get_node_attributes(bipartite_graph, 'pos')
    missing_pos = [n for n in client_nodes + es_nodes if n not in pos]
    if missing_pos:
        print(f"Error: Nodes {missing_pos} missing pos attributes in plot_assigned_graph")
        return
    plt.figure(figsize=(12, 10))
    nx.draw_networkx_nodes(bipartite_graph, pos, nodelist=client_nodes, node_color='lightblue', node_shape='o',
                           node_size=300, label='Clients')
    nx.draw_networkx_nodes(bipartite_graph, pos, nodelist=[es_nodes[i] for i in es_nodes_indices if i not in cluster_heads.values()],
                           node_color='lightcoral', node_shape='s', node_size=300, label='Edge Servers')
    nx.draw_networkx_nodes(bipartite_graph, pos, nodelist=[es_nodes[v] for v in cluster_heads.values() if v != -1],
                           node_color='gold', node_shape='*', node_size=500, label='Cluster Heads')
    assigned_edges = [(client_nodes[m], es_nodes[n]) for m, n in assignments]
    nx.draw_networkx_edges(bipartite_graph, pos, edgelist=assigned_edges, edge_color='navy', width=2)
    for ch_idx, es_indices in C2.items():
        if cluster_heads[ch_idx] == -1:
            continue
        ch_node = es_nodes[cluster_heads[ch_idx]]
        for es_idx in es_indices:
            if es_idx != cluster_heads[ch_idx]:
                nx.draw_networkx_edges(bipartite_graph, pos, edgelist=[(es_nodes[es_idx], ch_node)],
                                       edge_color='purple', width=1.5, style='dashed')
    plt.legend()
    plt.title("Assigned Bipartite Graph with Cluster Heads (Navy: Client-ES, Purple: ES-CH)")
    plt.savefig("assigned_bipartite_graph_with_ch.png")
    plt.close()

def establish_communication_channels(client_nodes, es_nodes, distance_matrix, pos, max_capacity=None):
    """
    为层次联邦学习系统计算通信速率并得到client-es关联策略。
    功能概述:
    1. 计算客户端到边缘服务器(client-es)的传输速率
    2. 根据最大传输速率确定客户端到边缘服务器的关联
    3. 筛选活跃的边缘服务器集合
    4. 计算活跃边缘服务器之间(es-es)的距离矩阵和传输速率
    5. 确定云服务器位置(位于活跃边缘服务器的质心)
    6. 计算边缘服务器到云服务器(es-cloud)的传输速率
    7. 计算客户端到云服务器(client-cloud)的直接传输速率
    
    Args:
        client_nodes: 客户端节点列表
        es_nodes: 边缘服务器节点列表
        distance_matrix: 客户端到边缘服务器的距离矩阵
        pos: 节点位置字典 {node_id: (lon, lat)}
        max_capacity: 每个边缘服务器最大容量(客户端数)
    
    Returns:
        client_nodes: 客户端节点列表
        active_es_nodes: 活跃边缘服务器节点列表
        association_matrix: 客户端到边缘服务器的关联矩阵
        r_client_to_es: 客户端到边缘服务器的传输速率矩阵
        r_es: 边缘服务器间的传输速率矩阵
        r_es_to_cloud: 边缘服务器到云服务器的传输速率矩阵
        r_client_to_cloud: 客户端到云服务器的直接传输速率矩阵
    """

    # 如果max_capacity为None或0，先尝试从args中读取，如果args.max_capacity也为0，再自动计算
    if max_capacity is None or max_capacity == 0:
        args = args_parser()
        if args.max_capacity > 0:
            max_capacity = args.max_capacity
            print(f"Using max_capacity: {max_capacity} from command line arguments")
        else:
            max_capacity = max(1, int(len(client_nodes) / len(es_nodes)) + 1)
            print(f"Automatically calculated max_capacity: {max_capacity} based on {len(client_nodes)} clients and {len(es_nodes)} edge servers")
            
    # 1. 初始化系统参数
    M = len(client_nodes)      # 客户端数量
    N = len(es_nodes)          # 边缘服务器数量
    B_cloud = 5e7              # 总云端带宽 (50 MHz)
    p_m = np.ones(M) * 1.0     # 客户端发射功率 (W)
    N0 = 10**(-20.4)           # 噪声功率谱密度 (W/Hz)
    path_loss_exponent = 3.5   # 路径损耗指数 (alpha)
    g0_at_1m = 1e-4            # 在参考距离d0=1米处的信道增益

    #=======================STEP 1: 计算client-es传输速率========================#
    print("\n========== 计算客户端到边缘服务器(client-es)传输速率 ==========")
    
    # 设置客户端到边缘服务器的通信参数
    client_center_bandwidth = 2e7        # 中心带宽 20 MHz
    client_bandwidth_sigma = 0.5         # 带宽对数正态分布的标准差
    client_bandwidth_range = (1e6, 5e7)  # 带宽范围 1-50 MHz
    
    # 使用通用函数计算传输速率
    r_client_to_es, g, B_mn = calculate_transmission_rates(
        distance_matrix=distance_matrix,
        transmit_power=p_m,              # 客户端发射功率
        center_bandwidth=client_center_bandwidth,
        bandwidth_sigma=client_bandwidth_sigma,
        noise_density=N0,
        path_loss_exponent=path_loss_exponent,
        g0_at_1m=g0_at_1m,
        bandwidth_range=client_bandwidth_range
    )

    
    # 打印计算结果示例
    print(f"信道增益矩阵 (g, 前5x5):\n{g[:min(5, M), :min(5, N)]}\n")
    print(f"带宽分配矩阵 (B_mn, 前5x5):\n{B_mn[:min(5, M), :min(5, N)]}\n")
    print(f"初始传输速率矩阵 (r_client_to_es, 前5x5) [bit/s]:\n{r_client_to_es[:min(5, M), :min(5, N)]}")
    
    #=======================STEP 2: 根据传输速率创建关联矩阵========================#
    print("\n========== 基于最大传输速率为客户端分配边缘服务器 ==========")

    # 基于最大传输速率为每个客户端选择边缘服务器
    assignments, association_matrix = create_association_based_on_rate(r_client_to_es)
    print(f"创建了基于最大传输速率的关联矩阵")
    print(f"总关联数: {len(assignments)}/{M} ({len(assignments)/M:.2%})")
    print(f"关联矩阵形状: {association_matrix.shape}")
    print(f"关联矩阵示例 (前5x5):\n{association_matrix[:min(5, M), :min(5, N)]}")

    #=======================STEP 3: 确定活跃边缘服务器集合========================#
    print("\n========== 根据关联矩阵确定活跃边缘服务器集合 ==========")

    # 找出与至少一个客户端关联的边缘服务器(活跃边缘服务器)
    active_es_indices = np.where(np.sum(association_matrix, axis=0) > 0)[0]
    active_es_nodes = [es_nodes[i] for i in active_es_indices]
    N_active = len(active_es_nodes)

    print(f"活跃边缘服务器数量: {N_active}/{N} ({N_active/N:.2%})")
    print(f"活跃边缘服务器索引: {active_es_indices}")

    if N_active == 0:
        # 报错并终止程序
        raise RuntimeError("错误: 没有活跃的边缘服务器，无法继续执行程序。请检查网络拓扑和分配算法。")

    #=======================STEP 4: 计算活跃es-es距离矩阵========================#
    print("\n========== 计算活跃边缘服务器之间的距离矩阵 ==========")
    
    # 生成活跃边缘服务器之间的距离矩阵
    active_es_distance_matrix = np.zeros((N_active, N_active))
    for i, idx1 in enumerate(active_es_indices):
        e1 = es_nodes[idx1]
        for j, idx2 in enumerate(active_es_indices):
            e2 = es_nodes[idx2]
            if e1 in pos and e2 in pos:
                e1_pos = pos[e1]
                e2_pos = pos[e2]
                # 计算地理距离(千米)并转换为米
                active_es_distance_matrix[i, j] = calculate_distance(e1_pos[1], e1_pos[0], e2_pos[1], e2_pos[0]) * 1000
            else:
                print(f"警告: 活跃边缘服务器 {e1} 或 {e2} 缺少位置信息，设置距离为无穷大")
                active_es_distance_matrix[i, j] = float('inf')
    
    print(f"生成活跃ES距离矩阵，形状: {active_es_distance_matrix.shape}")
    if N_active > 0:
        print(f"活跃ES距离矩阵示例 (前5x5) [米]:\n{active_es_distance_matrix[:min(5, N_active), :min(5, N_active)]}")
    
    #=======================STEP 5: 计算活跃es-es传输速率========================#
    print("\n========== 计算活跃边缘服务器之间的传输速率 ==========")
    
    # 定义ES-to-ES通信的专属参数
    p_es = np.ones(N) * 5.0            # ES通常有更高的发射功率 (5W)
    es_center_bandwidth = 1e8          # ES间通常有更高带宽 (100 MHz)
    es_bandwidth_sigma = 0.3           # 带宽对数正态分布的标准差
    es_bandwidth_range = (2e7, 2e8)    # 带宽范围 20-200 MHz
    
    # 使用通用函数计算活跃ES之间的传输速率
    r_es_active, g_es, B_es = calculate_transmission_rates(
        distance_matrix=active_es_distance_matrix,
        transmit_power=p_es,
        center_bandwidth=es_center_bandwidth,
        bandwidth_sigma=es_bandwidth_sigma,
        noise_density=N0,
        path_loss_exponent=path_loss_exponent,
        g0_at_1m=g0_at_1m,
        bandwidth_range=es_bandwidth_range,
        is_diagonal_zero=True,  # 对角线元素置零（自己到自己）
        node_indices=active_es_indices  # 使用活跃ES的索引
    )
    
    # 创建完整的r_es矩阵，默认为0
    r_es = np.zeros((N, N))
    
    # 将活跃ES之间的传输速率填入完整矩阵
    for i, idx_i in enumerate(active_es_indices):
        for j, idx_j in enumerate(active_es_indices):
            if idx_i != idx_j:
                r_es[idx_i, idx_j] = r_es_active[i, j]

    # 打印结果示例
    print(f"活跃ES间信道增益示例 (前5x5):\n{g_es[:min(5, N_active), :min(5, N_active)]}\n")
    print(f"活跃ES间带宽分配示例 (前5x5) [Hz]:\n{B_es[:min(5, N_active), :min(5, N_active)]}\n")
    print(f"活跃ES间传输速率示例 (前5x5) [bit/s]:\n{r_es_active[:min(5, N_active), :min(5, N_active)]}")
    
    # 打印平均传输速率
    if N_active > 1:
        avg_rate = np.sum(r_es_active) / (N_active * (N_active - 1))  # 排除对角线元素
        print(f"活跃ES之间的平均传输速率: {avg_rate/1e6:.2f} Mbps")
    
    # 对于演示，显示完整ES矩阵的示例
    print(f"完整ES传输速率矩阵示例 (前5x5) [bit/s]:\n{r_es[:min(5, N), :min(5, N)]}")
    
    #=======================STEP 6: 确定云服务器位置========================#
    print("\n========== 确定云服务器位置(活跃ES质心) ==========")

    # 计算活跃边缘服务器的地理质心作为云服务器的位置
    cloud_pos = None
    es_to_cloud_distance = None
    
    if N_active > 0:
        # 收集活跃边缘服务器的经纬度坐标
        active_es_coords = []
        for idx in active_es_indices:
            es_node = es_nodes[idx]
            if es_node in pos:
                # pos中坐标是(lon, lat)格式
                active_es_coords.append(pos[es_node])

        # 计算质心
        if active_es_coords:
            cloud_lon = sum(coord[0] for coord in active_es_coords) / len(active_es_coords)
            cloud_lat = sum(coord[1] for coord in active_es_coords) / len(active_es_coords)
            cloud_pos = (cloud_lon, cloud_lat)
            
            print(f"云服务器位于活跃ES质心: 经度={cloud_lon:.6f}, 纬度={cloud_lat:.6f}")
            
            #=======================STEP 7: 计算ES到云的距离========================#
            print("\n========== 计算边缘服务器到云服务器的距离 ==========")
            
            # 计算每个边缘服务器到云的距离
            es_to_cloud_distance = np.zeros(N)
            print("\n边缘服务器到云服务器的距离:")
            
            for idx in range(N):
                es_node = es_nodes[idx]
                if es_node in pos:
                    es_pos = pos[es_node]
                    # 计算地理距离(千米)并转换为米
                    distance_km = calculate_distance(es_pos[1], es_pos[0], cloud_lat, cloud_lon)
                    es_to_cloud_distance[idx] = distance_km * 1000
                    
                    # 只打印活跃的边缘服务器
                    if idx in active_es_indices:
                        print(f"  ES{idx} → Cloud: {distance_km:.3f} km ({distance_km*1000:.1f} m)")
                else:
                    print(f"警告: 边缘服务器 {es_node} 缺少位置信息，设置距离为无穷大")
                    es_to_cloud_distance[idx] = float('inf')
        else:
            print("警告: 活跃边缘服务器缺少位置信息，无法计算云服务器位置")
    else:
        print("警告: 没有活跃边缘服务器，无法计算云服务器位置")

    #=======================STEP 8: 计算ES到云的传输速率========================#
    print("\n========== 计算边缘服务器到云服务器(ES-Cloud)的传输速率 ==========")
    
    # 计算ES到云的传输速率
    r_es_to_cloud = None
    if es_to_cloud_distance is not None:
        # 定义ES-to-Cloud通信的专属参数
        cloud_center_bandwidth = 1.2e8        # 云连接通常带宽更高 (120 MHz)
        cloud_bandwidth_sigma = 0.25          # 带宽对数正态分布的标准差
        cloud_bandwidth_range = (5e7, 3e8)    # 带宽范围 50-300 MHz
        
        # 创建距离矩阵（每个ES到云的距离）
        es_cloud_distance_matrix = es_to_cloud_distance.reshape(-1, 1)  # 转为列向量
        
        # 使用通用函数计算ES到云的传输速率
        r_es_to_cloud, g_es_to_cloud, B_es_to_cloud = calculate_transmission_rates(
            distance_matrix=es_cloud_distance_matrix,
            transmit_power=p_es,              # 使用与ES-ES相同的发射功率
            center_bandwidth=cloud_center_bandwidth,
            bandwidth_sigma=cloud_bandwidth_sigma,
            noise_density=N0,
            path_loss_exponent=path_loss_exponent,
            g0_at_1m=g0_at_1m,
            bandwidth_range=cloud_bandwidth_range
        )
        
        # 打印ES到云传输速率
        print("\n边缘服务器到云服务器的传输速率:")
        for idx in active_es_indices[:min(10, len(active_es_indices))]:
            rate_mbps = r_es_to_cloud[idx, 0] / 1e6  # 转换为Mbps
            print(f"  ES{idx} → Cloud: {rate_mbps:.2f} Mbps")

        if len(active_es_indices) > 10:
            print(f"  ... 以及其他 {len(active_es_indices) - 10} 个边缘服务器")

        # 计算平均ES-Cloud传输速率
        active_es_to_cloud_rates = r_es_to_cloud[active_es_indices, 0]
        print(f"活跃ES到云的平均传输速率: {np.mean(active_es_to_cloud_rates)/1e6:.2f} Mbps")
    else:
        print("警告: 由于缺少距离信息，无法计算ES到云的传输速率")
        r_es_to_cloud = None

    #=======================STEP 9: 计算客户端到云的直接传输速率========================#
    print("\n========== 计算客户端到云服务器(Client-Cloud)的直接传输速率 ==========")
    
    # 计算客户端到云的传输速率
    r_client_to_cloud = None
    if cloud_pos is not None:
        # 定义客户端到云通信的专属参数
        cloud_direct_bandwidth = 8e7          # 直连云带宽较低 (80 MHz)
        cloud_direct_sigma = 0.3              # 带宽对数正态分布的标准差
        cloud_direct_range = (1e7, 2e8)       # 带宽范围 10-200 MHz
        
        # 计算每个客户端到云的距离
        client_to_cloud_distance = np.zeros((M, 1))
        for idx in range(M):
            client_node = client_nodes[idx]
            if client_node in pos:
                client_pos = pos[client_node]
                # 计算地理距离(千米)并转换为米
                distance_km = calculate_distance(client_pos[1], client_pos[0], cloud_pos[1], cloud_pos[0])
                client_to_cloud_distance[idx, 0] = distance_km * 1000
            else:
                print(f"警告: 客户端 {client_node} 缺少位置信息，设置距离为无穷大")
                client_to_cloud_distance[idx, 0] = float('inf')
        
        # 使用通用函数计算客户端到云的传输速率
        r_client_to_cloud, g_client_to_cloud, B_client_to_cloud = calculate_transmission_rates(
            distance_matrix=client_to_cloud_distance,
            transmit_power=p_m,               # 使用客户端发射功率
            center_bandwidth=cloud_direct_bandwidth,
            bandwidth_sigma=cloud_direct_sigma,
            noise_density=N0,
            path_loss_exponent=path_loss_exponent,
            g0_at_1m=g0_at_1m,
            bandwidth_range=cloud_direct_range
        )
        
        # 打印客户端到云传输速率统计信息
        print(f"\n客户端到云服务器的直接传输速率统计:")
        avg_rate = np.mean(r_client_to_cloud)
        min_rate = np.min(r_client_to_cloud)
        max_rate = np.max(r_client_to_cloud)
        print(f"  平均传输速率: {avg_rate/1e6:.2f} Mbps")
        print(f"  最小传输速率: {min_rate/1e6:.2f} Mbps")
        print(f"  最大传输速率: {max_rate/1e6:.2f} Mbps")
        
        # 显示一些示例值
        sample_size = min(5, M)
        print(f"\n客户端到云的传输速率示例 (前{sample_size}个):")
        for i in range(sample_size):
            rate_mbps = r_client_to_cloud[i, 0] / 1e6
            print(f"  客户端{i} → 云: {rate_mbps:.2f} Mbps")
    else:
        print("警告: 由于缺少云服务器位置信息，无法计算客户端到云的直接传输速率")
    
    # 返回精简的计算结果
    return (client_nodes,          # 客户端节点列表
            active_es_nodes,       # 活跃边缘服务器节点列表
            association_matrix,    # 客户端到ES的关联矩阵
            r_client_to_es,        # 客户端到ES的传输速率矩阵
            r_es,                  # ES到ES的传输速率矩阵
            r_es_to_cloud,         # ES到云的传输速率矩阵
            r_client_to_cloud)     # 客户端到云的直接传输速率矩阵

def validate_results(B_cloud, B_n, r, assignments, loads, max_capacity):
    N = len(B_n)
    M = len([i for i, _ in assignments])
    active = np.unique([j for _, j in assignments])
    if abs(np.sum(B_n) - B_cloud) > 1e-6:
        print(f"Error: Sum of B_n ({np.sum(B_n)}) != B_cloud ({B_cloud})")
    else:
        print("Pass: B_n sum correct")
    for n in range(N):
        if n not in active and B_n[n] != 0:
            print(f"Error: Inactive edge {n} has B_n != 0")
        elif n in active and B_n[n] == 0:
            print(f"Error: Active edge {n} has B_n == 0")
    if np.any(loads > max_capacity):
        print(f"Error: Loads exceed capacity {max_capacity} (loads: {loads})")
    else:
        print("Pass: Capacity constraints satisfied")
    mounted_ratio = len(assignments) / M if M > 0 else 0
    if mounted_ratio < 0.95:
        print(f"Warning: Low assignment coverage ({mounted_ratio:.2%})")
    else:
        print(f"Pass: Assignment coverage {mounted_ratio:.2%}")
    for m, n in assignments:
        if r[m, n] <= 0:
            print(f"Error: Assigned device {m} to edge {n} has r <= 0")
        for other_n in range(N):
            if other_n != n and r[m, other_n] > 0:
                print(f"Error: Non-assigned device {m} to edge {other_n} has r > 0")
    if M > 0:
        max_r_values = np.max(r, axis=1)
        sorted_max_r = np.sort(max_r_values)[::-1]
        half = M // 2
        if np.all(sorted_max_r[:half] >= sorted_max_r[-half:]):
            print("Pass: Prioritized high-rate devices")
        else:
            print("Warning: Sorting may not prioritize high-rate devices")
    print(f"Manual check: Load std (balance) = {np.std(loads):.2f} (lower is better)")
    print("Optimization Version: Hungarian + Simulated Annealing")

def run_bandwidth_allocation(graphml_file=None, es_ratio=None, max_capacity=None, visualize=False):
    """
    运行带宽分配算法的主函数，包含整个流程。

    参数:
    - graphml_file: GraphML网络拓扑文件路径
    - es_ratio: 边缘服务器在总节点中的占比
    - max_capacity: 每个边缘服务器可容纳的最大客户端数
    - visualize: 是否生成可视化图

    返回:
    - bipartite_graph: 构建的二部图对象
    - client_nodes: 客户端节点列表
    - active_es_nodes: 活跃边缘服务器节点列表
    - association_matrix: 客户端到边缘服务器的关联矩阵
    - r_client_to_es: 客户端到边缘服务器的初始传输速率矩阵
    - r_es: 边缘服务器间的传输速率矩阵
    - r_es_to_cloud: 边缘服务器到云服务器的传输速率矩阵
    - r_client_to_cloud: 客户端到云服务器的直接传输速率矩阵
    """
    # 如果参数为None，从args_parser()获取默认值
    if graphml_file is None or es_ratio is None:
        args = args_parser()
        if graphml_file is None:
            graphml_file = args.graphml_file
        if es_ratio is None:
            es_ratio = args.es_ratio
        if max_capacity is None:
            max_capacity = args.max_capacity
    
    # 构建二部图
    bipartite_graph, client_nodes, es_nodes, distance_matrix, pos = build_bipartite_graph(
        graphml_file, es_ratio, visualize)
    if bipartite_graph is None:
        return None, [], [], None, None, None, None, None
    print("===============构建二部图==================")
    print("Bipartite Graph Built Successfully!")
    print(f"Total Client Nodes: {len(client_nodes)}")
    print(f"Total ES Nodes: {len(es_nodes)}")
    print(f"Distance Matrix Shape: {distance_matrix.shape}")
    print(f"Distance Matrix (first 5x5):\n{distance_matrix[:5, :min(5, len(es_nodes))]}")
    
    # 分配带宽并获取传输速率
    client_nodes, active_es_nodes, association_matrix, r_client_to_es, r_es, r_es_to_cloud, r_client_to_cloud = establish_communication_channels(
        client_nodes, es_nodes, distance_matrix, pos, max_capacity)
    
    if r_client_to_es is not None:
        print(f"Final transmission rates r_m,n (bit/s, first 5x5):\n{r_client_to_es[:min(5, len(client_nodes)), :min(5, len(es_nodes))]}")
    
    # # 使用与establish_communication_channels中相同的max_capacity值进行验证
    # validate_results(B_cloud=5e7, B_n=B_n, r=r, assignments=assignments, loads=loads, max_capacity=max_capacity)
    
    # 返回精简的结果集
    return bipartite_graph, client_nodes, active_es_nodes, association_matrix, r_client_to_es, r_es, r_es_to_cloud, r_client_to_cloud


if __name__ == '__main__':
    args = args_parser()
    build_bipartite_graph(graphml_file=args.graphml_file, es_ratio=args.es_ratio)
