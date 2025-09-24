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


def visualize_node_distribution(client_nodes, es_nodes, pos, es_ratio, save_path=None, cloud_pos=None):
    """
    生成节点地理分布可视化图
    
    Args:
        client_nodes: 客户端节点列表
        es_nodes: 边缘服务器节点列表
        pos: 节点位置字典 {node_id: (lon, lat)}
        es_ratio: 边缘服务器占比
        save_path: 保存路径（可选）
        cloud_pos: 云服务器位置 (lon, lat)，如果提供则显示
    
    Returns:
        str: 保存的文件名
    """
    plt.figure(figsize=(12, 8))
    
    # 绘制客户端节点
    client_positions = np.array([pos[node] for node in client_nodes if node in pos])
    if len(client_positions) > 0:
        plt.scatter(client_positions[:, 0], client_positions[:, 1], 
                   c='#719aac', marker='o', s=100, alpha=0.7, label=f'Clients ({len(client_nodes)})')
    
    # 绘制边缘服务器节点
    es_positions = np.array([pos[node] for node in es_nodes if node in pos])
    if len(es_positions) > 0:
        plt.scatter(es_positions[:, 0], es_positions[:, 1], 
                   c='#e29135', marker='^', s=100, 
                   label=f'Edge Servers ({len(es_nodes)})')
    
    # 添加边缘服务器编号标注
    for i, node in enumerate(es_nodes):
        if node in pos:
            lon, lat = pos[node]
            plt.annotate(f'ES{i}', (lon, lat), xytext=(5, 5), 
                       textcoords='offset points', fontsize=8, fontweight='bold')
                       
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
    plt.title(f'Geographic Distribution of Nodes\n(ES Ratio: {es_ratio:.2f}, Total: {total_nodes} nodes)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存可视化图
    if save_path:
        viz_filename = save_path
    else:
        viz_filename = f"./save/node_distribution_es{len(es_nodes)}_ratio{es_ratio:.2f}.png"
    
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

def allocate_bandwidth_eba(client_nodes, es_nodes, distance_matrix, pos, model_size=1e7, max_capacity=None):
    """
    基于边缘计算的带宽分配算法 (Edge-Based Allocation)
    
    主要流程:
    1. 计算信道增益和传输速率
    2. 使用匈牙利算法进行初始分配
    3. 处理容量约束违反
    4. 模拟退火优化分配方案
    5. 计算边缘服务器间传输速率
    6. 分配云端带宽
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
    # 1. 初始化参数
    M = len(client_nodes)  # 客户端数量
    N = len(es_nodes)      # 边缘服务器数量
    B_cloud = 5e7          # 总云端带宽
    p_m = np.ones(M) * 1.0  # 设备发射功率
    N0 = 10**(-20.4)        # 噪声功率谱密度
    path_loss_exponent = 3.5 # 路径损耗指数 (alpha)
    # 在参考距离d0=1米处的信道增益
    g0_at_1m = 1e-4

    #=======================client-es========================#

    # 2-4. 计算客户端到边缘服务器的传输速率
    client_center_bandwidth = 2e7  # 20 MHz
    client_bandwidth_sigma = 0.5
    client_bandwidth_range = (1e6, 5e7)
    
    # 使用通用函数计算传输速率
    r_initial, g, B_mn = calculate_transmission_rates(
        distance_matrix=distance_matrix,
        transmit_power=p_m,
        center_bandwidth=client_center_bandwidth,
        bandwidth_sigma=client_bandwidth_sigma,
        noise_density=N0,
        path_loss_exponent=path_loss_exponent,
        g0_at_1m=g0_at_1m,
        bandwidth_range=client_bandwidth_range
    )
    
    print(f"Channel gain (g, first 5x5):\n{g[:min(5, M), :min(5, N)]}\n")
    print(f"Device-to-Edge Bandwidth (B_mn, first 5x5):\n{B_mn[:min(5, M), :min(5, N)]}\n")
    print(f"Initial transmission rates (r_initial, first 5x5):\n{r_initial[:min(5, M), :min(5, N)]}")

    # 5. 基于最大传输速率为每个客户端选择边缘服务器
    assignments_client_to_es, association_matrix = create_association_based_on_rate(r_initial)
    print(f"Created association matrix based on maximum transmission rate")
    print(f"Total associations: {len(assignments_client_to_es)}")
    print(f"Association matrix shape: {association_matrix.shape}")
    print(f"Association matrix (first 5x5):\n{association_matrix[:min(5, M), :min(5, N)]}")

    #=======================es-es========================#
    
    # 6. 根据关联矩阵确定活跃边缘服务器节点
    # 找出与至少一个客户端关联的边缘服务器
    active_es_indices = np.where(np.sum(association_matrix, axis=0) > 0)[0]
    active_es_nodes = [es_nodes[i] for i in active_es_indices]
    N_active = len(active_es_nodes)
    print(f"Active edge servers: {len(active_es_nodes)}/{N} ({len(active_es_nodes)/N:.2%})")
    print(f"Active edge server indices: {active_es_indices}")
    
    # 7. 生成活跃边缘服务器之间的距离矩阵
    active_es_distance_matrix = np.zeros((len(active_es_nodes), len(active_es_nodes)))
    for i, idx1 in enumerate(active_es_indices):
        e1 = es_nodes[idx1]
        for j, idx2 in enumerate(active_es_indices):
            e2 = es_nodes[idx2]
            if e1 in pos and e2 in pos:
                e1_pos = pos[e1]
                e2_pos = pos[e2]
                active_es_distance_matrix[i, j] = calculate_distance(e1_pos[1], e1_pos[0], e2_pos[1], e2_pos[0]) * 1000
            else:
                print(f"Warning: No position for active edge {e1} or {e2}, setting distance to infinity")
                active_es_distance_matrix[i, j] = float('inf')
    
    #=======================确定cloud========================#

    # 8. 计算活跃边缘服务器的地理质心作为云服务器的位置
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
            
            print(f"Cloud server positioned at the centroid of active ESs: lon={cloud_lon:.6f}, lat={cloud_lat:.6f}")
            
            # 9. 计算每个边缘服务器到云的距离
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
                    print(f"Warning: No position for edge server {es_node}, setting distance to infinity")
                    es_to_cloud_distance[idx] = float('inf')
        else:
            print("Warning: No position information for active edge servers, cannot calculate cloud position")
            cloud_pos = None
            es_to_cloud_distance = None
    else:
        print("Warning: No active edge servers found, cannot calculate cloud position")
        cloud_pos = None
        es_to_cloud_distance = None

    print(f"Generated active ES distance matrix with shape: {active_es_distance_matrix.shape}")
    if len(active_es_nodes) > 0:
        print(f"Active ES distance matrix (first 5x5):\n{active_es_distance_matrix[:min(5, len(active_es_nodes)), :min(5, len(active_es_nodes))]}")


    # 10. 计算边缘服务器间的传输速率

    # 定义ES-to-ES通信的专属参数
    p_es = np.ones(N) * 5.0  # ES通常有更高的发射功率
    es_center_bandwidth = 1e8  # 100 MHz - ES间通常有更高带宽
    es_bandwidth_sigma = 0.3
    es_bandwidth_range = (2e7, 2e8)  # 20MHz - 200MHz
    
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

    # --- 打印结果用于验证 ---
    print(f"Active ES-to-ES Channel Gain (g_es, first 5x5):\n{g_es[:min(5, N_active), :min(5, N_active)]}\n")
    print(f"Active ES-to-ES Bandwidth (B_es, first 5x5):\n{B_es[:min(5, N_active), :min(5, N_active)]}\n")
    print(f"Active ES-to-ES Transmission Rates (r_es_active, first 5x5):\n{r_es_active[:min(5, N_active), :min(5, N_active)]}")
    print(f"Full ES-to-ES Transmission Rates (r_es, first 5x5):\n{r_es[:min(5, N), :min(5, N)]}")
    
    # 11. 计算ES到云的传输速率
    if 'es_to_cloud_distance' in locals() and es_to_cloud_distance is not None:
        # 定义ES-to-Cloud通信的专属参数
        cloud_center_bandwidth = 1.2e8  # 120 MHz - 云连接通常带宽更高
        cloud_bandwidth_sigma = 0.25
        cloud_bandwidth_range = (5e7, 3e8)  # 50MHz - 300MHz
        
        # 创建距离矩阵（每个ES到云的距离）
        es_cloud_distance_matrix = es_to_cloud_distance.reshape(-1, 1)  # 转为列向量
        
        # 使用通用函数计算ES到云的传输速率
        r_es_to_cloud, g_es_to_cloud, B_es_to_cloud = calculate_transmission_rates(
            distance_matrix=es_cloud_distance_matrix,
            transmit_power=p_es,  # 使用与ES-ES相同的发射功率
            center_bandwidth=cloud_center_bandwidth,
            bandwidth_sigma=cloud_bandwidth_sigma,
            noise_density=N0,
            path_loss_exponent=path_loss_exponent,
            g0_at_1m=g0_at_1m,
            bandwidth_range=cloud_bandwidth_range
        )
        
        print("\n边缘服务器到云服务器的传输速率:")
        for idx in active_es_indices[:min(10, len(active_es_indices))]:
            rate_mbps = r_es_to_cloud[idx, 0] / 1e6  # 转换为Mbps
            print(f"  ES{idx} → Cloud: {rate_mbps:.2f} Mbps")
        if len(active_es_indices) > 10:
            print(f"  ... 以及其他 {len(active_es_indices) - 10} 个边缘服务器")
    else:
        print("Warning: Cannot calculate ES-to-Cloud transmission rates due to missing distance information")
        r_es_to_cloud = None
        g_es_to_cloud = None
        B_es_to_cloud = None
    
    # 12. 分配云端带宽给活跃边缘服务器
    B_n = np.zeros(N)
    min_bandwidth = 1e6
    
    # 根据负载权重分配云端带宽
    if N_active > 0:
        mean_B = 5e7
        std_B = 1e7
        base_bandwidths = np.random.normal(loc=mean_B, scale=std_B, size=N_active)
        base_bandwidths = np.clip(base_bandwidths, mean_B / 2, mean_B * 2)
        load_weights = loads[active_indices] / np.sum(loads[active_indices])
        B_n[active_indices] = base_bandwidths * load_weights * (B_cloud / np.sum(base_bandwidths * load_weights))
    B_n[B_n == 0] = min_bandwidth
    print(f"Active edge servers (indices): {active_indices}")
    print(f"Active edge servers (nodes): {active_es_nodes}")
    print(f"Edge-to-Cloud Bandwidth (B_n): {B_n}")

    # 12. 输出统计信息
    r_values = r[r > 0]
    r_random_values = r_random[r_random > 0]
    print(f"Bipartite r stats: mean={np.mean(r_values):.2e}, max={np.max(r_values):.2e}, min={np.min(r_values):.2e}")
    print(f"Random r stats: mean={np.mean(r_random_values):.2e}, max={np.max(r_random_values):.2e}, min={np.min(r_random_values):.2e}")
    print("Optimization Version: Hungarian + Simulated Annealing (Maximize Min Rate)")

    return r, r_random, assignments, loads, B_n, r_es, pos, active_es_nodes, r_initial, association_matrix, active_es_distance_matrix, cloud_pos, r_es_to_cloud

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

def run_bandwidth_allocation(graphml_file=None, model_size=1e7, es_ratio=None, max_capacity=None, visualize=False):
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
        return None, [], [], None, [], [], [], None, None, [], None, None
    print("===============构建二部图==================")
    print("Bipartite Graph Built Successfully!")
    print(f"Total Client Nodes: {len(client_nodes)}")
    print(f"Total ES Nodes: {len(es_nodes)}")
    print(f"Distance Matrix Shape: {distance_matrix.shape}")
    print(f"Distance Matrix (first 5x5):\n{distance_matrix[:5, :min(5, len(es_nodes))]}")
    r, r_random, assignments, loads, B_n, r_es, pos, active_es_nodes, r_initial, association_matrix, active_es_distance_matrix = allocate_bandwidth_eba(
        client_nodes, es_nodes, distance_matrix, pos, model_size, max_capacity)
    print(f"Final transmission rates r_m,n (bit/s, first 5x5):\n{r[:5, :min(5, len(es_nodes))]}")
    # 使用与allocate_bandwidth_eba中相同的max_capacity值进行验证
    validate_results(B_cloud=5e7, B_n=B_n, r=r, assignments=assignments, loads=loads, max_capacity=max_capacity)
    
    # 如果需要可视化并且有云服务器位置，再次生成带有云服务器的可视化
    if visualize and 'cloud_pos' in locals() and cloud_pos is not None:
        visualize_node_distribution(client_nodes, es_nodes, pos, es_ratio, cloud_pos=cloud_pos)
        print(f"Cloud Server Position: Longitude={cloud_pos[0]:.4f}, Latitude={cloud_pos[1]:.4f}")
        print(f"ES-to-Cloud Transmission Rates (Mbps):")
        for i, es in enumerate(active_es_nodes):
            print(f"  ES{i} to Cloud: {r_es_to_cloud[i]/1e6:.2f} Mbps")
    
    print("Optimization Version: Hungarian + Simulated Annealing")
    return bipartite_graph, client_nodes, es_nodes, distance_matrix, r, r_random, assignments, loads, B_n, r_es, pos, active_es_nodes, r_initial, association_matrix, active_es_distance_matrix, cloud_pos, r_es_to_cloud


if __name__ == '__main__':
    args = args_parser()
    build_bipartite_graph(graphml_file=args.graphml_file, es_ratio=args.es_ratio)