import random
import math
import networkx as nx
import numpy as np
# import cupy as cp
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from utils.options import args_parser

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

    # print("distance_matrix:", distance_matrix)
    # print("channel_gain:", channel_gain)
    # print("bandwidth:", bandwidth)
    # print("received_power:", received_power)
    # print("noise_power:", noise_power)
    # print("snr:", snr)
    # print("rate:", rate)
    #
    return rate, channel_gain, bandwidth

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    # cp.random.seed(seed)

def select_edge_servers_uniformly(node_ids, pos, es_ratio):
    """
    使用K-Means聚类和Medoid选择法，选择与节点密度成正比的边缘服务器。
    客户端密集的区域，边缘服务器也会更密集。

    Args:
        node_ids: 所有有效节点ID列表
        pos: 节点位置字典 {node_id: (lon, lat)}
        es_ratio: 边缘服务器占比

    Returns:
        tuple: (边缘服务器节点列表, 客户端节点列表)
    """
    if not node_ids:
        return [], []
        
    num_es = max(1, int(len(node_ids) * es_ratio))
    print(f"Selecting {num_es} edge servers from {len(node_ids)} nodes (ratio: {es_ratio:.2f}) using K-Means + Medoid")

    # 1. 准备用于聚类的位置数据
    # 使用一个有序列表来确保节点ID和位置的索引一致
    ordered_node_ids = sorted(list(pos.keys()))
    positions = np.array([pos[node_id] for node_id in ordered_node_ids])
    
    # 2. 使用K-Means将节点分区
    # n_init='auto' 是新版本 scikit-learn 的推荐用法
    kmeans = KMeans(n_clusters=num_es, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(positions)
    
    es_nodes = []
    # 3. 为每个聚类选择Medoid作为边缘服务器
    for i in range(num_es):
        # 找到当前聚类的所有节点的索引
        indices_in_cluster = np.where(cluster_labels == i)[0]
        
        if len(indices_in_cluster) == 0:
            continue # 如果某个簇是空的，则跳过

        # 获取该簇的所有节点ID和它们的坐标
        nodes_in_cluster = [ordered_node_ids[j] for j in indices_in_cluster]
        positions_in_cluster = positions[indices_in_cluster]
        
        # 计算该簇内节点间的距离矩阵
        # 使用广播和向量化操作，避免双重循环，提高效率
        distances = np.linalg.norm(positions_in_cluster[:, np.newaxis, :] - positions_in_cluster[np.newaxis, :, :], axis=2)
        
        # 计算每个节点到簇内其他所有节点的距离之和
        sum_of_distances = np.sum(distances, axis=1)
        
        # 找到距离之和最小的那个节点的索引（即Medoid的索引）
        medoid_index_in_cluster = np.argmin(sum_of_distances)
        
        # 从该簇的节点列表中选出Medoid
        best_node = nodes_in_cluster[medoid_index_in_cluster]
        es_nodes.append(best_node)
        
    # 确保边缘服务器数量正确 (以防有空簇)
    if len(es_nodes) < num_es:
        remaining_nodes = [n for n in node_ids if n not in es_nodes]
        # 从尚未被选中的节点中随机补充，直到数量达标
        import random
        needed = num_es - len(es_nodes)
        es_nodes.extend(random.sample(remaining_nodes, min(needed, len(remaining_nodes))))

    client_nodes = [node for node in node_ids if node not in es_nodes]
    
    print(f"Selected edge servers (Medoids): {es_nodes}")
    print(f"Client nodes count: {len(client_nodes)}, Edge servers count: {len(es_nodes)}")
    
    return es_nodes, client_nodes

'''
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
'''

def visualize_node_distribution(client_nodes, es_nodes, pos, es_ratio, save_path=None, cloud_pos=None, association_matrix=None, filter_info=None):
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
        filter_info: 地理筛选信息字典，包含 {'center': (lat, lon), 'radius': float}
    
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
    
    # 如果提供了筛选信息，绘制筛选范围圆圈
    if filter_info is not None and 'center' in filter_info and 'radius' in filter_info:
        center_lat, center_lon = filter_info['center']
        radius_km = filter_info['radius']
        
        # 绘制筛选中心点
        plt.scatter([center_lon], [center_lat], 
                   c='#ff6b6b', marker='x', s=200, linewidths=3,
                   label=f'Filter Center', zorder=5)
        plt.annotate(f'Filter Center\n({center_lat:.3f}, {center_lon:.3f})', 
                   (center_lon, center_lat), xytext=(10, 10),
                   textcoords='offset points', fontsize=8, fontweight='bold', 
                   color='#ff6b6b', zorder=5)
        
        # 计算在经纬度坐标系下的近似半径（简化计算）
        # 1度纬度约等于111km，经度则取决于纬度
        lat_radius = radius_km / 111.0  # 纬度半径
        lon_radius = radius_km / (111.0 * np.cos(np.radians(center_lat)))  # 经度半径
        
        # 绘制筛选范围圆圈
        circle = plt.Circle((center_lon, center_lat), max(lat_radius, lon_radius), 
                          fill=False, color='#ff6b6b', linewidth=2, linestyle='--', 
                          alpha=0.7, zorder=1)
        plt.gca().add_patch(circle)
        
        # 使用椭圆更准确地表示地理范围（考虑经纬度差异）
        from matplotlib.patches import Ellipse
        ellipse = Ellipse((center_lon, center_lat), 2*lon_radius, 2*lat_radius, 
                         fill=False, color='#ff6b6b', linewidth=2, linestyle=':', 
                         alpha=0.5, zorder=1)
        plt.gca().add_patch(ellipse)
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    total_nodes = len(client_nodes) + len(es_nodes)
    
    # 根据是否包含关联边、云服务器和筛选信息修改标题
    title = f'Geographic Distribution of Nodes\n(ES Ratio: {es_ratio:.2f}, Total: {total_nodes} nodes)'
    if association_matrix is not None:
        title += ' with Associations'
    if cloud_pos is not None:
        title += ' and Cloud Server'
    if filter_info is not None:
        if 'radius_ratio' in filter_info:
            title += f'\nFiltered (R={filter_info["radius"]:.1f}km, Ratio={filter_info["radius_ratio"]:.2f})'
        else:
            title += f'\nFiltered (R={filter_info["radius"]:.1f}km)'
    plt.title(title)
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存可视化图，文件名根据内容区分
    if save_path:
        viz_filename = save_path
    else:
        viz_filename = f"./save/node_distribution_es{len(es_nodes)}_ratio{es_ratio:.2f}"
        if filter_info is not None:
            if 'radius_ratio' in filter_info:
                viz_filename += f"_filtered_ratio{filter_info['radius_ratio']:.2f}_r{filter_info['radius']:.0f}km"
            else:
                viz_filename += f"_filtered_r{filter_info['radius']:.0f}km"
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

def filter_nodes_by_geographic_range(G, node_ids, pos, filter_radius_ratio=0.3, 
                                   center_lat=None, center_lon=None):
    """
    基于地理范围筛选节点：以指定中心点为圆心，在指定半径范围内筛选节点
    筛选半径 = 图地理范围 * 范围比例
    
    Args:
        G: NetworkX图对象
        node_ids: 所有有效节点ID列表
        pos: 节点位置字典 {node_id: (lon, lat)}
        filter_radius_ratio: 筛选半径比例（相对于图的地理范围）
        center_lat: 筛选中心纬度（如果为None，使用所有节点的质心）
        center_lon: 筛选中心经度（如果为None，使用所有节点的质心）
    
    Returns:
        tuple: (筛选后的图对象, 筛选后的节点ID列表, 筛选后的位置字典, 实际使用的中心点)
    """
    if not node_ids:
        print("Warning: No valid nodes to filter")
        return G, [], {}, None
        
    print(f"\n========== 地理范围节点筛选 ==========")
    print(f"筛选前节点总数: {len(node_ids)}")
    print(f"筛选半径比例: {filter_radius_ratio:.2f}")
    
    # 计算图的地理范围
    latitudes = [pos[node_id][1] for node_id in node_ids if node_id in pos]
    longitudes = [pos[node_id][0] for node_id in node_ids if node_id in pos]
    
    if not latitudes or not longitudes:
        print("Error: No valid positions to calculate graph range")
        return G, [], {}, None
    
    lat_min, lat_max = min(latitudes), max(latitudes)
    lon_min, lon_max = min(longitudes), max(longitudes)
    
    # 计算图的地理范围（使用对角线距离作为参考）
    graph_diagonal_km = calculate_distance(lat_min, lon_min, lat_max, lon_max)
    graph_range_km = graph_diagonal_km / 2  # 使用对角线的一半作为基准范围
    
    # 根据比例计算实际筛选半径
    filter_radius = graph_range_km * filter_radius_ratio
    
    print(f"图地理范围统计:")
    print(f"  纬度范围: [{lat_min:.4f}, {lat_max:.4f}] (跨度: {lat_max-lat_min:.4f}°)")
    print(f"  经度范围: [{lon_min:.4f}, {lon_max:.4f}] (跨度: {lon_max-lon_min:.4f}°)")
    print(f"  对角线距离: {graph_diagonal_km:.2f} km")
    print(f"  基准范围半径: {graph_range_km:.2f} km")
    print(f"  实际筛选半径: {filter_radius:.2f} km (比例: {filter_radius_ratio:.2f})")
    
    # 如果未指定中心点，计算所有节点的地理质心
    if center_lat is None or center_lon is None:
        center_lat = np.mean(latitudes)
        center_lon = np.mean(longitudes)
        print(f"使用图质心作为筛选中心: ({center_lat:.4f}, {center_lon:.4f})")
    else:
        print(f"使用指定中心点: ({center_lat:.4f}, {center_lon:.4f})")
    
    # 筛选在指定范围内的节点
    filtered_node_ids = []
    removed_node_ids = []
    filtered_pos = {}
    
    for node_id in node_ids:
        if node_id not in pos:
            removed_node_ids.append(node_id)
            continue
            
        lon, lat = pos[node_id]
        distance = calculate_distance(center_lat, center_lon, lat, lon)
        
        if distance <= filter_radius:
            filtered_node_ids.append(node_id)
            filtered_pos[node_id] = pos[node_id]
        else:
            removed_node_ids.append(node_id)
    
    # 从图中移除超出范围的节点
    if removed_node_ids:
        G_filtered = G.copy()
        G_filtered.remove_nodes_from(removed_node_ids)
        print(f"移除超出范围的节点数: {len(removed_node_ids)}")
        print(f"筛选后节点数: {len(filtered_node_ids)}")
        print(f"节点保留率: {len(filtered_node_ids)/len(node_ids):.2%}")
    else:
        G_filtered = G
        print("所有节点都在筛选范围内")
    
    # 打印距离统计
    if filtered_node_ids:
        distances = [calculate_distance(center_lat, center_lon, pos[node_id][1], pos[node_id][0]) 
                    for node_id in filtered_node_ids if node_id in pos]
        if distances:
            print(f"筛选节点到中心距离统计:")
            print(f"  最小距离: {min(distances):.2f} km")
            print(f"  最大距离: {max(distances):.2f} km")
            print(f"  平均距离: {np.mean(distances):.2f} km")
            print(f"  距离标准差: {np.std(distances):.2f} km")
    
    center_point = (center_lat, center_lon)
    return G_filtered, filtered_node_ids, filtered_pos, center_point

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
    
    # 地理范围节点筛选（如果启用）
    try:
        args = args_parser()
        if args.enable_node_filter:
            print(f"地理节点筛选已启用")
            G, node_ids, pos, filter_center = filter_nodes_by_geographic_range(
                G, node_ids, pos, 
                filter_radius_ratio=args.filter_radius_ratio,
                center_lat=args.filter_center_lat,
                center_lon=args.filter_center_lon
            )
            
            if not node_ids:
                print("Error: No nodes remaining after geographic filtering")
                return None, [], [], None, None, None
                
            print(f"地理筛选完成，剩余节点数: {len(node_ids)}")
        else:
            print(f"地理节点筛选未启用，使用所有有效节点: {len(node_ids)}")
    except Exception as e:
        print(f"Warning: Geographic filtering failed, using all valid nodes: {e}")

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
    print(f"min dist {np.min(distance_matrix, axis=1)}")
    # 打印节点样例信息
    if client_nodes:
        sample_node = G.nodes[client_nodes[0]]
        print(f"Node data sample: {sample_node}")

    # 准备筛选信息（如果启用了筛选）
    filter_info = None
    try:
        args = args_parser()
        if args.enable_node_filter:
            # 获取筛选中心点信息
            if args.filter_center_lat is not None and args.filter_center_lon is not None:
                center = (args.filter_center_lat, args.filter_center_lon)
            else:
                # 使用所有节点的质心
                all_nodes = client_nodes + es_nodes
                latitudes = [pos[node][1] for node in all_nodes if node in pos]
                longitudes = [pos[node][0] for node in all_nodes if node in pos]
                center = (np.mean(latitudes), np.mean(longitudes))
            
            # 计算实际的筛选半径用于显示
            all_nodes = client_nodes + es_nodes
            all_latitudes = [pos[node][1] for node in all_nodes if node in pos]
            all_longitudes = [pos[node][0] for node in all_nodes if node in pos]
            if all_latitudes and all_longitudes:
                lat_min, lat_max = min(all_latitudes), max(all_latitudes)
                lon_min, lon_max = min(all_longitudes), max(all_longitudes)
                graph_diagonal_km = calculate_distance(lat_min, lon_min, lat_max, lon_max)
                graph_range_km = graph_diagonal_km / 2
                actual_filter_radius = graph_range_km * args.filter_radius_ratio
            else:
                actual_filter_radius = 0
            
            filter_info = {
                'center': center,
                'radius': actual_filter_radius,
                'radius_ratio': args.filter_radius_ratio
            }
    except:
        pass  # 如果获取筛选信息失败，filter_info保持为None

    return bipartite_graph, client_nodes, es_nodes, distance_matrix, pos, filter_info

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


import numpy as np
from scipy.optimize import linear_sum_assignment

import pulp
import numpy as np
import math

'''
def create_association_based_on_rate(rate_matrix, load_range_tolerance=0):
    """
    使用整数线性规划 (ILP) 找到最优分配方案。
    目标是最大化所有分配中最低的通信速率 (Max-Min Fairness)，
    同时满足负载均衡约束。

    Args:
        rate_matrix (np.array): M x N 的速率矩阵。
        load_range_tolerance (int): 负载范围的容忍度。

    Returns:
        tuple: (分配列表, 关联矩阵)
               或在无解时返回 (None, None, None, None)。
    """
    M, N = rate_matrix.shape

    # 1. 创建问题实例
    prob = pulp.LpProblem("Client_Assignment_MaxMin_Rate", pulp.LpMaximize)

    # 2. 定义决策变量
    # x_mn = 1 if client m is assigned to server n, else 0
    x = pulp.LpVariable.dicts("x", (range(M), range(N)), cat='Binary')

    # --- 新增部分：为最低速率创建一个辅助变量 ---
    R_min = pulp.LpVariable("min_rate", lowBound=0, cat='Continuous')

    # 3. 定义目标函数
    # --- 修改目标函数为最大化 R_min ---
    prob += R_min

    # 4. 添加约束条件
    # --- 新增 Big-M 约束 ---
    # BigM 需要比任何可能的速率值都大
    BigM = np.max(rate_matrix) * 2
    for m in range(M):
        for n in range(N):
            prob += R_min <= rate_matrix[m][n] + BigM * (1 - x[m][n])

    # 约束: 每个客户端必须且只能分配给一个服务器
    for m in range(M):
        prob += pulp.lpSum([x[m][n] for n in range(N)]) == 1

    # 约束: 负载均衡
    avg_load = M / N
    min_load_allowed = 2# max(0, math.floor(avg_load) - load_range_tolerance)
    max_load_allowed = 3#math.ceil(avg_load) + load_range_tolerance

    print(f"Average load: {avg_load:.2f}")
    print(f"Load constraint per server set to range: [{min_load_allowed}, {max_load_allowed}]")

    for n in range(N):
        prob += pulp.lpSum([x[m][n] for m in range(M)]) <= max_load_allowed
        prob += pulp.lpSum([x[m][n] for m in range(M)]) >= min_load_allowed

    # 5. 求解问题
    solver = pulp.PULP_CBC_CMD(msg=0)
    prob.solve(solver)

    # 6. 解析并返回结果
    if pulp.LpStatus[prob.status] == 'Optimal':
        assignment_list = []
        association_matrix = np.zeros((M, N), dtype=int)
        for m in range(M):
            for n in range(N):
                if pulp.value(x[m][n]) == 1:
                    assignment_list.append((m, n))
                    association_matrix[m, n] = 1
                    break

        final_loads = np.sum(association_matrix, axis=0)
        maximized_min_rate = pulp.value(R_min)

        return assignment_list, association_matrix
    else:
        print(f"ILP solver could not find an optimal solution. Status: {pulp.LpStatus[prob.status]}")
        return None, None
'''
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
            # print(f"Using max_capacity: {max_capacity} from command line arguments")
        else:
            max_capacity = max(1, int(len(client_nodes) / len(es_nodes)) + 1)
            # print(f"Automatically calculated max_capacity: {max_capacity} based on {len(client_nodes)} clients and {len(es_nodes)} edge servers")
            
    # 1. 初始化系统参数
    M = len(client_nodes)      # 客户端数量
    N = len(es_nodes)          # 边缘服务器数量
    # B_cloud = 5e7              # 总云端带宽 (50 MHz)
    # p_m = np.ones(M) * 10.0    # 客户端发射功率 (W)
    N0 = 10**(-20.4)           # 噪声功率谱密度 (W/Hz)
    path_loss_exponent = 2.5   # 路径损耗指数 (alpha)
    # g0_at_1m = 1e-4            # 在参考距离d0=1米处的信道增益
    g0_client_es = 1e-3
    g0_es_es = 1e-2
    p_client = np.ones(M) * 20.0  # 客户端发射功率向量 (W)
    p_es = np.ones(N) * 50.0      # ES发射功率向量 (W)

    #=======================STEP 1: 计算client-es传输速率========================#
    print("\n========== 计算客户端到边缘服务器(client-es)传输速率 ==========")
    
    # 设置客户端到边缘服务器的通信参数
    client_center_bandwidth =8e7        # 中心带宽 80 MHz
    client_bandwidth_sigma = 0.5         # 带宽对数正态分布的标准差
    client_bandwidth_range = (5e7, 1e8)  # 带宽范围 50-100 MHz
    
    # 使用通用函数计算传输速率
    r_client_to_es, g, B_mn = calculate_transmission_rates(
        distance_matrix=distance_matrix,
        transmit_power=p_client,              # 客户端发射功率
        center_bandwidth=client_center_bandwidth,
        bandwidth_sigma=client_bandwidth_sigma,
        noise_density=N0,
        path_loss_exponent=path_loss_exponent,
        g0_at_1m=g0_client_es,
        bandwidth_range=client_bandwidth_range
    )

    
    # 打印计算结果示例
    # print(f"信道增益矩阵 (g, 前5x5):\n{g[:min(5, M), :min(5, N)]}\n")
    # print(f"带宽分配矩阵 (B_mn, 前5x5):\n{B_mn[:min(5, M), :min(5, N)]}\n")
    # print(f"初始传输速率矩阵 (r_client_to_es, 前5x5) [bit/s]:\n{r_client_to_es[:min(5, M), :min(5, N)]}")
    #
    #=======================STEP 2: 根据传输速率创建关联矩阵========================#
    print("\n========== 基于最大传输速率为客户端分配边缘服务器 ==========")

    # 基于最大传输速率为每个客户端选择边缘服务器
    assignments, original_association_matrix = create_association_based_on_rate(r_client_to_es)
    # print(f"创建了基于最大传输速率的关联矩阵")
    # print(f"总关联数: {len(assignments)}/{M} ({len(assignments)/M:.2%})")
    # print(f"原始关联矩阵形状: {original_association_matrix.shape}")
    # print(f"原始关联矩阵示例 (前5x5):\n{original_association_matrix[:min(5, M), :min(5, N)]}")

    #=======================STEP 3: 确定活跃边缘服务器集合并生成新关联矩阵========================#
    print("\n========== 根据关联矩阵确定活跃边缘服务器集合并生成(client, active_es)关联矩阵 ==========")

    # 找出与至少一个客户端关联的边缘服务器(活跃边缘服务器)
    active_es_indices = np.where(np.sum(original_association_matrix, axis=0) > 0)[0]
    active_es_nodes = [es_nodes[i] for i in active_es_indices]
    N_active = len(active_es_nodes)

    print(f"活跃边缘服务器数量: {N_active}/{N} ({N_active/N:.2%})")
    # print(f"活跃边缘服务器索引: {active_es_indices}")

    # 删除空列，生成(client, active_es)关联矩阵
    association_matrix = original_association_matrix[:, active_es_indices]
    # print(f"删除空列后的关联矩阵形状: {association_matrix.shape} (client×active_es)")
    # print(f"新关联矩阵示例 (前5x5):\n{association_matrix}")

    # 同时调整r_client_to_es，只保留客户端到活跃ES的传输速率
    r_client_to_active_es = r_client_to_es[:, active_es_indices]
    # print(f"调整后的客户端到活跃ES传输速率矩阵形状: {r_client_to_active_es.shape} (client×active_es)")
    # print(f"客户端到活跃ES传输速率示例 (前5x5) [bit/s]:\n{r_client_to_active_es[:min(5, M), :min(5, N_active)]}")

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
    print(f"min dist {np.min(active_es_distance_matrix, axis=1)}")
    print(f"生成活跃ES距离矩阵，形状: {active_es_distance_matrix.shape}")
    # if N_active > 0:
    #     print(f"活跃ES距离矩阵示例 (前5x5) [米]:\n{active_es_distance_matrix[:min(5, N_active), :min(5, N_active)]}")
    #
    #=======================STEP 5: 计算活跃es-es传输速率========================#
    print("\n========== 计算活跃边缘服务器之间的传输速率 ==========")
    
    # 定义ES-to-ES通信的专属参数
    # p_es = np.ones(N) * 5.0            # ES通常有更高的发射功率 (5W)
    es_center_bandwidth = 1e8          # ES间通常有更高带宽 (800 MHz)
    es_bandwidth_sigma = 0.3           # 带宽对数正态分布的标准差
    es_bandwidth_range = (8e7, 1.2e8)    # 带宽范围 600-1000 MHz
    
    # 使用通用函数计算活跃ES之间的传输速率
    r_es_active, g_es, B_es = calculate_transmission_rates(
        distance_matrix=active_es_distance_matrix,
        transmit_power=p_es,
        center_bandwidth=es_center_bandwidth,
        bandwidth_sigma=es_bandwidth_sigma,
        noise_density=N0,
        path_loss_exponent=path_loss_exponent,
        g0_at_1m=g0_es_es,
        bandwidth_range=es_bandwidth_range,
        is_diagonal_zero=True,  # 对角线元素置零（自己到自己）
        node_indices=active_es_indices  # 使用活跃ES的索引
    )
    
    # # 创建完整的r_es矩阵，默认为0
    # r_es = np.zeros((N, N))
    
    # # 将活跃ES之间的传输速率填入完整矩阵
    # for i, idx_i in enumerate(active_es_indices):
    #     for j, idx_j in enumerate(active_es_indices):
    #         if idx_i != idx_j:
    #             r_es[idx_i, idx_j] = r_es_active[i, j]

    # 打印结果示例
    # print(f"活跃ES间信道增益示例 (前5x5):\n{g_es[:min(5, N_active), :min(5, N_active)]}\n")
    # print(f"活跃ES间带宽分配示例 (前5x5) [Hz]:\n{B_es[:min(5, N_active), :min(5, N_active)]}\n")
    # print(f"活跃ES间传输速率示例 (前5x5) [bit/s]:\n{r_es_active[:min(5, N_active), :min(5, N_active)]}")
    #
    # 打印平均传输速率
    if N_active > 1:
        avg_rate = np.sum(r_es_active) / (N_active * (N_active - 1))  # 排除对角线元素
        # print(f"活跃ES之间的平均传输速率: {avg_rate/1e6:.2f} Mbps")
        #
    # 对于演示，显示完整ES矩阵的示例
    # print(f"完整ES传输速率矩阵示例 (前5x5) [bit/s]:\n{r_es[:min(5, N), :min(5, N)]}")
    
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
            
            #=======================STEP 7: 计算活跃ES到云的距离========================#
            print("\n========== 计算活跃边缘服务器到云服务器的距离 ==========")
            
            # 计算每个活跃边缘服务器到云的距离
            active_es_to_cloud_distance = np.zeros((N_active, 1))
            # print("\n活跃边缘服务器到云服务器的距离:")
            #
            for i, idx in enumerate(active_es_indices):
                es_node = es_nodes[idx]
                if es_node in pos:
                    es_pos = pos[es_node]
                    # 计算地理距离(千米)并转换为米
                    distance_km = calculate_distance(es_pos[1], es_pos[0], cloud_lat, cloud_lon)
                    active_es_to_cloud_distance[i, 0] = distance_km * 1000
                    # print(f"  活跃ES{i}(原ES{idx}) → Cloud: {distance_km:.3f} km ({distance_km*1000:.1f} m)")
                else:
                    print(f"警告: 活跃边缘服务器 {es_node} 缺少位置信息，设置距离为无穷大")
                    active_es_to_cloud_distance[i, 0] = float('inf')
            print(f"min dist {np.min(active_es_to_cloud_distance, axis=1)}")
            # print(f"活跃ES到云距离矩阵形状: {active_es_to_cloud_distance.shape} (N_active×1)")
        else:
            print("警告: 活跃边缘服务器缺少位置信息，无法计算云服务器位置")
            active_es_to_cloud_distance = None
    else:
        print("警告: 没有活跃边缘服务器，无法计算云服务器位置")
        active_es_to_cloud_distance = None

    #=======================STEP 8: 计算活跃ES到云的传输速率========================#
    print("\n========== 计算活跃边缘服务器到云服务器(Active ES-Cloud)的传输速率 ==========")
    
    # 计算活跃ES到云的传输速率
    r_es_to_cloud = None
    if active_es_to_cloud_distance is not None:
        # 定义ES-to-Cloud通信的专属参数
        cloud_center_bandwidth = 1.2e8        # 云连接通常带宽更高 (120 MHz)
        cloud_bandwidth_sigma = 0.25          # 带宽对数正态分布的标准差
        cloud_bandwidth_range = (5e7, 3e8)    # 带宽范围 50-300 MHz
        
        # 使用活跃ES到云的距离矩阵计算传输速率
        r_es_to_cloud, g_es_to_cloud, B_es_to_cloud = calculate_transmission_rates(
            distance_matrix=active_es_to_cloud_distance,
            transmit_power=p_es,              # 使用与ES-ES相同的发射功率
            center_bandwidth=cloud_center_bandwidth,
            bandwidth_sigma=cloud_bandwidth_sigma,
            noise_density=N0,
            path_loss_exponent=path_loss_exponent,
            g0_at_1m=g0_es_es,
            bandwidth_range=cloud_bandwidth_range,
            node_indices=active_es_indices    # 使用活跃ES的索引
        )
        
        print(f"活跃ES到云传输速率矩阵形状: {r_es_to_cloud.shape} (N_active×1)")
        
        # 打印活跃ES到云传输速率
        print("\n活跃边缘服务器到云服务器的传输速率:")
        display_count = min(10, N_active)
        for i in range(display_count):
            original_idx = active_es_indices[i]
            rate_mbps = r_es_to_cloud[i, 0] / 1e6  # 转换为Mbps
            # print(f"  活跃ES{i}(原ES{original_idx}) → Cloud: {rate_mbps:.2f} Mbps")

        # if N_active > 10:
        #     print(f"  ... 以及其他 {N_active - 10} 个活跃边缘服务器")

        # 计算平均活跃ES-Cloud传输速率
        print(f"活跃ES到云的平均传输速率: {np.mean(r_es_to_cloud)/1e6:.2f} Mbps")
    else:
        print("警告: 由于缺少距离信息，无法计算活跃ES到云的传输速率")
        r_es_to_cloud = None

    #=======================STEP 9: 计算客户端到云的直接传输速率========================#
    print("\n========== 计算客户端到云服务器(Client-Cloud)的直接传输速率 ==========")
    
    # 计算客户端到云的传输速率
    r_client_to_cloud = None
    if cloud_pos is not None:
        # 定义客户端到云通信的专属参数
        cloud_direct_bandwidth = 8e7          # 直连云带宽较低 (80 MHz)
        cloud_direct_sigma = 0.3              # 带宽对数正态分布的标准差
        cloud_direct_range = (5e7, 1e8)       # 带宽范围 50-100 MHz

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
        print(f"min dist {np.min(client_to_cloud_distance, axis=1)}")
        # 使用通用函数计算客户端到云的传输速率
        r_client_to_cloud, g_client_to_cloud, B_client_to_cloud = calculate_transmission_rates(
            distance_matrix=client_to_cloud_distance,
            transmit_power=p_client,               # 使用客户端发射功率
            center_bandwidth=cloud_direct_bandwidth,
            bandwidth_sigma=cloud_direct_sigma,
            noise_density=N0,
            path_loss_exponent=path_loss_exponent,
            g0_at_1m=g0_client_es,
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
        # sample_size = min(5, M)
        # print(f"\n客户端到云的传输速率示例 (前{sample_size}个):")
        # for i in range(sample_size):
        #     rate_mbps = r_client_to_cloud[i, 0] / 1e6
        #     print(f"  客户端{i} → 云: {rate_mbps:.2f} Mbps")
    else:
        print("警告: 由于缺少云服务器位置信息，无法计算客户端到云的直接传输速率")

    # 返回精简的计算结果
    return (client_nodes,          # 客户端节点列表
            active_es_nodes,       # 活跃边缘服务器节点列表
            original_association_matrix,
            association_matrix,    # 客户端到活跃ES的关联矩阵
            r_client_to_active_es, # 客户端到活跃ES的传输速率矩阵
            r_es_active,           # 活跃ES到活跃ES的传输速率矩阵
            r_es_to_cloud,         # ES到云的传输速率矩阵
            r_client_to_cloud,     # 客户端到云的直接传输速率矩阵
            cloud_pos)

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
    # print(f"Manual check: Load std (balance) = {np.std(loads):.2f} (lower is better)")
    # print("Optimization Version: Hungarian + Simulated Annealing")

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
    bipartite_graph, client_nodes, es_nodes, distance_matrix, pos, filter_info = build_bipartite_graph(
        graphml_file, es_ratio, visualize)
    if bipartite_graph is None:
        return None, [], [], None, None, None, None, None
    print("===============构建二部图==================")
    print("Bipartite Graph Built Successfully!")
    print(f"Total Client Nodes: {len(client_nodes)}")
    print(f"Total ES Nodes: {len(es_nodes)}")
    print(f"Distance Matrix Shape: {distance_matrix.shape}")
    # print(f"Distance Matrix (first 5x5):\n{distance_matrix[:5, :min(5, len(es_nodes))]}")

    # 分配带宽并获取传输速率
    client_nodes, active_es_nodes, original_association_matrix, association_matrix, r_client_to_active_es, r_es, r_es_to_cloud, r_client_to_cloud, cloud_pos = establish_communication_channels(
        client_nodes, es_nodes, distance_matrix, pos, max_capacity)
    
    if r_client_to_active_es is not None:
        print(f"Final transmission rates r_m,n (bit/s, first 5x5):\n{r_client_to_active_es[:min(5, len(client_nodes)), :min(5, len(active_es_nodes))]}")
    
    # # 使用与establish_communication_channels中相同的max_capacity值进行验证
    # validate_results(B_cloud=5e7, B_n=B_n, r=r, assignments=assignments, loads=loads, max_capacity=max_capacity)
    visualize_node_distribution(client_nodes, es_nodes, pos, es_ratio, cloud_pos=cloud_pos, association_matrix=original_association_matrix, filter_info=filter_info)
    # 返回精简的结果集
    return bipartite_graph, client_nodes, active_es_nodes, association_matrix, r_client_to_active_es, r_es, r_es_to_cloud, r_client_to_cloud


if __name__ == '__main__':
    args = args_parser()
    result = build_bipartite_graph(graphml_file='graph-example/Ulaknet.graphml', es_ratio=args.es_ratio)
    if result:
        bipartite_graph, client_nodes, es_nodes, distance_matrix, pos, filter_info = result
        print(f"Built bipartite graph with {len(client_nodes)} clients and {len(es_nodes)} edge servers")
        if filter_info:
            print(f"Applied geographic filter: center {filter_info['center']}, radius {filter_info['radius']} km")
