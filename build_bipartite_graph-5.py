import random
import math
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def calculate_distance(lat1, lon1, lat2, lon2):
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    R = 6371.0  # 地球半径 (km)
    return R * c


def build_bipartite_graph(graphml_file):
    original_graph = nx.read_graphml(graphml_file)
    nodes_to_remove = [node for node, data in original_graph.nodes(data=True) if
                       'Latitude' not in data or 'Longitude' not in data]
    if nodes_to_remove:
        original_graph.remove_nodes_from(nodes_to_remove)
        print(f"移除了 {len(nodes_to_remove)} 个缺少经纬度数据的节点: {nodes_to_remove}")
    node_ids = list(original_graph.nodes)
    if not node_ids:
        print("图中已无有效节点，程序终止。")
        return None, [], [], None
    print("节点数据示例:", list(original_graph.nodes(data=True))[0][1])
    num_es = max(1, int(len(node_ids) * 0.25))
    es_nodes = random.sample(node_ids, num_es)
    client_nodes = [node for node in node_ids if node not in es_nodes]
    bipartite_graph = nx.Graph()
    for node in client_nodes:
        bipartite_graph.add_node(node, bipartite=0, **original_graph.nodes[node])
    for node in es_nodes:
        bipartite_graph.add_node(node, bipartite=1, **original_graph.nodes[node])
    distance_matrix = np.zeros((len(client_nodes), len(es_nodes)))
    for i, client in enumerate(client_nodes):
        for j, es in enumerate(es_nodes):
            client_lat = original_graph.nodes[client]['Latitude']
            client_lon = original_graph.nodes[client]['Longitude']
            es_lat = original_graph.nodes[es]['Latitude']
            es_lon = original_graph.nodes[es]['Longitude']
            weight = calculate_distance(client_lat, client_lon, es_lat, es_lon)
            distance_matrix[i, j] = weight
            bipartite_graph.add_edge(client, es, weight=weight)
    return bipartite_graph, client_nodes, es_nodes, distance_matrix


def plot_graph(bipartite_graph, client_nodes, es_nodes):
    pos = {node: (data['Longitude'], data['Latitude']) for node, data in bipartite_graph.nodes(data=True)}
    plt.figure(figsize=(12, 10))
    nx.draw_networkx_nodes(bipartite_graph, pos, nodelist=client_nodes, node_color='#a6c0e5', node_size=200,
                           label="Client Nodes")
    nx.draw_networkx_nodes(bipartite_graph, pos, nodelist=es_nodes, node_color='#ef8b67', node_size=250,
                           label="ES Nodes", node_shape='s')
    nx.draw_networkx_edges(bipartite_graph, pos, width=1.0, alpha=0.5, edge_color='#6f6f6f')
    plt.title('Bipartite Graph: Client Nodes and ES Nodes', fontsize=15)
    plt.legend(scatterpoints=1)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.grid(True)
    plt.show()


def plot_assigned_graph(bipartite_graph, client_nodes, es_nodes, assignments):
    """
    绘制设备与对应挂载边缘服务器的图
    """
    pos = {node: (data['Longitude'], data['Latitude']) for node, data in bipartite_graph.nodes(data=True)}
    plt.figure(figsize=(12, 10))
    # 绘制节点
    nx.draw_networkx_nodes(bipartite_graph, pos, nodelist=client_nodes, node_color='#a6c0e5', node_size=200,
                           label="Client Nodes")
    nx.draw_networkx_nodes(bipartite_graph, pos, nodelist=es_nodes, node_color='#ef8b67', node_size=250,
                           label="ES Nodes", node_shape='s')
    # 只绘制挂载的边（绿色粗线）
    assigned_edges = [(client_nodes[m], es_nodes[assignments[m]]) for m in range(len(assignments)) if
                      assignments[m] != -1]
    nx.draw_networkx_edges(bipartite_graph, pos, edgelist=assigned_edges, width=2.0, alpha=0.8, edge_color='green')
    plt.title('Assigned Graph: Devices to Assigned ES Nodes', fontsize=15)
    plt.legend(scatterpoints=1)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.grid(True)
    plt.show()


def allocate_bandwidth_eba(client_nodes, es_nodes, distance_matrix):
    """
    基于容量约束的设备挂载和 EBA 带宽分配 (无传输延迟)
    """
    M = len(client_nodes)  # 设备数
    N = len(es_nodes)  # 边缘服务器数
    B_cloud = 1e7  # 云总带宽 (Hz)
    max_capacity = 5  # 每个边缘最大挂载设备数

    # 初始化：所有边缘平均分配带宽
    B_n_init = np.ones(N) * (B_cloud / N)
    print(f"初始化带宽分配 (B_n_init): {B_n_init}")

    p_m = np.ones(M) * 0.1  # 发送功率 (W)，固定为0.1 W
    N0 = 1e-20  # 噪声功率谱密度 (W/Hz)
    alpha = 3.5  # 路径损耗指数
    k = 1e-3  # 信道增益归一化常数

    # 修正距离单位：假设 km 转换为 m
    distance_matrix = distance_matrix / 1000  # km to m

    # 计算信道增益 (基于距离)
    g = k / (distance_matrix ** alpha + 1e-10)  # 避免除零

    # 计算基准传输速率 r_m,n (theta=1，用于排序挂载)
    SNR_base = (p_m[:, None] * g) / (B_n_init * N0)
    r_base = B_n_init * np.log2(1 + SNR_base)

    # 对设备按 max_r 降序排序 (优先高 r 设备)
    max_r_per_device = np.max(r_base, axis=1)
    sorted_device_indices = np.argsort(max_r_per_device)[::-1]  # 降序
    print(
        f"设备排序 (按 max_r 降序): {sorted_device_indices[:5]}... (max_r: {max_r_per_device[sorted_device_indices[:5]]})")

    # 初始化挂载: assignments = -1 (未挂载), loads = 0
    assignments = np.full(M, -1, dtype=int)
    loads = np.zeros(N, dtype=int)

    # 逐个挂载排序设备
    for idx in sorted_device_indices:
        available_n = np.where(loads < max_capacity)[0]
        if len(available_n) == 0:
            print(f"警告: 设备 {idx} 无可用边缘 (所有满载)")
            continue
        r_for_device = r_base[idx, available_n]
        sorted_n = available_n[np.argsort(r_for_device)[::-1]]  # 按 r 降序排序
        for n in sorted_n:
            if loads[n] < max_capacity:
                assignments[idx] = n
                loads[n] += 1
                break

    print(f"挂载结果 (assignments): {assignments}")
    print(f"每个边缘负载 (loads): {loads}")

    # 构建关联集 A_n
    A_n = [np.where(assignments == n)[0] for n in range(N)]

    # 识别活跃边缘并关闭无挂载边缘，重新分配带宽
    active = [n for n in range(N) if len(A_n[n]) > 0]
    N_active = len(active)
    B_n = np.zeros(N)
    if N_active > 0:
        B_n[active] = B_cloud / N_active
    print(f"活跃边缘: {active}, 重新分配带宽 (B_n): {B_n}")

    # EBA 分配
    theta = np.zeros((M, N))
    for n in active:
        if len(A_n[n]) > 0:
            theta[A_n[n], n] = 1 / len(A_n[n])

    # 计算实际传输速率 r_m,n(t)
    SNR = (p_m[:, None] * g) / (theta * B_n + 1e-10)  # 避免除零
    r = theta * B_n * np.log2(1 + SNR / (theta * B_n * N0 + 1e-10))

    # 处理 r 异常
    r_threshold = 1e3  # 阈值，低于此值视为异常
    abnormal_count = 0
    for m in range(M):
        if assignments[m] != -1:
            n = assignments[m]
            if r[m, n] < r_threshold:
                print(f"警告: 设备 {m} 到边缘 {n} 的 r = {r[m, n]} < {r_threshold} bit/s (异常低速率)")
                r[m, n] = 0  # 设置为0避免影响
                abnormal_count += 1
    print(f"总异常 r 值数量: {abnormal_count}")

    return r, assignments, loads, B_n, theta


def validate_results(B_cloud, B_n, theta, r, assignments, loads, max_capacity):
    """
    检验代码结果的正确性 (包括容量约束)
    """
    N = len(B_n)
    M = len(theta)
    active = np.unique(assignments[assignments != -1])

    # 检查带宽总和
    if abs(np.sum(B_n) - B_cloud) > 1e-6:
        print("错误: 重新分配后 B_n 总和 != B_cloud")
    else:
        print("通过: B_n 总和正确")

    # 检查无挂载 B_n=0
    for n in range(N):
        if n not in active and B_n[n] != 0:
            print(f"错误: 无挂载边缘 {n} B_n != 0")
        elif n in active and B_n[n] == 0:
            print(f"错误: 活跃边缘 {n} B_n == 0")

    # 检查容量约束
    if np.any(loads > max_capacity):
        print(f"错误: 负载超过容量 {max_capacity} (loads: {loads})")
    else:
        print("通过: 容量约束满足")

    # 检查挂载覆盖率
    mounted_ratio = np.sum(assignments != -1) / M
    if mounted_ratio < 0.95:
        print(f"警告: 挂载覆盖率低 ({mounted_ratio:.2%})")
    else:
        print(f"通过: 挂载覆盖率 {mounted_ratio:.2%}")

    # 检查 r 仅在关联处非零
    for m in range(M):
        if assignments[m] != -1:
            n = assignments[m]
            if r[m, n] <= 0:
                print(f"错误: 关联设备 {m} 到 {n} 的 r <= 0")
        for other_n in range(N):
            if other_n != assignments[m] and (r[m, other_n] > 0):
                print(f"错误: 非关联 {m} 到 {other_n} 的 r > 0")

    # 检查排序有效性 (前半设备 max_r 高于后半)
    if M > 0:
        max_r_values = np.max(r, axis=1)
        sorted_max_r = np.sort(max_r_values)[::-1]
        half = M // 2
        if np.all(sorted_max_r[:half] >= sorted_max_r[-half:]):
            print("通过: 挂载优先高 r 设备")
        else:
            print("警告: 排序挂载可能未优化")

    print("手动检查: loads std (均衡度) = {:.2f} (越低越好)".format(np.std(loads)))


def main():
    graphml_file = "graph-example/Ulaknet.graphml"
    bipartite_graph, client_nodes, es_nodes, distance_matrix = build_bipartite_graph(graphml_file)
    if bipartite_graph is None:
        return
    print("Bipartite Graph Built Successfully!")
    print(f"Total Client Nodes: {len(client_nodes)}")
    print(f"Total ES Nodes: {len(es_nodes)}")
    print(f"Distance Matrix Shape: {distance_matrix.shape}")
    print(f"Distance Matrix (first 5x5):\n{distance_matrix[:5, :5]}")

    # 容量约束挂载和 EBA 分配
    r, assignments, loads, B_n, theta = allocate_bandwidth_eba(client_nodes, es_nodes, distance_matrix)
    print(f"传输速率 r_m,n (bit/s，第一5x5):\n{r[:5, :5]}")  # 仅显示前 5x5

    # 检验
    max_capacity = 5
    validate_results(B_cloud=1e7, B_n=B_n, theta=theta, r=r, assignments=assignments, loads=loads,
                     max_capacity=max_capacity)

    # 生成设备与挂载边缘服务器的图
    plot_assigned_graph(bipartite_graph, client_nodes, es_nodes, assignments)

    plot_graph(bipartite_graph, client_nodes, es_nodes)
    nx.write_graphml(bipartite_graph, "bipartite_graph.graphml")
    print("\nBipartite graph saved to bipartite_graph.graphml")


if __name__ == "__main__":
    main()