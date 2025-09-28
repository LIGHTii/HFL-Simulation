# models/cluster.py
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import torch
import copy
from models.Update import LocalUpdate  # 导入LocalUpdate

def model_to_vector(model_params):
    """将模型参数字典转换为向量"""
    vectors = []
    for param in model_params.values():
        # 将参数转换为numpy数组并展平
        vectors.append(param.cpu().numpy().flatten())
    return np.concatenate(vectors)


def distance(x1, x2):
    """计算两个样本点之间的欧式距离"""
    return np.sqrt(np.sum((x1 - x2) ** 2))


def get_dist_matrix(data):
    """获取距离矩阵"""
    n = len(data)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist_matrix[i][j] = dist_matrix[j][i] = distance(data[i], data[j])
    return dist_matrix

def getW(data):
    """获得对称的权重矩阵 (这里直接用距离矩阵)"""
    return get_dist_matrix(data)

def getD(W):
    """
    计算图的度矩阵 D，W 是邻接矩阵
    """
    # 计算每一行的和，得到每个节点的度数
    degrees = np.sum(W, axis=1)

    # 创建度矩阵，度数作为对角线元素
    D = np.diag(degrees)

    return D


def normalize_W(W):
    """
    标准化邻接矩阵 W，使用公式 W_标准化 = D^(-1/2) W D^(-1/2)
    """
    # 计算度矩阵 D
    D = getD(W)

    # 计算 D^(-1/2)，即度矩阵的平方根的逆
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))

    # 计算 W_标准化
    W_normalized = D_inv_sqrt @ W @ D_inv_sqrt

    return W_normalized


def getEigen(W_normalized, cluster_num):
    """
    获得距离矩阵 W 的前 cluster_num 个最小特征值对应的特征向量
    """
    eigval, eigvec = np.linalg.eig(W_normalized)
    idx = np.argsort(eigval.real)  # 按实部 从小到大排序
    selected_idx = idx[:cluster_num]  # 改为取最小的 cluster_num 个特征值
    return eigvec[:, selected_idx].real


def spectralPartitionGraph(W_normalized, cluster_num):
    """
    使用距离矩阵 W 进行谱聚类
    """
    # 获取特征向量
    eigvec = getEigen(W_normalized, cluster_num)

    # 标准化特征向量
    norms = np.linalg.norm(eigvec, axis=1, keepdims=True)
    norms[norms == 0] = 1
    eigvec_normalized = eigvec / norms

    # KMeans 聚类
    kmeans = KMeans(n_clusters=cluster_num, n_init=10, random_state=42)
    labels = kmeans.fit_predict(eigvec_normalized)

    return labels


def calculate_intra_cluster_distance(data, labels, cluster_num):
    """计算簇内距离和，用来评估聚类效果"""
    total_distance = 0
    for i in range(cluster_num):
        cluster_points = data[labels == i]
        if len(cluster_points) > 0:
            centroid = np.mean(cluster_points, axis=0)
            total_distance += np.sum((cluster_points - centroid) ** 2)
    return total_distance


def find_optimal_clusters_binary_search(data, epsilon=None, max_clusters=None):
    """
    使用二分搜索找到满足簇内距离和小于 epsilon 的最小簇数
    """
    n = len(data)
    if max_clusters is None:
        max_clusters = n

    # 计算距离矩阵
    W = getW(data)
    W_normalized = normalize_W(W)

    # 全局方差作为参考
    global_centroid = np.mean(data, axis=0)
    global_variance = np.sum((data - global_centroid) ** 2)

    if epsilon is None:
        epsilon = 0.6 * global_variance
        print(f"使用自动计算的 epsilon 阈值: {epsilon:.4f}")

    min_clusters = 1
    best_clusters = max_clusters
    best_labels = None
    search_history = []

    while min_clusters <= max_clusters:
        mid_clusters = (min_clusters + max_clusters) // 2
        labels = spectralPartitionGraph(W_normalized, mid_clusters)
        intra_distance = calculate_intra_cluster_distance(data, labels, mid_clusters)

        search_history.append((mid_clusters, intra_distance))
        print(f"簇数: {mid_clusters}, 簇内距离和: {intra_distance:.4f}, 阈值: {epsilon:.4f}")

        if intra_distance <= epsilon:
            best_clusters, best_labels = mid_clusters, labels
            max_clusters = mid_clusters - 1
        else:
            min_clusters = mid_clusters + 1

    if best_labels is None:
        best_clusters = max_clusters
        best_labels = spectralPartitionGraph(W_normalized, best_clusters)
        print(f"未找到满足条件的簇数，使用最大簇数: {best_clusters}")

    if search_history:
        clusters, distances = zip(*search_history)
        plt.figure(figsize=(10, 6))
        plt.plot(clusters, distances, 'bo-', label='Intra-cluster Distance Sum')
        plt.axhline(y=epsilon, color='r', linestyle='--', label='Threshold')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Intra-cluster Distance Sum')
        plt.title('Binary Search Process')
        plt.legend()
        plt.grid(True)
        plt.show()

    return best_clusters, best_labels


def cluster(data, epsilon=None):
    """
    谱聚类入口函数
    自动选择最优簇数
    """
    cluster_num, labels = find_optimal_clusters_binary_search(data, epsilon=epsilon)
    return cluster_num, labels


# ==================================== ES聚类 ========================================
def calculate_es_label_distributions(A, client_label_distributions):
    """
    计算每个ES的标签分布，通过汇总连接到该ES的所有客户端的标签分布

    参数:
        A: 客户端-ES关联矩阵，形状为(n_clients, n_es)
        client_label_distributions: 每个客户端的标签分布，形状为(n_clients, 10)

    返回:
        es_label_distributions: 每个ES的标签分布，形状为(n_es, 10)
    """
    n_es = A.shape[1]
    es_label_distributions = np.zeros((n_es, 10))

    for es_idx in range(n_es):
        # 找到连接到当前ES的所有客户端
        client_indices = np.where(A[:, es_idx] == 1)[0]

        if len(client_indices) > 0:
            # 汇总这些客户端的标签分布
            es_label_distributions[es_idx] = np.sum(client_label_distributions[client_indices], axis=0)

    return es_label_distributions

'''
def visualize_es_clustering_result(es_label_distributions, cluster_labels,
                                   save_path='./save/es_clustering_result.png'):
    """
    可视化ES聚类结果，显示每个ES的标签分布并按聚类结果分组

    参数:
        es_label_distributions: 每个ES的标签分布，形状为(n_es, 10)的数组
        cluster_labels: ES聚类标签列表
        save_path: 保存路径
    """
    # 获取唯一的聚类标签和数量
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)

    # 创建图形
    fig, ax = plt.subplots(figsize=(16, 8))

    # 按聚类结果排序
    sorted_indices = np.argsort(cluster_labels)
    sorted_labels = cluster_labels[sorted_indices]
    sorted_distributions = es_label_distributions[sorted_indices]

    # 计算每个聚类的起始位置
    cluster_boundaries = []
    start_idx = 0
    for cluster_id in unique_clusters:
        cluster_size = np.sum(sorted_labels == cluster_id)
        cluster_boundaries.append((start_idx, start_idx + cluster_size - 1))
        start_idx += cluster_size

    # 创建堆叠柱状图
    x = np.arange(len(sorted_distributions))
    bottom = np.zeros(len(sorted_distributions))

    # 定义单色渐变颜色（蓝色系，10个标签）
    cmap = plt.cm.Blues
    label_colors = cmap(np.linspace(0.3, 1, 10))  # 从浅到深取 10 种颜色

    for i in range(10):  # 0-9 共10个标签
        values = sorted_distributions[:, i]
        ax.bar(x, values, bottom=bottom, color=label_colors[i], label=f'Label {i}', alpha=0.9)
        bottom += values

    # 添加聚类分界线
    for boundary in cluster_boundaries:
        if boundary[1] < len(x) - 1:  # 不是最后一个ES
            ax.axvline(x=boundary[1] + 0.5, color='black', linestyle='--', linewidth=2)

    # 设置x轴为聚类ID
    cluster_centers = [(boundary[0] + boundary[1]) / 2 for boundary in cluster_boundaries]
    ax.set_xticks(cluster_centers)
    ax.set_xticklabels([f'Cluster {i}' for i in unique_clusters])

    ax.set_title('Label Distribution Across Edge Servers (Grouped by Cluster)')
    ax.set_xlabel('Edge Server Cluster')
    ax.set_ylabel('Number of Samples')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[谱聚类] ES聚类结果已保存到: {save_path}")
'''


def visualize_clustering_comparison(es_label_distributions, cluster_labels,
                                    save_path='./save/clustering_comparison.png'):
    """
    对比谱聚类分簇和随机分簇的效果，使用完全渐变的柱状图

    参数:
        es_label_distributions: 每个ES的标签分布，形状为(n_es, 10)的数组
        cluster_labels: ES聚类标签列表
        save_path: 保存路径
    """
    n_es = len(es_label_distributions)
    n_clusters = len(np.unique(cluster_labels))

    # 随机分簇，每个 ES 随机分到 0~n_clusters-1 的簇
    np.random.seed()  # 可去掉固定随机种子，每次生成不同随机结果
    random_cluster_labels = np.random.randint(0, n_clusters, size=n_es)

    # 创建图形，包含两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('Comparison of Spectral Clustering vs Random Clustering', fontsize=16)

    # 使用jet颜色映射
    cmap = plt.cm.viridis

    # 绘制谱聚类结果（左侧子图）
    _plot_continuous_gradient_clustering_result(
        ax1, es_label_distributions, cluster_labels,
        "Spectral Clustering Result", cmap
    )

    # 绘制随机分簇结果（右侧子图）
    _plot_continuous_gradient_clustering_result(
        ax2, es_label_distributions, random_cluster_labels,
        "Random Clustering Result", cmap
    )

    # 添加传统图例（水平颜色条和标签）
    legend_elements = []
    for i in range(10):
        color = cmap(i / 9.0)
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, fc=color, label=f'Label {i}'))

    fig.legend(handles=legend_elements,
               loc='lower center',
               bbox_to_anchor=(0.5, 0.02),
               ncol=10,
               frameon=False)

    # 调整布局并保存
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # 为图例留出空间
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"聚类对比可视化已保存到: {save_path}")


def _plot_continuous_gradient_clustering_result(ax, distributions, labels, title, cmap):
    """
    辅助函数：在指定的轴上绘制完全渐变的柱状图

    参数:
        ax: matplotlib轴对象
        distributions: 标签分布数据
        labels: 聚类标签
        title: 子图标题
        cmap: 颜色映射
    """
    # 获取唯一的聚类标签和数量
    unique_clusters = np.unique(labels)

    # 按聚类结果排序
    sorted_indices = np.argsort(labels)
    sorted_labels = labels[sorted_indices]
    sorted_distributions = distributions[sorted_indices]

    # 计算每个聚类的起始位置
    cluster_boundaries = []
    start_idx = 0
    for cluster_id in unique_clusters:
        cluster_size = np.sum(sorted_labels == cluster_id)
        cluster_boundaries.append((start_idx, start_idx + cluster_size - 1))
        start_idx += cluster_size

    # 创建完全渐变的柱状图
    x = []
    gap = 1  # 簇之间的额外间距，可以自己调大
    current_x = 0
    for cluster_id in unique_clusters:
        indices = np.where(sorted_labels == cluster_id)[0]
        for _ in indices:
            x.append(current_x)
            current_x += 1  # 同簇内部柱子间距保持1
        current_x += gap  # 簇之间增加间距
    x = np.array(x)

    # 计算每个ES的总样本数
    total_samples = np.sum(sorted_distributions, axis=1)

    # 为每个ES创建渐变颜色
    for i, (x_pos, es_dist) in enumerate(zip(x, sorted_distributions)):
        # 计算每个标签在ES中的累积比例
        cum_proportions = np.cumsum(es_dist) / total_samples[i]

        # 为每个ES创建渐变颜色条
        for j in range(100):  # 将每个柱状图分成100个小段
            start_frac = j / 100.0
            end_frac = (j + 1) / 100.0

            # 找到这个分段对应的标签
            label_idx = np.searchsorted(cum_proportions, start_frac)
            if label_idx >= len(cum_proportions):
                label_idx = len(cum_proportions) - 1

            # 计算颜色
            color = cmap(label_idx / 9.0)

            # 计算这个分段的高度
            height = total_samples[i] / 100.0

            # 绘制这个分段
            ax.bar(x_pos, height, bottom=start_frac * total_samples[i],
                   color=color, width=0.8, alpha=0.9, edgecolor=None)

    # 设置x轴为聚类ID，改成根据新的 x 坐标计算簇中心
    cluster_centers = []
    for boundary in cluster_boundaries:
        left_idx = boundary[0]
        right_idx = boundary[1]
        center = (x[left_idx] + x[right_idx]) / 2  # 使用新的 x 坐标
        cluster_centers.append(center)

    ax.set_xticks(cluster_centers)
    ax.set_xticklabels([f'Cluster {i}' for i in unique_clusters])

    ax.set_title(title)
    ax.set_xlabel('Edge Server Cluster')
    ax.set_ylabel('Number of Samples')
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')

def FedAvg_weighted(models, sizes=None):
    """
    支持加权的联邦平均算法

    参数:
        models: 模型参数列表
        sizes: 每个模型对应的数据量，如果为None则使用简单平均

    返回:
        w_avg: 平均后的模型参数
    """
    if sizes is None:
        sizes = [1] * len(models)

    total_size = sum(sizes)
    w_avg = copy.deepcopy(models[0])

    for k in w_avg.keys():
        w_avg[k] = w_avg[k] * sizes[0]
        for i in range(1, len(models)):
            w_avg[k] += models[i][k] * sizes[i]
        w_avg[k] = torch.div(w_avg[k], total_size)

    return w_avg


def train_initial_models(args, dataset_train, dict_users, net_glob, num_users):
    """训练初始本地模型，用于构建ES相似度图 """
    w_locals = []
    client_label_distributions = []  # 存储每个客户端的标签分布

    print("Training initial local models for graph construction...")
    print(f"使用谱聚类学习率: {args.lr_init}")

    # 创建一个临时args对象，使用不同的学习率
    temp_args = copy.deepcopy(args)
    temp_args.lr = args.lr_init  # 使用谱聚类专用学习率
    
    for user_idx in range(num_users):
        # 创建本地更新实例，使用谱聚类专用学习率
        local = LocalUpdate(
            args=temp_args,  # 使用修改后的参数
            dataset=dataset_train,
            idxs=dict_users[user_idx]
        )

        # 复制全局模型作为本地模型的初始状态
        local_net = copy.deepcopy(net_glob)

        # 训练本地模型
        w_local, loss_local = local.train(net=local_net.to(args.device))

        w_locals.append(copy.deepcopy(w_local))

        # 计算该客户端的标签分布
        labels = [dataset_train[i][1] for i in dict_users[user_idx]]
        label_count = np.zeros(10)  # 10个类别
        for label in labels:
            label_count[label] += 1
        client_label_distributions.append(label_count)

    return w_locals, np.array(client_label_distributions)

def aggregate_es_models(w_locals, A, dict_users, net_glob):
    """
    根据客户端-ES关联矩阵A，聚合每个ES的模型

    参数:
        w_locals: 所有客户端的模型参数列表
        A: 客户端-ES关联矩阵
        dict_users: 客户端数据索引字典
        net_glob: 全局模型（用于初始化空ES）

    返回:
        es_models: 每个ES的聚合模型参数列表
    """
    num_ESs = A.shape[1]
    es_models = [None] * num_ESs

    for es_idx in range(num_ESs):
        # 找到连接到当前ES的所有客户端
        client_indices = np.where(A[:, es_idx] == 1)[0]

        if len(client_indices) > 0:
            # 获取这些客户端的模型
            client_models = [w_locals[i] for i in client_indices]

            # 计算每个客户端的数据量（用于加权平均）
            client_sizes = [len(dict_users[i]) for i in client_indices]

            # 使用加权平均进行聚合
            w_es = FedAvg_weighted(client_models, client_sizes)
            es_models[es_idx] = w_es
        else:
            # 如果ES没有连接任何客户端，设为None
            es_models[es_idx] = None #copy.deepcopy(net_glob.state_dict())

    return es_models


def spectral_clustering_es(es_models, epsilon=None):
    """
    对边缘服务器模型进行谱聚类

    参数:
        es_models: 边缘服务器模型列表
        epsilon: 簇内距离阈值

    返回:
        B: ES-EH关联矩阵
        cluster_labels: 聚类标签
    """
    print("[谱聚类] 开始对ES模型进行谱聚类...")

    # 将模型参数转换为向量
    model_vectors = []
    for i, model in enumerate(es_models):
        vec = model_to_vector(model)
        model_vectors.append(vec)
        print(f"[谱聚类] ES {i} 模型向量维度: {vec.shape}")

    model_vectors = np.array(model_vectors)
    print(f"[谱聚类] 模型向量矩阵形状: {model_vectors.shape}")

    # 进行谱聚类
    cluster_num, cluster_labels = cluster(model_vectors, epsilon=epsilon)
    print(f"[谱聚类] 自动确定的最佳簇数: {cluster_num}")

    # 构建B矩阵
    num_ESs = len(es_models)
    B = np.zeros((num_ESs, cluster_num), dtype=int)
    for es_idx, label in enumerate(cluster_labels):
        B[es_idx, label] = 1

    # 打印聚类分配结果
    print("[谱聚类] ES聚类分配结果:")
    for es_idx, label in enumerate(cluster_labels):
        print(f"  ES {es_idx} -> EH {label}")

    return B, cluster_labels