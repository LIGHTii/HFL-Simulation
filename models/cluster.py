# models/cluster.py
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
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


def getW(data, sigma):
    """构建邻接矩阵 W，直接使用欧式距离"""
    n = len(data)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                W[i][j] = distance(data[i], data[j])
    return W

def calculateGraphData(data, sigma):
    """
    计算图的相似度矩阵 W 和归一化拉普拉斯矩阵 L_sym
    L_sym = D^{-1/2} (D - W) D^{-1/2}
    """
    W = getW(data, sigma)
    D = np.diag(np.sum(W, axis=1))
    D_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
    L_sym = D_sqrt @ (D - W) @ D_sqrt
    return W, L_sym


def getEigen(L, cluster_num):
    """
    计算拉普拉斯矩阵的特征向量
    取前 cluster_num 个最大特征值对应的特征向量
    """
    eigval, eigvec = np.linalg.eig(L)
    idx = np.argsort(eigval.real)  # 按实部 从小到大排序
    selected_idx = idx[-cluster_num:]  # 改为取最大的 cluster_num 个特征值
    return eigvec[:, selected_idx].real


def spectralPartitionGraph(L_sym, cluster_num):
    """使用谱聚类对图进行划分"""
    # 计算特征向量
    eigvec = getEigen(L_sym, cluster_num)
    # 标准化（避免长度差异影响）
    eigvec_normalized = eigvec / (np.linalg.norm(eigvec, axis=1, keepdims=True) + 1e-12)
    # 用 KMeans 在特征空间中聚类
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


def find_optimal_clusters_binary_search(data, sigma=0.4, epsilon=None):
    """
    使用二分搜索选择最优簇数
    目标：找到最小的簇数，使簇内距离和 <= epsilon
    """
    n = len(data)
    W, L_sym = calculateGraphData(data, sigma)

    # 全局方差（作为阈值的基准）
    global_centroid = np.mean(data, axis=0)
    global_variance = np.sum((data - global_centroid) ** 2)

    if epsilon is None:
        # 如果用户没指定，默认阈值 = 全局方差的 2%
        epsilon = 0.02 * global_variance
        print(f"[谱聚类] 自动阈值 ε = {epsilon:.4f}")

    min_clusters, max_clusters = 1, n
    best_clusters, best_labels = n, None

    # 二分搜索过程
    while min_clusters <= max_clusters:
        mid = (min_clusters + max_clusters) // 2
        labels = spectralPartitionGraph(L_sym, mid)
        intra_distance = calculate_intra_cluster_distance(data, labels, mid)
        print(f"[谱聚类] 尝试簇数: {mid}, 簇内距离和: {intra_distance:.4f}, 阈值: {epsilon:.4f}")

        if intra_distance <= epsilon:
            best_clusters, best_labels = mid, labels
            max_clusters = mid - 1  # 尝试更少的簇
        else:
            min_clusters = mid + 1  # 需要更多的簇

    # 如果没有满足条件的，兜底用最大簇数
    if best_labels is None:
        best_clusters = n
        best_labels = spectralPartitionGraph(L_sym, best_clusters)
        print(f"[谱聚类] 未找到满足条件的簇数，使用最大簇数: {best_clusters}")

    return best_clusters, best_labels


def cluster(data, sigma=0.4, epsilon=None, cluster_num=None):
    """
    谱聚类入口函数
    - 如果 cluster_num=None，则自动选择最优簇数
    - 否则使用指定的簇数
    """
    if cluster_num is None:
        return find_optimal_clusters_binary_search(data, sigma, epsilon)
    else:
        W, L_sym = calculateGraphData(data, sigma)
        labels = spectralPartitionGraph(L_sym, cluster_num)
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


def visualize_es_clustering_result(es_label_distributions, cluster_labels,
                                   save_path='./save/es_clustering_result.png'):
    """
    可视化ES聚类结果，显示每个ES的标签分布并按聚类结果分组

    参数:
        es_label_distributions: 每个ES的标签分布，形状为(n_es, 10)的数组
        cluster_labels: ES聚类标签列表
        save_path: 保存路径
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 设置使用英文字体，避免中文字体警告
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'axes.unicode_minus': True  # 正确显示负号
    })

    # 获取唯一的聚类标签和数量
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)

    # 创建图形
    fig, ax = plt.subplots(figsize=(16, 8))

    # 对ES按照聚类结果排序
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

    # 定义标签颜色
    label_colors = plt.cm.Set3(np.linspace(0, 1, 10))

    for i in range(10):  # 0-9共10个标签
        values = sorted_distributions[:, i]
        ax.bar(x, values, bottom=bottom, color=label_colors[i], label=f'Label {i}', alpha=0.8)
        bottom += values

    # 添加聚类分界线
    for boundary in cluster_boundaries:
        if boundary[1] < len(x) - 1:  # 不是最后一个ES
            ax.axvline(x=boundary[1] + 0.5, color='black', linestyle='--', linewidth=2)

    # 设置x轴标签为聚类ID
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
    """
    训练初始本地模型，用于构建ES相似度图
    """
    w_locals = []
    client_label_distributions = []  # 存储每个客户端的标签分布

    print("Training initial local models for graph construction...")

    for user_idx in range(num_users):
        # 创建本地更新实例
        local = LocalUpdate(
            args=args,
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
            # 如果ES没有连接任何客户端，使用全局模型初始化
            es_models[es_idx] = copy.deepcopy(net_glob.state_dict())

    return es_models


def spectral_clustering_es(es_models, num_EHs=None, sigma=0.4, epsilon=None):
    """
    对边缘服务器模型进行谱聚类

    参数:
        es_models: 边缘服务器模型列表
        num_EHs: 期望的聚类数量，如果为None则自动确定
        sigma: 高斯核参数
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
    if num_EHs is None:
        cluster_num, cluster_labels = cluster(model_vectors, sigma=sigma, epsilon=epsilon)
        print(f"[谱聚类] 自动确定的最佳簇数: {cluster_num}")
    else:
        cluster_num, cluster_labels = cluster(model_vectors, cluster_num=num_EHs, sigma=sigma, epsilon=epsilon)
        print(f"[谱聚类] 使用指定的簇数: {cluster_num}")


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