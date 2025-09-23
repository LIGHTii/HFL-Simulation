# models/cluster1.py
# 相似度矩阵 拉普拉斯 特征值取小
import numpy as np
from sklearn.cluster import KMeans

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
        epsilon = 0.65 * global_variance
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

    return best_clusters, best_labels


def cluster1(data, epsilon=None):
    """
    谱聚类入口函数
    自动选择最优簇数
    """
    cluster_num, labels = find_optimal_clusters_binary_search(data, epsilon=epsilon)
    return cluster_num, labels
