# models/cluster2.py
# 相似度矩阵 拉普拉斯 特征值取小
import numpy as np
from sklearn.cluster import KMeans

def distance(x1, x2):
    """计算两个样本点之间的欧式距离"""
    return np.sqrt(np.sum((x1 - x2) ** 2))


def getEigen(L, cluster_num):
    """获得拉普拉斯矩阵的特征矩阵，确保返回实数"""
    eigval, eigvec = np.linalg.eig(L)

    # 按特征值的实部排序
    idx = np.argsort(eigval.real)

    # 选择前cluster_num个最小的特征值对应的特征向量
    selected_idx = idx[:cluster_num]

    # 只取特征向量的实部
    eigvec_real = eigvec[:, selected_idx].real

    return eigvec_real

def calculateGraphData(data, sigma):
    """计算图的相似度矩阵和拉普拉斯矩阵"""
    """获得对称的邻接矩阵 W （get_W）"""
    n = len(data)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                W[i][j] = np.exp(-distance(data[i], data[j]) ** 2 / (2 * sigma ** 2))

    # 计算度矩阵
    D = np.diag(np.sum(W, axis=1))

    # 计算归一化拉普拉斯矩阵 L_sym = D^{-1/2} L D^{-1/2}
    D_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
    L_sym = np.dot(np.dot(D_sqrt, (D - W)), D_sqrt)

    return W, L_sym

def calculate_intra_cluster_distance(data, labels, cluster_num):
    """计算簇内距离和"""
    total_distance = 0
    for i in range(cluster_num):
        cluster_points = data[labels == i]
        if len(cluster_points) > 0:
            centroid = np.mean(cluster_points, axis=0)
            total_distance += np.sum((cluster_points - centroid) ** 2)
    return total_distance

def spectralPartitionGraph(L_sym, cluster_num):
    """使用谱聚类进行图分割"""
    # 获取特征向量
    eigvec = getEigen(L_sym, cluster_num)

    # 标准化特征向量
    norms = np.linalg.norm(eigvec, axis=1, keepdims=True)
    norms[norms == 0] = 1  # 避免除以零
    eigvec_normalized = eigvec / norms

    # 使用KMeans聚类
    kmeans = KMeans(n_clusters=cluster_num, n_init=10, random_state=42)
    labels = kmeans.fit_predict(eigvec_normalized)

    return labels

def find_optimal_clusters_binary_search(data, epsilon=None, max_clusters=None):
    """
    使用二分搜索找到满足簇内距离和小于 epsilon 的最小簇数
    """
    n = len(data)
    if max_clusters is None:
        max_clusters = n

    # 预先计算相似度矩阵和拉普拉斯矩阵
    sigma = 0.4
    W, L_sym = calculateGraphData(data, sigma)

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
        labels = spectralPartitionGraph(W, mid_clusters)
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
        best_labels = spectralPartitionGraph(W, best_clusters)
        print(f"未找到满足条件的簇数，使用最大簇数: {best_clusters}")

    return best_clusters, best_labels


def cluster2(data, epsilon=None):
    """
    谱聚类入口函数
    自动选择最优簇数
    """
    cluster_num, labels = find_optimal_clusters_binary_search(data, epsilon=epsilon)
    return cluster_num, labels
