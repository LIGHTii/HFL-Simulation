import numpy as np
import matplotlib.pyplot as plt
import torch
import copy
import torch.multiprocessing as mp
from tqdm import tqdm
from models.Update import LocalUpdate  # 导入LocalUpdate
from models.cluster2 import cluster2
from models.Nets import MLP, CNNMnist, CNNCifar, LR, ResNet18, VGG11, MobileNetCifar, LeNet5
# ==================================== ES聚类 ========================================
def model_to_vector(model_params):
    """将模型参数字典转换为向量"""
    vectors = []
    for param in model_params.values():
        # 将参数转换为numpy数组并展平
        vectors.append(param.cpu().numpy().flatten())
    return np.concatenate(vectors)

def calculate_es_label_distributions(A, client_label_distributions):
    """
    计算每个ES的标签分布，通过汇总连接到该ES的所有客户端的标签分布

    参数:
        A: 客户端-ES关联矩阵，形状为(n_clients, n_es)
        client_label_distributions: 每个客户端的标签分布，形状为(n_clients, n_classes)

    返回:
        es_label_distributions: 每个ES的标签分布，形状为(n_es, n_classes)
    """
    n_es = A.shape[1]
    n_classes = client_label_distributions.shape[1]  # 动态获取类别数
    es_label_distributions = np.zeros((n_es, n_classes))

    for es_idx in range(n_es):
        # 找到连接到当前ES的所有客户端
        client_indices = np.where(A[:, es_idx] == 1)[0]

        if len(client_indices) > 0:
            # 汇总这些客户端的标签分布
            es_label_distributions[es_idx] = np.sum(
                client_label_distributions[client_indices], axis=0
            )

    return es_label_distributions

def visualize_clustering_comparison(es_label_distributions, cluster_labels,
                                    save_path='./save/clustering_comparison.png'):
    """
    对比谱聚类分簇和随机分簇的效果，使用标签分布的堆叠柱状图
    """
    n_es = len(es_label_distributions)
    n_clusters = len(np.unique(cluster_labels))
    n_classes = es_label_distributions.shape[1]

    print(f"[可视化] ES数量: {n_es}, 簇数: {n_clusters}, 类别数: {n_classes}")

    # 随机分簇，每个 ES 随机分到 0~n_clusters-1 的簇
    np.random.seed(42)  # 固定种子以便重现
    random_cluster_labels = np.random.randint(0, n_clusters, size=n_es)

    # 创建图形，包含两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), constrained_layout=True)
    fig.suptitle('Comparison of Spectral Clustering vs Random Clustering', fontsize=16)

    cmap = plt.cm.viridis

    # 【修复】预先生成颜色映射
    class_colors = [cmap(i / max(n_classes - 1, 1)) for i in range(n_classes)]
    print(f"[可视化] 生成 {n_classes} 种不同颜色")

    # 绘制谱聚类结果（左侧子图）
    _plot_gradient_clustering_result(
        ax1, es_label_distributions, cluster_labels,
        "Spectral Clustering Result", class_colors, n_classes
    )

    # 绘制随机分簇结果（右侧子图）
    _plot_gradient_clustering_result(
        ax2, es_label_distributions, random_cluster_labels,
        "Random Clustering Result", class_colors, n_classes
    )

    # 在所有子图下方添加连续颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=n_classes-1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax1, ax2], orientation='horizontal', fraction=0.05, pad=0.12)
    cbar.ax.tick_params(labelsize=8)

    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Clustering comparison visualization saved to: {save_path}")


def _plot_gradient_clustering_result(ax, distributions, labels, title, class_colors, n_classes):
    """
    绘制堆叠柱状图，每个标签使用不同的渐变色，实现真正的100种颜色效果

    参数:
        ax: matplotlib轴对象
        distributions: 标签分布数据，形状(n_es, n_classes)
        labels: 聚类标签
        title: 子图标题
        class_colors: 预生成的100种颜色列表
        n_classes: 类别数
    """
    unique_clusters = np.unique(labels)

    # 按聚类结果排序
    sorted_indices = np.argsort(labels)
    sorted_labels = labels[sorted_indices]
    sorted_distributions = distributions[sorted_indices]

    # 计算每个聚类的边界
    cluster_boundaries = []
    cluster_sizes = {}
    start_idx = 0
    for cluster_id in unique_clusters:
        cluster_size = np.sum(sorted_labels == cluster_id)
        cluster_boundaries.append((start_idx, start_idx + cluster_size - 1))
        cluster_sizes[cluster_id] = cluster_size
        start_idx += cluster_size

    # 创建x坐标（每个簇之间有间隙）
    x_positions = []
    current_x = 0
    for cluster_id in unique_clusters:
        for _ in range(cluster_sizes[cluster_id]):
            x_positions.append(current_x)
            current_x += 1
        current_x += 0.5  # 簇间 间隙
    x = np.array(x_positions)

    # 计算每个ES的总样本数
    total_samples = np.sum(sorted_distributions, axis=1)

    # 绘制堆叠柱状图 - 每个标签一个颜色
    bottom = np.zeros(len(sorted_distributions))

    for class_idx in range(n_classes):
        # 当前标签在所有ES中的样本数
        class_samples = sorted_distributions[:, class_idx]

        # 只绘制有数据的标签
        if np.sum(class_samples) > 0:
            # 为每个ES绘制该标签的堆叠部分
            for es_idx, (x_pos, samples, bottom_pos) in enumerate(zip(x, class_samples, bottom)):
                if samples > 0:
                    # 使用预生成的颜色
                    color = class_colors[class_idx]
                    # 绘制堆叠条形图
                    bar = ax.bar(x_pos, samples, bottom=bottom_pos,
                                 color=color, width=0.8, alpha=0.85,
                                 edgecolor='none', linewidth=0)
            # 更新底部位置
            bottom += class_samples

    # 设置x轴（簇标签）
    cluster_centers = []
    for boundary in cluster_boundaries:
        left_idx = boundary[0]
        right_idx = boundary[1]
        if left_idx <= right_idx:
            center = (x[left_idx] + x[right_idx]) / 2
        else:
            center = x[left_idx]
        cluster_centers.append(center)

    ax.set_xticks(cluster_centers)
    ax.set_xticklabels([f'Cluster {int(i)}' for i in unique_clusters], fontsize=10)

    # 设置标题和标签
    ax.set_title(title, fontsize=12, pad=20)
    ax.set_xlabel('Edge Server Cluster', fontsize=10)
    ax.set_ylabel('Number of Samples', fontsize=10)

    # 设置y轴范围
    max_height = np.max(bottom) * 1.05
    ax.set_ylim(0, max_height)

    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')


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
        label_count = np.zeros(args.num_classes)  # 动态类别数
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
    cluster_num, cluster_labels = cluster2(model_vectors, epsilon=epsilon)
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

def train_single_client_for_es(args, user_idx, dataset_train, dict_users, w_input, net_glob):
    """
    单个客户端的训练函数，用于ES初始化阶段的多进程训练
    
    Args:
        args: 训练参数
        user_idx: 客户端索引
        dataset_train: 训练数据集
        dict_users: 客户端数据索引字典
        w_input: 输入模型权重
        net_glob: 全局模型架构（用于重新构建模型）
    
    Returns:
        tuple: (user_idx, w_local, loss_local)
    """
    # 在子进程中重新构建模型架构
    if hasattr(dataset_train, '__getitem__'):
        img_size = dataset_train[0][0].shape
    else:
        # 如果无法获取，使用默认值
        if args.dataset == 'mnist':
            img_size = (1, 28, 28)
        elif args.dataset in ['cifar', 'cifar100']:
            img_size = (3, 32, 32)
        else:
            img_size = (1, 28, 28)  # 默认值

    if args.model == 'cnn':
        if args.dataset in ['cifar', 'cifar100']:
            local_net = CNNCifar(args=args).to(args.device)
        elif args.dataset == 'mnist':
            local_net = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        local_net = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    elif args.model == 'lr' and args.dataset == 'mnist':
        len_in = 1
        for x in img_size:
            len_in *= x
        local_net = LR(dim_in=len_in, dim_out=args.num_classes).to(args.device)
    elif args.model == 'lenet5' and args.dataset == 'mnist':
        local_net = LeNet5(args=args).to(args.device)
    elif args.model == 'vgg11' and args.dataset in ['cifar', 'cifar100']:
        local_net = VGG11(args=args).to(args.device)
    elif args.model == 'resnet18' and args.dataset in ['cifar', 'cifar100']:
        local_net = ResNet18(args=args).to(args.device)
    else:
        # 默认使用CNN
        if args.dataset in ['cifar', 'cifar100']:
            local_net = CNNCifar(args=args).to(args.device)
        else:
            local_net = CNNMnist(args=args).to(args.device)
    
    # 创建本地更新实例
    local = LocalUpdate(
        args=args,
        dataset=dataset_train,
        idxs=dict_users[user_idx]
    )
    
    # 加载输入权重
    local_net.load_state_dict(w_input)
    
    # 训练本地模型
    w_local, loss_local = local.train(net=local_net.to(args.device))
    
    # 返回结果，包括 user_idx 以便后续排序
    return (user_idx, copy.deepcopy(w_local), loss_local)

def train_initial_models_with_es_aggregation(args, dataset_train, dict_users, net_glob, A_design, num_users):
    """
    训练初始本地模型并聚合到ES层，遵循联邦学习机制
    
    Args:
        args: 训练参数
        dataset_train: 训练数据集
        dict_users: 客户端数据索引字典
        net_glob: 全局模型
        A_design: 客户端-ES关联矩阵
        num_users: 客户端数量
    
    Returns:
        es_models: ES层聚合后的模型列表
        client_label_distributions: 客户端标签分布
    """
    from models.Fed import FedAvg_layered
    
    print("Training initial local models with ES aggregation for clustering...")
    print(f"每个客户端训练 {args.local_ep} 轮本地更新")
    print(f"ES层聚合 {args.ES_k2} 轮")
    print("📊 ES层聚合将使用基于数据量的加权平均")
    
    # 构建C1层级结构（客户端->ES映射）
    num_ESs = A_design.shape[1]
    C1 = {j: [] for j in range(num_ESs)}
    for i in range(num_users):
        for j in range(num_ESs):
            if A_design[i][j] == 1:
                C1[j].append(i)
    
    print(f"客户端-ES映射关系: {dict(C1)}")
    
    # 计算每个客户端的数据量
    client_data_counts = {}
    for client_id, data_indices in dict_users.items():
        if isinstance(data_indices, set):
            client_data_counts[client_id] = len(data_indices)
        elif isinstance(data_indices, np.ndarray):
            client_data_counts[client_id] = len(data_indices)
        else:
            client_data_counts[client_id] = len(list(data_indices))
    
    print(f"📊 客户端数据量统计: 总数={sum(client_data_counts.values())}, 平均={np.mean(list(client_data_counts.values())):.1f}")
    print(f"📊 数据量范围: [{min(client_data_counts.values())}, {max(client_data_counts.values())}]")
    
    # 初始化ES层模型权重（从全局模型开始）
    w_glob = net_glob.state_dict()
    ESs_ws = [copy.deepcopy(w_glob) for _ in range(num_ESs)]
    
    # 计算每个客户端的标签分布
    client_label_distributions = []
    for user_idx in range(num_users):
        labels = [dataset_train[i][1] for i in dict_users[user_idx]]
        label_count = np.zeros(args.num_classes)
        for label in labels:
            label_count[label] += 1
        client_label_distributions.append(label_count)
    
    # ES层聚合k2轮
    for es_round in range(args.ES_k2):
        print(f"\n--- ES聚合轮次 {es_round + 1}/{args.ES_k2} ---")
        
        # ES层->客户端层权重分发
        w_locals_input = [None] * num_users
        for ES_idx, user_indices in C1.items():
            for user_idx in user_indices:
                w_locals_input[user_idx] = copy.deepcopy(ESs_ws[ES_idx])
        
        # 客户端本地训练 - 使用多进程并行
        w_locals_output = [None] * num_users
        
        # 准备并行训练任务
        print(f"  开始并行训练 {num_users} 个客户端，使用 5 个进程...")
        
        # 准备传递给每个子进程的参数
        tasks = []
        for user_idx in range(num_users):
            task_args = (
                args, user_idx, dataset_train, dict_users,
                w_locals_input[user_idx], net_glob
            )
            tasks.append(task_args)
        
        # 创建进程池并分发任务
        with mp.Pool(processes=5) as pool:
            results = pool.starmap(train_single_client_for_es, tqdm(tasks, desc=f"ES聚合轮次 {es_round + 1} 客户端训练"))
        
        # 收集并整理所有客户端的训练结果，按user_idx排序
        for result in results:
            u_idx, w_local, loss_local = result
            w_locals_output[u_idx] = w_local
            
            if u_idx < 5:  # 只打印前5个客户端的损失
                print(f"  客户端 {u_idx}: 损失 {loss_local:.4f}")
        
        # 客户端->ES层聚合 - 使用加权平均
        print(f"  📊 开始ES层加权平均聚合 (基于{len(client_data_counts)}个客户端数据量)")
        ESs_ws = FedAvg_layered(w_locals_output, C1, client_data_counts)
        print(f"  ES聚合完成，共聚合到 {len([es for es in ESs_ws if es is not None])} 个ES")
    
    print(f"\n✅ 初始训练完成！经过 {args.ES_k2} 轮ES加权聚合，生成 {len(ESs_ws)} 个ES模型用于谱聚类")
    print("📊 所有ES聚合均已使用基于数据量的加权平均，确保训练质量")
    
    return ESs_ws, np.array(client_label_distributions)