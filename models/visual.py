def visualize_client_label_distribution(client_label_distributions, dict_users, dataset_train,
                                        save_path='./save/client_label_distribution.png'):
    """
    可视化每个客户端的数据分布情况

    参数:
        client_label_distributions: 每个客户端的标签分布，形状为(n_clients, 10)
        dict_users: 用户数据索引字典
        dataset_train: 训练数据集
        save_path: 保存路径
    """
    import matplotlib.pyplot as plt
    import numpy as np

    num_clients = len(dict_users)

    # 创建图形
    plt.figure(figsize=(16, 8))

    # 创建堆叠柱状图
    x = np.arange(num_clients)
    bottom = np.zeros(num_clients)

    # 定义标签颜色
    label_colors = plt.cm.Set3(np.linspace(0, 1, 10))

    for i in range(10):  # 0-9共10个标签
        values = client_label_distributions[:, i]
        plt.bar(x, values, bottom=bottom, color=label_colors[i], label=f'Label {i}', alpha=0.8)
        bottom += values

    plt.title('Label Distribution Across Different Clients')
    plt.xlabel('Client ID')
    plt.ylabel('Number of Samples')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')

    # 添加每个客户端总样本数的标注
    for i, total in enumerate(np.sum(client_label_distributions, axis=1)):
        plt.text(i, total + 5, str(int(total)), ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"客户端数据分布可视化已保存到: {save_path}")