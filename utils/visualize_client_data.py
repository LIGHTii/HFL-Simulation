import matplotlib.pyplot as plt
import numpy as np
import os


def visualize_client_data_distribution(dict_users, dataset_train, args):
    """可视化每个客户端的数据分布"""
    if not os.path.exists('./save'):
        os.makedirs('./save')
        
    # 设置使用英文字体，避免中文字体警告
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'axes.unicode_minus': True  # 正确显示负号
    })

    num_users = args.num_users
    num_classes = args.num_classes if hasattr(args, 'num_classes') else 10  # 默认为10类

    # 初始化统计数组
    client_data_count = [len(dict_users[i]) for i in range(num_users)]
    client_class_count = np.zeros((num_users, num_classes))

    # 计算每个客户端拥有的每个类别的数据量
    for client_idx in range(num_users):
        for img_idx in dict_users[client_idx]:
            _, label = dataset_train[img_idx]
            client_class_count[client_idx][label] += 1

    # 绘制每个客户端的数据量
    plt.figure(figsize=(12, 6))
    plt.bar(range(num_users), client_data_count, color='skyblue')
    plt.xlabel('Client ID', fontsize=12)
    plt.ylabel('Number of Training Samples', fontsize=12)
    plt.title('Data Distribution Across Clients', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('./save/client_data_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 绘制每个客户端的类别分布
    rows = (num_users // 5) + (1 if num_users % 5 > 0 else 0)
    plt.figure(figsize=(15, 3 * rows))

    for i in range(num_users):
        plt.subplot(rows, 5, i + 1)
        bars = plt.bar(range(num_classes), client_class_count[i], color='lightcoral')
        plt.title(f'Client {i}', fontsize=10)

        # 添加数据标签
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # 只为非零值添加标签
                plt.text(bar.get_x() + bar.get_width() / 2., height + 1,
                         int(height), ha='center', va='bottom', fontsize=8, rotation=90)

        plt.xticks(range(num_classes), fontsize=8)
        plt.yticks(fontsize=8)

        if i % 5 == 0:
            plt.ylabel('Sample Count', fontsize=10)
        if i >= num_users - (num_users % 5 if num_users % 5 > 0 else 5):
            plt.xlabel('Class', fontsize=10)

    plt.tight_layout()
    plt.savefig('./save/client_class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("客户端数据分布可视化已保存到 ./save/ 目录")