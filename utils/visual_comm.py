import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import warnings

# 禁用EPS透明度警告
warnings.filterwarnings('ignore', message='The PostScript backend does not support transparency')

# 创建保存目录
save_dir = './save/comm'
os.makedirs(save_dir, exist_ok=True)

# ================= 可调整的参数 =================
# 图片尺寸 (英寸)
FIG_SIZE = (10, 8)  # 宽度, 高度

# 字体大小
LABEL_FONT_SIZE = 24
TICK_FONT_SIZE = 20
LEGEND_FONT_SIZE = 18

# 颜色设置
COLORS = {
    'FedAvg': '#f6903c',  # 橙色
    'GP-HFL': '#55aa5a',  # 绿色
    'RC-HFL': '#4d84bd'  # 蓝色
}

# 柱状图图案设置
HATCHES = {
    'FedAvg': '',
    'GP-HFL': '//',
    'RC-HFL': '\\\\'
}

# 柱状图宽度和间距
BAR_WIDTH = 0.2  # 增加柱宽
GROUP_SPACING = 0.2  # 组间间距


# ================= 数据读取 =================
df = pd.read_excel('./result/communication.xlsx')
models = df['Network Scale'].unique().tolist()
datasets = df['dataset'].unique().tolist()
methods = ['FedAvg', 'RC-HFL', 'GP-HFL']

# 构造 completion_time 和 communications 字典，保持原有结构
completion_time = {
    'FedAvg': {},
    'GP-HFL': {},
    'RC-HFL': {}
}
communications = {
    'FedAvg': {},
    'GP-HFL': {},
    'RC-HFL': {}
}
for dataset in datasets:
    sub_df = df[df['dataset'] == dataset]
    completion_time['FedAvg'][dataset] = sub_df['sfl_t'].values * 12
    completion_time['GP-HFL'][dataset] = sub_df['hfl_cluster_t'].values * 12
    completion_time['RC-HFL'][dataset] = sub_df['hfl_random_t'].values * 12
    communications['FedAvg'][dataset] = sub_df['sfl_p'].values * 12
    communications['GP-HFL'][dataset] = sub_df['hfl_cluster_p'].values * 12
    communications['RC-HFL'][dataset] = sub_df['hfl_random_p'].values * 12

# 设置全局字体
plt.rcParams.update({
    'font.size': TICK_FONT_SIZE,
    'axes.labelsize': LABEL_FONT_SIZE,
    'xtick.labelsize': TICK_FONT_SIZE,
    'ytick.labelsize': TICK_FONT_SIZE,
    'legend.fontsize': LEGEND_FONT_SIZE,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],  # 指定衬线字体为Times New Roman
})

# ================= 图1 & 2: 网络规模 vs 通信时间 (MNIST 和 CIFAR-10) =================
for dataset in datasets:
    plt.figure(figsize=FIG_SIZE)
    x = np.arange(len(models)) * (len(methods) * BAR_WIDTH + GROUP_SPACING)

    for i, method in enumerate(methods):
        bars = plt.bar(x + i * BAR_WIDTH, completion_time[method][dataset], width=BAR_WIDTH,
                       label=method, color=COLORS[method], edgecolor='black', linewidth=0.8, alpha=1.0)
        for bar in bars:
            bar.set_hatch(HATCHES[method])

    plt.xticks(x + BAR_WIDTH * (len(methods) - 1) / 2, models)
    plt.xlabel('Network Scale', fontsize=LABEL_FONT_SIZE, fontweight='bold')
    plt.ylabel('Completion Time', fontsize=LABEL_FONT_SIZE, fontweight='bold')
    plt.yscale('log')
    yticks = [1, 10, 100, 1000, 10000, 100000]
    yticklabels = [r'$10^{{{}}}$'.format(int(np.log10(y))) for y in yticks]
    plt.yticks(yticks, yticklabels)
    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.tight_layout()
    filename = f'completion_time_vs_scale_{dataset}'
    plt.savefig(os.path.join(save_dir, filename + '.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, filename + '.eps'), format='eps', dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.show()
    plt.close()

# ================= 图3 & 4: 网络规模 vs 通信开销 (MNIST 和 CIFAR-10) =================
for dataset in datasets:
    plt.figure(figsize=FIG_SIZE)
    x = np.arange(len(models)) * (len(methods) * BAR_WIDTH + GROUP_SPACING)

    for i, method in enumerate(methods):
        bars = plt.bar(x + i * BAR_WIDTH, communications[method][dataset], width=BAR_WIDTH,
                       label=method, color=COLORS[method], edgecolor='black', linewidth=0.8, alpha=1.0)
        for bar in bars:
            bar.set_hatch(HATCHES[method])

    plt.xticks(x + BAR_WIDTH * (len(methods) - 1) / 2, models)
    plt.xlabel('Network Scale', fontsize=LABEL_FONT_SIZE, fontweight='bold')
    plt.ylabel('Power Consumption', fontsize=LABEL_FONT_SIZE, fontweight='bold')
    plt.yscale('log')
    yticks = [1, 10, 100, 1000, 10000, 100000, 1000000]
    yticklabels = [r'$10^{{{}}}$'.format(int(np.log10(y))) for y in yticks]
    plt.yticks(yticks, yticklabels)
    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.tight_layout()
    filename = f'Power Consumption_vs_scale_{dataset}'
    plt.savefig(os.path.join(save_dir, filename + '.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, filename + '.eps'), format='eps', dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.show()
    plt.close()

# ================= 图5-8: 数据集 vs 通信开销 (不同网络规模) =================
for model in models:
    idx = models.index(model)
    plt.figure(figsize=FIG_SIZE)
    x = np.arange(len(datasets)) * (len(methods) * BAR_WIDTH + GROUP_SPACING)

    for i, method in enumerate(methods):
        bars = plt.bar(x + i * BAR_WIDTH, [communications[method][dataset][idx] for dataset in datasets],
                       width=BAR_WIDTH,
                       label=method, color=COLORS[method], edgecolor='black', linewidth=0.8, alpha=1.0)
        for bar in bars:
            bar.set_hatch(HATCHES[method])

    plt.xticks(x + BAR_WIDTH * (len(methods) - 1) / 2, [d.upper() for d in datasets])
    plt.xlabel('Datasets', fontsize=LABEL_FONT_SIZE, fontweight='bold')
    plt.ylabel('Power Consumption', fontsize=LABEL_FONT_SIZE, fontweight='bold')
    plt.yscale('log')
    yticks = [1, 10, 100, 1000, 10000, 100000, 1000000]
    yticklabels = [r'$10^{{{}}}$'.format(int(np.log10(y))) for y in yticks]
    plt.yticks(yticks, yticklabels)
    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.tight_layout()
    filename = f'Power Consumption_vs_datasets_scale_{model}'
    plt.savefig(os.path.join(save_dir, filename + '.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, filename + '.eps'), format='eps', dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()

# ================= 图9-12: 数据集 vs 通信时间 (不同网络规模) =================
for model in models:
    idx = models.index(model)
    plt.figure(figsize=FIG_SIZE)
    x = np.arange(len(datasets)) * (len(methods) * BAR_WIDTH + GROUP_SPACING)

    for i, method in enumerate(methods):
        bars = plt.bar(x + i * BAR_WIDTH, [completion_time[method][dataset][idx] for dataset in datasets],
                       width=BAR_WIDTH,
                       label=method, color=COLORS[method], edgecolor='black', linewidth=0.8, alpha=1.0)
        for bar in bars:
            bar.set_hatch(HATCHES[method])

    plt.xticks(x + BAR_WIDTH * (len(methods) - 1) / 2, [d.upper() for d in datasets])
    plt.xlabel('Datasets', fontsize=LABEL_FONT_SIZE, fontweight='bold')
    plt.ylabel('Completion Time', fontsize=LABEL_FONT_SIZE, fontweight='bold')
    plt.yscale('log')
    yticks = [1, 10, 100, 1000, 10000, 100000]
    yticklabels = [r'$10^{{{}}}$'.format(int(np.log10(y))) for y in yticks]
    plt.yticks(yticks, yticklabels)
    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.tight_layout()
    filename = f'completion_time_vs_datasets_scale_{model}'
    plt.savefig(os.path.join(save_dir, filename + '.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, filename + '.eps'), format='eps', dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()

print(f"所有图表已保存到 {save_dir} 文件夹")