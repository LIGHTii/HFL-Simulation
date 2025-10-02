#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 三种模型对比的高质量可视化工具

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import rcParams
import os

# 设置matplotlib的全局参数，提高图表质量
rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 12,
    'axes.labelsize': 24,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 16,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],  # 指定衬线字体为Times New Roman
    'axes.unicode_minus': False,
    'figure.figsize': (11, 8)
})


def clean_tensor_format(value):
    """
    清理tensor格式的数据，提取实际数值
    例: tensor(14.4400) -> 14.4400
    """
    if isinstance(value, str):
        if value.startswith('tensor(') and value.endswith(')'):
            numeric_part = value[7:-1]
            try:
                return float(numeric_part)
            except ValueError:
                return 0.0
    return value


def aggregate_eh_data(df, aggregation_method='mean'):
    """
    聚合EH层级的数据，将同一轮同一模型的多个EH记录合并
    """
    if 'level' not in df.columns:
        return df

    global_data = df[df['level'] == 'Global'].copy()
    eh_data = df[df['level'] == 'EH'].copy()

    if len(eh_data) == 0:
        return global_data

    group_columns = ['epoch', 'eh_round', 'es_round', 'model_type']

    if aggregation_method == 'mean':
        eh_aggregated = eh_data.groupby(group_columns).agg({
            'train_loss': 'mean',
            'test_loss': 'mean',
            'test_acc': 'mean'
        }).reset_index()
        eh_aggregated['level'] = 'EH_Aggregated'
        eh_aggregated['eh_idx'] = -1
        print(f"EH数据聚合完成: {len(eh_data)} 条记录 -> {len(eh_aggregated)} 条记录 (均值)")
    elif aggregation_method == 'median':
        eh_aggregated = eh_data.groupby(group_columns).agg({
            'train_loss': 'median',
            'test_loss': 'median',
            'test_acc': 'median'
        }).reset_index()
        eh_aggregated['level'] = 'EH_Aggregated'
        eh_aggregated['eh_idx'] = -1
        print(f"EH数据聚合完成: {len(eh_data)} 条记录 -> {len(eh_aggregated)} 条记录 (中位数)")
    else:  # boxplot_data
        eh_aggregated = eh_data.copy()
        eh_aggregated['level'] = 'EH_Aggregated'
        print(f"EH数据保留用于箱式图: {len(eh_aggregated)} 条记录")

    result_df = pd.concat([global_data, eh_aggregated], ignore_index=True)
    return result_df


def safe_format(value, format_str):
    """
    安全格式化数值，处理字符串和数值类型
    """
    try:
        if isinstance(value, str):
            value = float(value)
        return format_str.format(value)
    except (ValueError, TypeError):
        return 'N/A'


def convert_old_format_to_new(df):
    """
    将旧格式CSV转换为新格式
    """
    new_rows = []

    for _, row in df.iterrows():
        epoch = row['epoch']
        hfl_acc = clean_tensor_format(row['hfl_test_acc'])
        hfl_loss = row['hfl_test_loss']
        HFL_acc = clean_tensor_format(row['HFL_test_acc'])
        HFL_loss = row['HFL_test_loss']

        new_rows.append({
            'epoch': epoch,
            'eh_round': 1,
            'es_round': 1,
            'train_loss': hfl_loss,
            'test_loss': hfl_loss,
            'test_acc': hfl_acc,
            'model_type': 'HFL_Standard_B',
            'level': 'Global',
            'eh_idx': -1
        })
        new_rows.append({
            'epoch': epoch,
            'eh_round': 1,
            'es_round': 1,
            'train_loss': HFL_loss,
            'test_loss': HFL_loss,
            'test_acc': HFL_acc,
            'model_type': 'HFL',
            'level': 'Global',
            'eh_idx': -1
        })

    return pd.DataFrame(new_rows)


def create_comparison_plots(csv_file, save_dir='./save/'):
    """
    从CSV文件读取三种模型的训练结果，生成四张高质量对比图表
    """
    os.makedirs(save_dir, exist_ok=True)

    df = pd.read_csv(csv_file, encoding='utf-8')
    df['epoch'] = pd.to_numeric(df['epoch'], errors='coerce')

    columns = df.columns.tolist()
    is_new_format = 'model_type' in columns

    if not is_new_format:
        print("检测到旧格式CSV文件，转换为新格式...")
        df = convert_old_format_to_new(df)

    numeric_columns = ['train_loss', 'test_loss', 'test_acc', 'eh_round', 'es_round']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    print("正在聚合EH层级数据...")
    df_train = aggregate_eh_data(df, aggregation_method='mean')

    if len(df_train) == 0:
        print("警告: 没有找到训练数据，跳过可视化")
        return None, None, None, None, None

        # 过滤掉初始化epoch(-1)的数据
    df_train = df_train[df_train['epoch'] >= -1]

    # 创建组合时间轴：考虑epoch和EH轮次
    df_train = df_train.copy()
    if 'eh_round' in df_train.columns:
        eh_rate = 1 / df_train['eh_round'].max() if df_train['eh_round'].max() > 0 else 0.1
        df_train['combined_time'] = df_train.apply(
            lambda row: row['epoch'] if row['epoch'] == -1
            else row['epoch'] + (row['eh_round'] - 1) * eh_rate,
            axis=1
        )
    else:
        df_train['combined_time'] = df_train['epoch']

    df_train['combined_time'] = df_train['combined_time'] * 72

    # 过滤掉combined_time小于0的行
    df_train = df_train[df_train['combined_time'] >= 0]

    base_name = os.path.basename(csv_file).replace('.csv', '')
    colors = {'HFL_Random_B': '#0D5720', 'HFL_Cluster_B': '#e64012', 'HFL': '#BE1FCC'}
    markers = {'HFL_Random_B': 'o', 'HFL_Cluster_B': 's', 'HFL': '^'}
    labels = {'HFL_Random_B': 'HFL (Random B)', 'HFL_Cluster_B': 'HFL (Clustered B)', 'HFL': 'HFL'}

    # 1. 全局测试损失对比图
    plt.figure(figsize=(11, 8))
    for model in df_train['model_type'].unique():
        model_data = df_train[(df_train['model_type'] == model) &
                              (df_train['level'] == 'Global') &
                              (df_train['test_loss'].notna())]
        if len(model_data) > 0:
            plt.plot(model_data['combined_time'], model_data['test_loss'],
                     label=labels.get(model, model), linewidth=2, marker=markers.get(model, 'o'),
                     markersize=8, color=colors.get(model, '#000000'), alpha=0.8)

    plt.xlabel('Epoch', fontweight='bold')
    plt.ylabel('Global Test Loss', fontweight='bold')
    plt.legend(frameon=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    global_loss_file = os.path.join(save_dir, f'Global_test_loss_comparison_{base_name}.png')
    plt.savefig(global_loss_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # 2. 全局测试准确率对比图
    plt.figure(figsize=(11, 8))
    for model in df_train['model_type'].unique():
        model_data = df_train[(df_train['model_type'] == model) &
                              (df_train['level'] == 'Global') &
                              (df_train['test_acc'].notna())]
        if len(model_data) > 0:
            plt.plot(model_data['combined_time'], model_data['test_acc'],
                     label=labels.get(model, model), linewidth=2, marker=markers.get(model, 'o'),
                     markersize=8, color=colors.get(model, '#000000'), alpha=0.8)

    plt.xlabel('Epoch', fontweight='bold')
    plt.ylabel('Global Test Accuracy (%)', fontweight='bold')
    plt.legend(frameon=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    global_acc_file = os.path.join(save_dir, f'Global_test_acc_comparison_{base_name}.png')
    plt.savefig(global_acc_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # 3. EH测试损失对比图
    plt.figure(figsize=(11, 8))
    for model in df_train['model_type'].unique():
        model_data = df_train[(df_train['model_type'] == model) &
                              ((df_train['level'] == 'EH_Aggregated') |
                               (df_train['model_type'] == 'HFL')) &
                              (df_train['test_loss'].notna())].copy()
        if len(model_data) > 0:
            plt.plot(model_data['combined_time'], model_data['test_loss'],
                     label=labels.get(model, model), linewidth=2, marker=markers.get(model, 'o'),
                     markersize=8, color=colors.get(model, "#32A544"), alpha=0.8)

    plt.xlabel('Training Progress (Epoch + EH Round)', fontweight='bold')
    plt.ylabel('EH Test Loss', fontweight='bold')
    plt.legend(frameon=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    EH_loss_file = os.path.join(save_dir, f'EH_test_loss_comparison_{base_name}.png')
    plt.savefig(EH_loss_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # 4. EH测试准确率对比图
    plt.figure(figsize=(11, 8))  # 统一尺寸，之前为 (11, 8)
    for model in df_train['model_type'].unique():
        model_data = df_train[(df_train['model_type'] == model) &
                              ((df_train['level'] == 'EH_Aggregated') |
                               (df_train['model_type'] == 'HFL')) &
                              (df_train['test_acc'].notna())].copy()
        if len(model_data) > 0:
            plt.plot(model_data['combined_time'], model_data['test_acc'],
                     label=labels.get(model, model), linewidth=2, marker=markers.get(model, 'o'),
                     markersize=8, color=colors.get(model, '#000000'), alpha=0.8)

    plt.xlabel('Training Progress (Epoch + EH Round)', fontweight='bold')
    plt.ylabel('EH Test Accuracy (%)', fontweight='bold')
    plt.legend(frameon=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    EH_acc_file = os.path.join(save_dir, f'EH_test_accuracy_comparison_{base_name}.png')
    plt.savefig(EH_acc_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # 5. 生成性能总结表
    performance_summary = []
    final_data = df_train[df_train['epoch'] == df_train['epoch'].max()]

    for _, row in final_data.iterrows():
        performance_summary.append({
            'Model': labels.get(row['model_type'], row['model_type'].replace('_', ' ')),
            'Final Global Test Loss': safe_format(row['test_loss'] if row['level'] == 'Global' else None, "{:.4f}"),
            'Final Global Test Accuracy': safe_format(row['test_acc'] if row['level'] == 'Global' else None, "{:.2f}%"),
            'Final EH Test Loss': safe_format(row['test_loss'] if row['level'] == 'EH_Aggregated' else None, "{:.4f}"),
            'Final EH Test Accuracy': safe_format(row['test_acc'] if row['level'] == 'EH_Aggregated' else None,
                                                  "{:.2f}%"),
            'Final Train Loss': safe_format(row['train_loss'], "{:.4f}")
        })

    summary_df = pd.DataFrame(performance_summary)
    summary_file = os.path.join(save_dir, f'performance_summary_{base_name}.csv')
    summary_df.to_csv(summary_file, index=False, encoding='utf-8')

    print("\n=== 性能总结 ===")
    print(summary_df.to_string(index=False))

    print(f"\n=== 可视化完成 ===")
    print(f"所有图表保存到: {save_dir}")
    print(f"- 全局测试损失对比: {global_loss_file}")
    print(f"- 全局测试准确率对比: {global_acc_file}")
    print(f"- EH测试损失对比: {EH_loss_file}")
    print(f"- EH测试准确率对比: {EH_acc_file}")
    print(f"- 性能总结表: {summary_file}")

    return global_loss_file, global_acc_file, EH_loss_file, EH_acc_file, summary_file


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("使用方法: python visualization_tool.py <csv_file_path>")
        print("例如: python visualization_tool.py ./results/training_results_xxx.csv")
    else:
        csv_file = sys.argv[1]
        if os.path.exists(csv_file):
            create_comparison_plots(csv_file)
        else:

            print(f"错误: 文件 {csv_file} 不存在")