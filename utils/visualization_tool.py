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
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'font.family': 'DejaVu Sans',
    'axes.unicode_minus': False,
    'figure.figsize': (12, 8)
})

def clean_tensor_format(value):
    """
    清理tensor格式的数据，提取实际数值
    例: tensor(14.4400) -> 14.4400
    """
    if isinstance(value, str):
        if value.startswith('tensor(') and value.endswith(')'):
            # 提取括号内的数值
            numeric_part = value[7:-1]  # 去掉 'tensor(' 和 ')'
            try:
                return float(numeric_part)
            except ValueError:
                return 0.0
    return value

def aggregate_eh_data(df, aggregation_method='mean'):
    """
    聚合EH层级的数据，将同一轮同一模型的多个EH记录合并
    
    Args:
        df: 原始数据DataFrame
        aggregation_method: 聚合方法 ('mean', 'median', 'boxplot_data')
    
    Returns:
        aggregated_df: 聚合后的DataFrame
    """
    if 'level' not in df.columns:
        return df
    
    # 分离Global级别和EH级别的数据
    global_data = df[df['level'] == 'Global'].copy()
    eh_data = df[df['level'] == 'EH'].copy()
    
    if len(eh_data) == 0:
        return global_data
    
    # 对EH数据按 epoch, eh_round, es_round, model_type 分组聚合
    group_columns = ['epoch', 'eh_round', 'es_round', 'model_type']
    
    if aggregation_method == 'mean':
        # 计算均值
        eh_aggregated = eh_data.groupby(group_columns).agg({
            'train_loss': 'mean',
            'test_loss': 'mean', 
            'test_acc': 'mean'
        }).reset_index()
        
        # 添加必要的列
        eh_aggregated['level'] = 'EH_Aggregated'
        eh_aggregated['eh_idx'] = -1  # 聚合数据没有具体的eh_idx
        
        print(f"EH数据聚合完成: {len(eh_data)} 条记录 -> {len(eh_aggregated)} 条记录 (均值)")
        
    elif aggregation_method == 'median':
        # 计算中位数
        eh_aggregated = eh_data.groupby(group_columns).agg({
            'train_loss': 'median',
            'test_loss': 'median',
            'test_acc': 'median'
        }).reset_index()
        
        eh_aggregated['level'] = 'EH_Aggregated'
        eh_aggregated['eh_idx'] = -1
        
        print(f"EH数据聚合完成: {len(eh_data)} 条记录 -> {len(eh_aggregated)} 条记录 (中位数)")
        
    else:  # aggregation_method == 'boxplot_data'
        # 保留所有EH数据用于箱式图，但标记为聚合数据
        eh_aggregated = eh_data.copy()
        eh_aggregated['level'] = 'EH_Aggregated'
        
        print(f"EH数据保留用于箱式图: {len(eh_aggregated)} 条记录")
    
    # 合并Global数据和聚合后的EH数据
    result_df = pd.concat([global_data, eh_aggregated], ignore_index=True)
    
    return result_df

def safe_format(value, format_str):
    """
    安全格式化数值，处理字符串和数值类型
    """
    try:
        # 如果是字符串，尝试转换为数值
        if isinstance(value, str):
            value = float(value)
        return format_str.format(value)
    except (ValueError, TypeError):
        return str(value)

def convert_old_format_to_new(df):
    """
    将旧格式CSV转换为新格式
    旧格式: epoch,hfl_test_acc,hfl_test_loss,sfl_test_acc,sfl_test_loss
    新格式: epoch,eh_round,es_round,train_loss,test_loss,test_acc,model_type,level,eh_idx
    """
    new_rows = []
    
    for _, row in df.iterrows():
        epoch = row['epoch']
        
        # 清理tensor格式的数据
        hfl_acc = clean_tensor_format(row['hfl_test_acc'])
        hfl_loss = row['hfl_test_loss']
        sfl_acc = clean_tensor_format(row['sfl_test_acc'])
        sfl_loss = row['sfl_test_loss']
        
        # 创建HFL模型记录 (假设这是Random B版本)
        # new_rows.append({
        #     'epoch': epoch,
        #     'eh_round': 1,
        #     'es_round': 1,
        #     'train_loss': hfl_loss,  # 旧格式没有训练损失，用测试损失代替
        #     'test_loss': hfl_loss,
        #     'test_acc': hfl_acc,
        #     'model_type': 'HFL_Random_B',
        #     'level': 'Global',
        #     'eh_idx': -1
        # })
        
        # 创建SFL模型记录
        # new_rows.append({
        #     'epoch': epoch,
        #     'eh_round': 1,
        #     'es_round': 1,
        #     'train_loss': sfl_loss,  # 旧格式没有训练损失，用测试损失代替
        #     'test_loss': sfl_loss,
        #     'test_acc': sfl_acc,
        #     'model_type': 'SFL',
        #     'level': 'Global',
        #     'eh_idx': -1
        # })
    
    return pd.DataFrame(new_rows)

def create_comparison_plots(csv_file, save_dir='./save/'):
    """
    从CSV文件读取三种模型的训练结果，生成高质量对比图表
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 读取CSV数据
    df = pd.read_csv(csv_file, encoding='utf-8')
    
    # 确保epoch列为数值类型
    df['epoch'] = pd.to_numeric(df['epoch'], errors='coerce')
    
    # 检测CSV格式 - 判断是新格式还是旧格式
    columns = df.columns.tolist()
    is_new_format = 'model_type' in columns
    
    if not is_new_format:
        # 旧格式: 只有HFL和SFL两种模型
        print("检测到旧格式CSV文件，转换为新格式...")
        df = convert_old_format_to_new(df)
    
    # 确保数值列为数值类型
    numeric_columns = ['train_loss', 'test_loss', 'test_acc', 'eh_round', 'es_round']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 应用EH数据聚合（默认使用均值方法）
    print("正在聚合EH层级数据...")
    df_train = aggregate_eh_data(df, aggregation_method='mean')
    
    # 过滤掉初始化epoch(-1)的数据
    df_train = df_train[df_train['epoch'] >= -1]
    
    # 获取基本信息用于文件命名
    base_name = os.path.basename(csv_file).replace('.csv', '')
    
    # 创建组合时间轴：考虑epoch和EH轮次
    df_train = df_train.copy()
    eh_rate = 1/df_train['eh_round'].max()
    df_train['combined_time'] = df_train['epoch'] + (df_train['eh_round']-1) * eh_rate
    print(df_train['combined_time'])

    
    # 1. 训练损失对比图
    plt.figure(figsize=(14, 10))
    
    for model in df_train['model_type'].unique():
        print(model)
        # 只显示Global级别的训练损失（EH级别没有训练损失）
        model_data = df_train[(df_train['model_type'] == model) & 
                             (df_train['level'] == 'Global') & 
                             (df_train['train_loss'] > 0) &
                              (df_train['epoch'] <20)]
        
        if len(model_data) > 0:
            if model == 'HFL_Random_B':
                plt.plot(model_data['epoch'], model_data['train_loss'],
                        label='HFL (Random B Matrix)', 
                        linewidth=3, marker='o', markersize=6, alpha=0.8)
            elif model == 'HFL_Cluster_B':
                plt.plot(model_data['epoch'], model_data['train_loss'],
                        label='HFL (Clustered B Matrix)', 
                        linewidth=3, marker='s', markersize=6, alpha=0.8)
            elif model == 'SFL':
                plt.plot(model_data['epoch'], model_data['train_loss'],
                        label='SFL (Single Layer)', 
                        linewidth=3, marker='^', markersize=6, alpha=0.8)
            # plt.plot(model_data['epoch'], model_data['train_loss'],
            #         label='HFL (Clustered B Matrix)',
            #         linewidth=3, marker='s', markersize=6, alpha=0.8)
        elif model == 'SFL':
            plt.plot(model_data['epoch'], model_data['train_loss'], 
                    label='SFL (Single Layer)', 
                    linewidth=3, marker='^', markersize=6, alpha=0.8)
    
    plt.xlabel('Epoch', fontweight='bold')
    plt.ylabel('Training Loss', fontweight='bold')
    plt.title('Training Loss Comparison: Three Federated Learning Approaches', 
              fontweight='bold', pad=20)
    plt.legend(frameon=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    train_loss_file = os.path.join(save_dir, f'new_train_loss_comparison_{base_name}.png')
    plt.savefig(train_loss_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 2. 测试损失对比图
    plt.figure(figsize=(14, 8))
    
    for model in df_train['model_type'].unique():
        # 合并Global和EH_Aggregated级别的测试数据
        model_data = df_train[(df_train['model_type'] == model) &
                             ((df_train['level'] != 'Global')|(model=='SFL')|(df_train['epoch']==-1))].copy()

        
        if len(model_data) > 0:
            if model == 'HFL_Random_B':
                plt.plot(model_data['combined_time'], model_data['test_loss'], 
                        label='HFL (Random B Matrix)', 
                        linewidth=3, marker='o', markersize=6, alpha=0.8)
            elif model == 'HFL_Cluster_B':
                plt.plot(model_data['combined_time'], model_data['test_loss'], 
                        label='HFL (Clustered B Matrix)', 
                        linewidth=3, marker='s', markersize=6, alpha=0.8)
            elif model == 'SFL':
                plt.plot(model_data['combined_time'], model_data['test_loss'], 
                        label='SFL (Single Layer)', 
                        linewidth=3, marker='^', markersize=6, alpha=0.8)
    
    plt.xlabel('Training Progress (Epoch + EH Round)', fontweight='bold')
    plt.ylabel('Test Loss', fontweight='bold')
    plt.title('Test Loss Comparison: Three Federated Learning Approaches (EH Data Aggregated)', 
              fontweight='bold', pad=20)
    plt.legend(frameon=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    test_loss_file = os.path.join(save_dir, f'new_test_loss_comparison_{base_name}.png')
    plt.savefig(test_loss_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 3. 测试准确率对比图
    plt.figure(figsize=(14, 8))
    
    for model in df_train['model_type'].unique():
        # 合并Global和EH_Aggregated级别的测试数据
        model_data = df_train[(df_train['model_type'] == model) &
                             ((df_train['level'] != 'Global')|(model=='SFL')|(df_train['epoch']==-1))].copy()
        
        if len(model_data) > 0:
            if model == 'HFL_Random_B':
                plt.plot(model_data['combined_time'], model_data['test_acc'], 
                        label='HFL (Random B Matrix)', 
                        linewidth=3, marker='o', markersize=6, alpha=0.8)
            elif model == 'HFL_Cluster_B':
                plt.plot(model_data['combined_time'], model_data['test_acc'], 
                        label='HFL (Clustered B Matrix)', 
                        linewidth=3, marker='s', markersize=6, alpha=0.8)
            elif model == 'SFL':
                plt.plot(model_data['combined_time'], model_data['test_acc'], 
                        label='SFL (Single Layer)', 
                        linewidth=3, marker='^', markersize=6, alpha=0.8)
    
    plt.xlabel('Training Progress (Epoch + EH Round)', fontweight='bold')
    plt.ylabel('Test Accuracy (%)', fontweight='bold')
    plt.title('Test Accuracy Comparison: Three Federated Learning Approaches (EH Data Aggregated)', 
              fontweight='bold', pad=20)
    plt.legend(frameon=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    test_acc_file = os.path.join(save_dir, f'new_test_accuracy_comparison_{base_name}.png')
    plt.savefig(test_acc_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 4. 综合对比图（2x2子图）
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comprehensive Comparison: Three Federated Learning Approaches', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # 训练损失
    for model in df_train['model_type'].unique():
        model_data = df_train[df_train['model_type'] == model]
        
        if model == 'HFL_Random_B':
            ax1.plot(model_data['epoch'], model_data['train_loss'], 
                    label='HFL (Random B)', linewidth=2, marker='o', markersize=4)
        elif model == 'HFL_Cluster_B':
            ax1.plot(model_data['epoch'], model_data['train_loss'], 
                    label='HFL (Clustered B)', linewidth=2, marker='s', markersize=4)
        elif model == 'SFL':
            ax1.plot(model_data['epoch'], model_data['train_loss'], 
                    label='SFL', linewidth=2, marker='^', markersize=4)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 测试损失
    for model in df_train['model_type'].unique():
        model_data = df_train[df_train['model_type'] == model]
        
        if model == 'HFL_Random_B':
            ax2.plot(model_data['combined_time'], model_data['test_loss'], 
                    label='HFL (Random B)', linewidth=2, marker='o', markersize=4)
        elif model == 'HFL_Cluster_B':
            ax2.plot(model_data['combined_time'], model_data['test_loss'], 
                    label='HFL (Clustered B)', linewidth=2, marker='s', markersize=4)
        elif model == 'SFL':
            ax2.plot(model_data['combined_time'], model_data['test_loss'], 
                    label='SFL', linewidth=2, marker='^', markersize=4)
    
    ax2.set_xlabel('Training Progress')
    ax2.set_ylabel('Test Loss')
    ax2.set_title('Test Loss', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 测试准确率
    for model in df_train['model_type'].unique():
        model_data = df_train[df_train['model_type'] == model]
        
        if model == 'HFL_Random_B':
            ax3.plot(model_data['combined_time'], model_data['test_acc'], 
                    label='HFL (Random B)', linewidth=2, marker='o', markersize=4)
        elif model == 'HFL_Cluster_B':
            ax3.plot(model_data['combined_time'], model_data['test_acc'], 
                    label='HFL (Clustered B)', linewidth=2, marker='s', markersize=4)
        elif model == 'SFL':
            ax3.plot(model_data['combined_time'], model_data['test_acc'], 
                    label='SFL', linewidth=2, marker='^', markersize=4)
    
    ax3.set_xlabel('Training Progress')
    ax3.set_ylabel('Test Accuracy (%)')
    ax3.set_title('Test Accuracy', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 最终性能对比（柱状图）
    if len(df_train) > 0:
        # 获取最后一个epoch的数据
        final_data = df_train[df_train['epoch'] == df_train['epoch'].max()]
        
        models = final_data['model_type'].tolist()
        test_accs = final_data['test_acc'].tolist()
        
        bars = ax4.bar(range(len(models)), test_accs, alpha=0.7, 
                      color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax4.set_xlabel('Model')
        ax4.set_ylabel('Final Test Accuracy (%)')
        ax4.set_title('Final Performance Comparison', fontweight='bold')
        ax4.set_xticks(range(len(models)))
        ax4.set_xticklabels([m.replace('_', ' ') for m in models], rotation=15)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 在柱状图上添加数值标签
        for bar, acc in zip(bars, test_accs):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    safe_format(acc, '{:.1f}%'), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # 为主标题留出空间
    
    comprehensive_file = os.path.join(save_dir, f'comprehensive_comparison_{base_name}.png')
    plt.savefig(comprehensive_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 5. 生成性能总结表
    if len(df_train) > 0:
        # 获取最后一个epoch的数据作为最终性能
        final_performance = df_train[df_train['epoch'] == df_train['epoch'].max()]
        
        performance_summary = []
        for _, row in final_performance.iterrows():
            performance_summary.append({
                'Model': row['model_type'].replace('_', ' '),
                'Final Train Loss': safe_format(row['train_loss'], "{:.4f}"),
                'Final Test Loss': safe_format(row['test_loss'], "{:.4f}"),
                'Final Test Accuracy': safe_format(row['test_acc'], "{:.2f}%")
            })
        
        # 保存性能总结到CSV
        summary_df = pd.DataFrame(performance_summary)
        summary_file = os.path.join(save_dir, f'performance_summary_{base_name}.csv')
        summary_df.to_csv(summary_file, index=False, encoding='utf-8')
        
        print("\n=== 性能总结 ===")
        print(summary_df.to_string(index=False))
    
    print(f"\n=== 可视化完成 ===")
    print(f"所有图表保存到: {save_dir}")
    print(f"- 训练损失对比: {train_loss_file}")
    print(f"- 测试损失对比: {test_loss_file}")  
    print(f"- 测试准确率对比: {test_acc_file}")
    print(f"- 综合对比图: {comprehensive_file}")
    if len(df_train) > 0:
        print(f"- 性能总结表: {summary_file}")

def create_enhanced_visualizations(csv_file, save_dir='./save/', eh_aggregation='mean'):
    """
    创建增强版可视化图表，支持EH数据聚合
    
    Args:
        csv_file: CSV数据文件路径
        save_dir: 保存目录
        eh_aggregation: EH数据聚合方法 ('mean', 'median', 'boxplot')
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 读取CSV数据
    df = pd.read_csv(csv_file, encoding='utf-8')
    
    # 确保epoch列为数值类型
    df['epoch'] = pd.to_numeric(df['epoch'], errors='coerce')
    
    # 检测CSV格式 - 判断是新格式还是旧格式
    columns = df.columns.tolist()
    is_new_format = 'model_type' in columns
    
    if not is_new_format:
        # 旧格式: 只有HFL和SFL两种模型
        print("检测到旧格式CSV文件，转换为新格式...")
        df = convert_old_format_to_new(df)
    
    # 确保数值列为数值类型
    numeric_columns = ['train_loss', 'test_loss', 'test_acc', 'eh_round', 'es_round']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 应用EH数据聚合
    print(f"正在聚合EH层级数据（方法: {eh_aggregation}）...")
    if eh_aggregation == 'boxplot':
        df_train = aggregate_eh_data(df, aggregation_method='boxplot_data')
    else:
        df_train = aggregate_eh_data(df, aggregation_method=eh_aggregation)
    
    # 过滤掉初始化epoch(-1)的数据
    df_train = df_train[df_train['epoch'] >= 0]
    
    if len(df_train) == 0:
        print("警告: 没有找到训练数据，跳过可视化")
        return
    
    # 创建组合时间轴：考虑epoch和EH轮次
    df_train = df_train.copy()
    df_train['combined_time'] = df_train['epoch'] + (df_train['eh_round'] - 1) * 0.1
    
    # 获取基本信息用于文件命名
    base_name = os.path.basename(csv_file).replace('.csv', '')
    
    # 设置高质量绘图参数
    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'font.size': 11,
        'axes.labelsize': 13,
        'axes.titlesize': 15,
        'legend.fontsize': 11,
        'figure.figsize': (12, 8)
    })
    
    # 综合对比图（2x2子图）
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Three Federated Learning Approaches Comparison', 
                fontsize=18, fontweight='bold', y=0.98)
    
    colors = {'HFL_Random_B': '#1f77b4', 'HFL_Cluster_B': '#ff7f0e', 'SFL': '#2ca02c'}
    markers = {'HFL_Random_B': 'o', 'HFL_Cluster_B': 's', 'SFL': '^'}
    labels = {'HFL_Random_B': 'HFL (Random B)', 'HFL_Cluster_B': 'HFL (Clustered B)', 'SFL': 'SFL'}
    
    # 训练损失
    for model in df_train['model_type'].unique():
        model_data = df_train[df_train['model_type'] == model]
        ax1.plot(model_data['epoch'], model_data['train_loss'], 
                label=labels[model], linewidth=2.5, 
                marker=markers[model], markersize=5, 
                color=colors[model], alpha=0.8)
    
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Training Loss', fontweight='bold')
    ax1.set_title('Training Loss Comparison', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 测试损失
    for model in df_train['model_type'].unique():
        model_data = df_train[df_train['model_type'] == model]
        ax2.plot(model_data['combined_time'], model_data['test_loss'], 
                label=labels[model], linewidth=2.5, 
                marker=markers[model], markersize=5, 
                color=colors[model], alpha=0.8)
    
    ax2.set_xlabel('Training Progress (Epoch + EH Round)', fontweight='bold')
    ax2.set_ylabel('Test Loss', fontweight='bold')
    ax2.set_title('Test Loss Comparison', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 测试准确率
    for model in df_train['model_type'].unique():
        model_data = df_train[df_train['model_type'] == model]
        ax3.plot(model_data['combined_time'], model_data['test_acc'], 
                label=labels[model], linewidth=2.5, 
                marker=markers[model], markersize=5, 
                color=colors[model], alpha=0.8)
    
    ax3.set_xlabel('Training Progress (Epoch + EH Round)', fontweight='bold')
    ax3.set_ylabel('Test Accuracy (%)', fontweight='bold')
    ax3.set_title('Test Accuracy Comparison', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 最终性能对比（柱状图）
    final_data = df_train[df_train['epoch'] == df_train['epoch'].max()]
    
    model_names = []
    test_accs = []
    for _, row in final_data.iterrows():
        model_names.append(labels[row['model_type']])
        test_accs.append(row['test_acc'])
    
    bars = ax4.bar(range(len(model_names)), test_accs, 
                  color=[colors[list(labels.keys())[list(labels.values()).index(name)]] for name in model_names],
                  alpha=0.7, edgecolor='black', linewidth=1)
    ax4.set_xlabel('Model', fontweight='bold')
    ax4.set_ylabel('Final Test Accuracy (%)', fontweight='bold')
    ax4.set_title('Final Performance Comparison', fontweight='bold')
    ax4.set_xticks(range(len(model_names)))
    ax4.set_xticklabels(model_names, rotation=0)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 在柱状图上添加数值标签
    for bar, acc in zip(bars, test_accs):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                safe_format(acc, '{:.1f}%'), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    comprehensive_file = os.path.join(save_dir, f'comprehensive_comparison_{base_name}.png')
    plt.savefig(comprehensive_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 生成性能总结
    performance_summary = []
    for _, row in final_data.iterrows():
        performance_summary.append({
            'Model': labels[row['model_type']],
            'Final Test Accuracy': safe_format(row['test_acc'], "{:.2f}%"),
            'Final Test Loss': safe_format(row['test_loss'], "{:.4f}"),
            'Final Train Loss': safe_format(row['train_loss'], "{:.4f}")
        })
    
    summary_df = pd.DataFrame(performance_summary)
    summary_file = os.path.join(save_dir, f'performance_summary_{base_name}.csv')
    summary_df.to_csv(summary_file, index=False, encoding='utf-8')
    
    return comprehensive_file, summary_file

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