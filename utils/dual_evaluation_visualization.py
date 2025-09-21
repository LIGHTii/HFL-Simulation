#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
双重评估结果可视化工具
提供本地vs全局性能对比的高级可视化功能
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
import os

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def create_dual_evaluation_plots(dual_eval_history, save_dir='./save/', timestamp=''):
    """
    创建双重评估的综合可视化图表
    
    Args:
        dual_eval_history: 双重评估历史数据
        save_dir: 保存目录
        timestamp: 时间戳
    """
    if not dual_eval_history:
        print("警告: 没有双重评估数据")
        return
    
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 提取数据
    epochs = [result['epoch'] for result in dual_eval_history]
    model_names = list(dual_eval_history[0]['global_performance'].keys())
    
    # 创建大图表
    fig = plt.figure(figsize=(20, 16))
    
    # 定义颜色方案
    colors = {
        'HFL_Random_B': '#1f77b4',   # 蓝色
        'HFL_Cluster_B': '#ff7f0e',  # 橙色  
        'SFL': '#2ca02c'             # 绿色
    }
    
    # 定义中文标签
    labels_zh = {
        'HFL_Random_B': 'HFL(随机B矩阵)',
        'HFL_Cluster_B': 'HFL(聚类B矩阵)',
        'SFL': 'SFL(单层)'
    }
    
    # 1. 全局vs本地准确率对比 (2x3 布局)
    for i, model_name in enumerate(model_names):
        ax = plt.subplot(2, 3, i+1)
        
        # 提取该模型的数据
        global_accs = [r['global_performance'][model_name]['accuracy'] for r in dual_eval_history]
        local_mean_accs = [r['local_performance'][model_name]['mean_accuracy'] for r in dual_eval_history if model_name in r['local_performance']]
        local_std_accs = [r['local_performance'][model_name]['std_accuracy'] for r in dual_eval_history if model_name in r['local_performance']]
        
        if local_mean_accs:
            local_mean_accs = np.array(local_mean_accs)
            local_std_accs = np.array(local_std_accs)
            
            # 绘制全局准确率线
            ax.plot(epochs, global_accs, 'o-', color=colors[model_name], 
                   linewidth=2.5, markersize=6, label='全局测试', alpha=0.9)
            
            # 绘制本地平均准确率线和置信区间
            ax.plot(epochs, local_mean_accs, 's--', color=colors[model_name], 
                   linewidth=2, markersize=5, label='本地平均', alpha=0.7)
            ax.fill_between(epochs, local_mean_accs - local_std_accs, 
                           local_mean_accs + local_std_accs,
                           color=colors[model_name], alpha=0.2, label='本地方差')
        
        ax.set_title(f'{labels_zh[model_name]} - 准确率对比', fontsize=14, fontweight='bold')
        ax.set_xlabel('训练轮次', fontsize=12)
        ax.set_ylabel('准确率 (%)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
    
    # 2. 性能差距分析 (2x3 布局)
    for i, model_name in enumerate(model_names):
        ax = plt.subplot(2, 3, i+4)
        
        # 提取性能差距数据
        acc_gaps = [r['performance_comparison'][model_name]['accuracy_gap'] for r in dual_eval_history if model_name in r['performance_comparison']]
        local_variances = [r['performance_comparison'][model_name]['local_variance_acc'] for r in dual_eval_history if model_name in r['performance_comparison']]
        
        if acc_gaps:
            # 绘制准确率差距
            ax.bar(epochs, acc_gaps, color=colors[model_name], alpha=0.7, 
                  label='全局-本地差距', width=0.8)
            
            # 添加零线
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            # 绘制本地方差（作为误差线）
            ax2 = ax.twinx()
            ax2.plot(epochs, local_variances, 'ro-', alpha=0.6, markersize=4, 
                    label='客户端间方差')
            ax2.set_ylabel('本地方差 (%)', fontsize=10, color='red')
        
        ax.set_title(f'{labels_zh[model_name]} - 性能差距', fontsize=14, fontweight='bold')
        ax.set_xlabel('训练轮次', fontsize=12)
        ax.set_ylabel('准确率差距 (%)', fontsize=12)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/dual_evaluation_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/dual_evaluation_analysis_{timestamp}.pdf', bbox_inches='tight')
    print(f"双重评估分析图表已保存到: {save_dir}/dual_evaluation_analysis_{timestamp}.png")
    plt.close()


def create_client_performance_heatmap(dual_eval_history, save_dir='./save/', timestamp=''):
    """
    创建客户端性能热力图
    
    Args:
        dual_eval_history: 双重评估历史数据
        save_dir: 保存目录
        timestamp: 时间戳
    """
    if not dual_eval_history:
        return
    
    # 只分析最后几轮的数据
    recent_results = dual_eval_history[-3:] if len(dual_eval_history) > 3 else dual_eval_history
    model_names = list(dual_eval_history[0]['global_performance'].keys())
    
    fig, axes = plt.subplots(1, len(model_names), figsize=(6*len(model_names), 8))
    if len(model_names) == 1:
        axes = [axes]
    
    for idx, model_name in enumerate(model_names):
        # 收集所有客户端的性能数据
        all_client_data = []
        
        for result in recent_results:
            if model_name in result['local_performance']:
                individual_results = result['local_performance'][model_name]['individual_results']
                for client_result in individual_results:
                    all_client_data.append({
                        'epoch': result['epoch'],
                        'client_id': client_result['client_id'],
                        'accuracy': client_result['accuracy'],
                        'test_size': client_result['test_size']
                    })
        
        if all_client_data:
            # 转换为DataFrame
            df = pd.DataFrame(all_client_data)
            
            # 创建数据透视表
            pivot_table = df.pivot(index='client_id', columns='epoch', values='accuracy')
            
            # 绘制热力图
            sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='RdYlBu_r',
                       ax=axes[idx], cbar_kws={'label': '准确率 (%)'})
            
            axes[idx].set_title(f'{model_name} - 客户端性能热力图', fontsize=14, fontweight='bold')
            axes[idx].set_xlabel('训练轮次', fontsize=12)
            axes[idx].set_ylabel('客户端ID', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/client_performance_heatmap_{timestamp}.png', dpi=300, bbox_inches='tight')
    print(f"客户端性能热力图已保存到: {save_dir}/client_performance_heatmap_{timestamp}.png")
    plt.close()


def create_performance_summary_report(dual_eval_history, save_dir='./save/', timestamp=''):
    """
    创建性能总结报告
    
    Args:
        dual_eval_history: 双重评估历史数据
        save_dir: 保存目录
        timestamp: 时间戳
    """
    if not dual_eval_history:
        return
    
    # 分析最终性能
    final_result = dual_eval_history[-1]
    model_names = list(final_result['global_performance'].keys())
    
    # 创建总结报告
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 最终性能对比柱状图
    global_accs = [final_result['global_performance'][model]['accuracy'] for model in model_names]
    local_accs = [final_result['local_performance'][model]['mean_accuracy'] for model in model_names if model in final_result['local_performance']]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    ax1.bar(x - width/2, global_accs, width, label='全局测试', alpha=0.8)
    if local_accs:
        ax1.bar(x + width/2, local_accs, width, label='本地平均', alpha=0.8)
    
    ax1.set_xlabel('模型类型')
    ax1.set_ylabel('准确率 (%)')
    ax1.set_title('最终性能对比')
    ax1.set_xticks(x)
    ax1.set_xticklabels([name.replace('_', '\n') for name in model_names])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 性能差距分析
    acc_gaps = [final_result['performance_comparison'][model]['accuracy_gap'] for model in model_names if model in final_result['performance_comparison']]
    if acc_gaps:
        colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c'][:len(model_names)]
        bars = ax2.bar(model_names, acc_gaps, color=colors_list, alpha=0.7)
        
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.set_xlabel('模型类型')
        ax2.set_ylabel('准确率差距 (%)')
        ax2.set_title('全局-本地性能差距')
        ax2.set_xticklabels([name.replace('_', '\n') for name in model_names])
        
        # 添加数值标签
        for bar, gap in zip(bars, acc_gaps):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{gap:+.1f}%', ha='center', va='bottom')
    
    # 3. 客户端间方差分析
    variances = [final_result['performance_comparison'][model]['local_variance_acc'] for model in model_names if model in final_result['performance_comparison']]
    if variances:
        ax3.bar(model_names, variances, color=colors_list, alpha=0.7)
        ax3.set_xlabel('模型类型')
        ax3.set_ylabel('准确率标准差 (%)')
        ax3.set_title('客户端间性能方差')
        ax3.set_xticklabels([name.replace('_', '\n') for name in model_names])
        ax3.grid(True, alpha=0.3)
    
    # 4. 训练过程趋势
    epochs = [r['epoch'] for r in dual_eval_history]
    for model_name in model_names:
        global_trend = [r['global_performance'][model_name]['accuracy'] for r in dual_eval_history]
        ax4.plot(epochs, global_trend, 'o-', label=f'{model_name}(全局)', linewidth=2)
    
    ax4.set_xlabel('训练轮次')
    ax4.set_ylabel('准确率 (%)')
    ax4.set_title('训练过程准确率趋势')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/performance_summary_report_{timestamp}.png', dpi=300, bbox_inches='tight')
    print(f"性能总结报告已保存到: {save_dir}/performance_summary_report_{timestamp}.png")
    plt.close()


def generate_dual_evaluation_report(dual_eval_history, save_dir='./save/', timestamp=''):
    """
    生成完整的双重评估报告
    
    Args:
        dual_eval_history: 双重评估历史数据
        save_dir: 保存目录
        timestamp: 时间戳
    """
    print("\n" + "="*50)
    print("生成双重评估可视化报告")
    print("="*50)
    
    if not dual_eval_history:
        print("警告: 没有双重评估数据可供分析")
        return
    
    try:
        # 1. 创建综合分析图表
        create_dual_evaluation_plots(dual_eval_history, save_dir, timestamp)
        
        # 2. 创建客户端性能热力图
        create_client_performance_heatmap(dual_eval_history, save_dir, timestamp)
        
        # 3. 创建性能总结报告
        create_performance_summary_report(dual_eval_history, save_dir, timestamp)
        
        # 4. 生成数字化分析报告
        final_result = dual_eval_history[-1]
        
        print("\n📊 双重评估最终性能分析:")
        print("-" * 40)
        
        for model_name in final_result['global_performance'].keys():
            global_perf = final_result['global_performance'][model_name]
            local_perf = final_result['local_performance'].get(model_name, {})
            comparison = final_result['performance_comparison'].get(model_name, {})
            
            print(f"\n🔸 {model_name}:")
            print(f"   全局测试准确率: {global_perf['accuracy']:.2f}%")
            if local_perf:
                print(f"   本地平均准确率: {local_perf['mean_accuracy']:.2f} ± {local_perf['std_accuracy']:.2f}%")
                print(f"   性能差距: {comparison.get('accuracy_gap', 0):+.2f}%")
                print(f"   客户端间方差: {comparison.get('local_variance_acc', 0):.2f}%")
        
        print("\n" + "="*50)
        print("双重评估报告生成完成!")
        print(f"所有图表已保存到: {save_dir}")
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"生成双重评估报告时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("双重评估可视化工具测试")