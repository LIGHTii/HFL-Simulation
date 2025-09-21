#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 快速测试脚本，验证三种模型对比功能

import subprocess
import sys
import os
import time

def run_quick_test():
    """运行快速测试验证三种模型对比功能"""
    print("="*60)
    print("三种联邦学习模型对比 - 快速测试")
    print("="*60)
    
    # 测试参数：小规模快速测试
    test_params = [
        "python", "main_fed.py",
        "--epochs", "2",
        "--num_users", "10", 
        "--local_ep", "3",
        "--dataset", "mnist",
        "--model", "cnn",
        "--k2", "2",
        "--k3", "2", 
        "--num_processes", "4",
        "--iid",
        "--verbose"
    ]
    
    print(f"测试命令: {' '.join(test_params)}")
    print("预期行为:")
    print("1. 训练三种模型：SFL、HFL(随机B)、HFL(聚类B)")
    print("2. 每个epoch显示三种模型的性能")
    print("3. 保存结果到CSV文件")
    print("4. 生成对比图表")
    print("5. 显示最终性能对比")
    print("\n开始测试...\n")
    
    start_time = time.time()
    
    try:
        # 在当前目录运行测试
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
        result = subprocess.run(test_params, 
                              capture_output=True, 
                              text=True, 
                              timeout=600)  # 10分钟超时
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n测试完成！耗时: {duration:.1f}秒")
        
        if result.returncode == 0:
            print("✅ 测试成功！")
            print("\n输出摘要:")
            # 提取关键输出信息
            lines = result.stdout.split('\n')
            for line in lines:
                if any(keyword in line for keyword in ['Final', 'HFL_Random', 'HFL_Cluster', 'SFL', '结果已保存', '可视化']):
                    print(f"  {line}")
        else:
            print("❌ 测试失败！")
            print("错误信息:")
            print(result.stderr)
            
        # 检查输出文件
        check_output_files()
            
    except subprocess.TimeoutExpired:
        print("❌ 测试超时（超过10分钟）")
    except Exception as e:
        print(f"❌ 测试出错: {e}")

def check_output_files():
    """检查输出文件是否生成"""
    print("\n检查输出文件:")
    
    # 检查results目录
    results_dir = "./results"
    if os.path.exists(results_dir):
        csv_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
        if csv_files:
            latest_csv = sorted(csv_files)[-1]
            print(f"✅ CSV结果文件: {latest_csv}")
        else:
            print("❌ 未找到CSV结果文件")
    else:
        print("❌ results目录不存在")
    
    # 检查save目录  
    save_dir = "./save"
    if os.path.exists(save_dir):
        png_files = [f for f in os.listdir(save_dir) if f.endswith('.png')]
        if png_files:
            print(f"✅ 生成了 {len(png_files)} 个图表文件")
        else:
            print("❌ 未找到图表文件")
    else:
        print("❌ save目录不存在")

def check_requirements():
    """检查运行环境"""
    print("检查运行环境:")
    
    required_modules = ['torch', 'torchvision', 'matplotlib', 'pandas', 'numpy', 'sklearn', 'tqdm']
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module} - 请安装此模块")
            return False
    
    print("✅ 所有依赖模块都已安装\n")
    return True

if __name__ == "__main__":
    print("开始环境检查...")
    
    if not check_requirements():
        print("❌ 环境检查失败，请安装缺失的依赖模块")
        sys.exit(1)
    
    run_quick_test()
    
    print("\n" + "="*60)
    print("测试完成！")
    print("如果测试成功，您可以使用完整参数运行实际实验：")
    print("python main_fed.py --epochs 10 --num_users 50 --local_ep 20 --dataset mnist --model cnn --beta 0.1")
    print("="*60)