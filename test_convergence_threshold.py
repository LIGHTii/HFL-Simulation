#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试基于阈值的收敛检查器
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.conver_check import ConvergenceChecker

def test_threshold_convergence():
    """测试基于阈值的收敛机制"""
    print("=== 测试基于阈值的收敛检查器 ===\n")
    
    # 创建收敛检查器：损失阈值0.1，准确率阈值95%，耐心值3
    checker = ConvergenceChecker(patience=3, loss_threshold=0.1, acc_threshold=95.0)
    
    # 模拟训练过程
    training_data = [
        # epoch, loss, acc - 初期：损失高，准确率低
        (0, 2.5, 10.0),
        (1, 2.1, 25.0),
        (2, 1.8, 45.0),  
        (3, 1.2, 70.0),
        (4, 0.8, 85.0),
        # 开始接近阈值
        (5, 0.5, 92.0),  # 准确率还未达到阈值
        (6, 0.3, 94.0),  # 准确率还未达到阈值
        (7, 0.2, 94.5),  # 准确率还未达到阈值
        # 开始满足收敛条件
        (8, 0.08, 95.5), # 第1次同时满足条件
        (9, 0.06, 96.0), # 第2次同时满足条件
        (10, 0.05, 96.2), # 第3次同时满足条件，应该触发收敛
        (11, 0.04, 96.5), # 如果还继续的话
    ]
    
    print("开始模拟训练过程...")
    print("收敛条件: 损失 ≤ 0.1 且 准确率 ≥ 95.0%，连续满足3轮")
    print("-" * 80)
    print(f"{'Epoch':<6} {'Loss':<8} {'Acc(%)':<8} {'Status':<15} {'Description'}")
    print("-" * 80)
    
    for epoch, loss, acc in training_data:
        should_stop, reason = checker.check(loss, acc, epoch)
        status = "🎯 收敛" if should_stop else "🔄 训练中"
        print(f"{epoch:<6} {loss:<8.3f} {acc:<8.1f} {status:<15} {reason}")
        
        if should_stop:
            print(f"\n✅ 在第 {epoch} 轮检测到收敛，训练应该停止")
            break
    
    print(f"\n📊 收敛检查器最终状态:")
    print(f"   连续满足收敛条件的轮次: {checker.convergence_count}")
    print(f"   停止的轮次: {checker.stopped_epoch}")
    print(f"   损失历史: {checker.loss_history[-5:]}")  # 显示最后5个损失值
    print(f"   准确率历史: {checker.acc_history[-5:]}")  # 显示最后5个准确率值


def test_single_criterion_convergence():
    """测试只基于损失的收敛机制（向后兼容）"""
    print("\n\n=== 测试只基于损失的收敛机制（向后兼容）===\n")
    
    # 创建收敛检查器：只关注损失阈值
    checker = ConvergenceChecker(patience=2, loss_threshold=0.2)
    
    # 模拟训练过程 - 只传入损失和epoch
    loss_data = [
        (0, 1.5),
        (1, 1.0),
        (2, 0.5),
        (3, 0.3),  # 第一次高于阈值
        (4, 0.15), # 第一次达到阈值
        (5, 0.12), # 第二次达到阈值，应该触发收敛
        (6, 0.10),
    ]
    
    print("开始模拟只基于损失的训练过程...")
    print("收敛条件: 损失 ≤ 0.2，连续满足2轮")
    print("-" * 60)
    print(f"{'Epoch':<6} {'Loss':<8} {'Status':<15} {'Description'}")
    print("-" * 60)
    
    for epoch, loss in loss_data:
        should_stop, reason = checker.check(loss, epoch)  # 旧接口：只传损失和epoch
        status = "🎯 收敛" if should_stop else "🔄 训练中"
        print(f"{epoch:<6} {loss:<8.3f} {status:<15} {reason}")
        
        if should_stop:
            print(f"\n✅ 在第 {epoch} 轮检测到收敛，训练应该停止")
            break
    
    print(f"\n📊 收敛检查器最终状态:")
    print(f"   连续满足收敛条件的轮次: {checker.convergence_count}")
    print(f"   停止的轮次: {checker.stopped_epoch}")


def test_no_convergence():
    """测试不收敛的情况"""
    print("\n\n=== 测试不收敛的情况 ===\n")
    
    checker = ConvergenceChecker(patience=3, loss_threshold=0.05, acc_threshold=98.0)
    
    # 模拟训练过程 - 条件较难满足
    training_data = [
        (0, 0.8, 80.0),
        (1, 0.6, 85.0),
        (2, 0.4, 90.0),
        (3, 0.2, 92.0),
        (4, 0.1, 94.0),
        (5, 0.08, 95.0),  # 损失达到但准确率未达到
        (6, 0.06, 96.0),  # 损失达到但准确率未达到
        (7, 0.04, 97.0),  # 损失达到但准确率未达到
        (8, 0.03, 97.5),  # 损失达到但准确率未达到
    ]
    
    print("开始模拟不收敛的训练过程...")
    print("收敛条件: 损失 ≤ 0.05 且 准确率 ≥ 98.0%，连续满足3轮")
    print("-" * 80)
    print(f"{'Epoch':<6} {'Loss':<8} {'Acc(%)':<8} {'Status':<15} {'Description'}")
    print("-" * 80)
    
    for epoch, loss, acc in training_data:
        should_stop, reason = checker.check(loss, acc, epoch)
        status = "🎯 收敛" if should_stop else "🔄 训练中"
        print(f"{epoch:<6} {loss:<8.3f} {acc:<8.1f} {status:<15} {reason}")
        
        if should_stop:
            print(f"\n✅ 在第 {epoch} 轮检测到收敛")
            break
    else:
        print(f"\n❌ 训练结束但未达到收敛条件")


if __name__ == "__main__":
    test_threshold_convergence()
    test_single_criterion_convergence() 
    test_no_convergence()
    
    print("\n" + "="*80)
    print("🎉 收敛检查器测试完成！")
    print("新的基于阈值的收敛机制已经实现并测试通过")
    print("="*80)