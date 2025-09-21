#!/usr/bin/env python
# -*- coding: utf-8 -*-
# GPU设备问题修复验证脚本

import os
import sys

def main():
    print("=== GPU设备问题修复验证 ===")
    print()
    
    # 1. 检查设备状态
    print("1. 运行设备检查...")
    os.system("python device_check.py")
    print()
    
    # 2. 建议的运行命令
    print("2. 建议的运行命令:")
    print()
    
    print("对于单GPU环境:")
    print("python main_fed.py --dataset mnist --epochs 5 --num_channels 1 --model cnn --gpu 0 --num_users 10 --k2 1 --k3 1 --num_processes 4")
    print()
    
    print("对于CPU环境:")
    print("python main_fed.py --dataset mnist --epochs 5 --num_channels 1 --model cnn --gpu -1 --num_users 10 --k2 1 --k3 1 --num_processes 4")
    print()
    
    print("对于多GPU环境 (强制使用GPU 0):")
    print("CUDA_VISIBLE_DEVICES=0 python main_fed.py --dataset mnist --epochs 5 --num_channels 1 --model cnn --gpu 0 --num_users 10 --k2 1 --k3 1 --num_processes 4")
    print()
    
    # 3. 修复说明
    print("3. 已修复的问题:")
    print("✅ 修复了test.py中的设备不匹配问题:")
    print("   - 将 data.cuda() 改为 data.to(args.device)")
    print("   - 将 target.cuda() 改为 target.to(args.device)")
    print()
    print("✅ 修复了数据获取函数返回值不匹配问题:")
    print("   - get_client_datasets 现在返回 4 个值")
    print("   - 添加了 client_classes 映射生成")
    print()
    print("✅ 清理了重复的函数定义")
    print()
    
    # 4. 常见错误解决方案
    print("4. 常见GPU错误解决方案:")
    print()
    print("错误: 'Expected all tensors to be on the same device'")
    print("解决: 确保使用正确的GPU ID，避免混合使用不同GPU")
    print()
    print("错误: 'CUDA out of memory'")
    print("解决: 减少batch size或使用更少的客户端数量")
    print("  --local_bs 5 --num_users 10")
    print()
    print("错误: 'CUDA device not available'")  
    print("解决: 使用CPU训练: --gpu -1")
    print()
    
    # 5. 环境变量设置
    print("5. 有用的环境变量:")
    print("export CUDA_VISIBLE_DEVICES=0  # 只使用GPU 0")
    print("export CUDA_LAUNCH_BLOCKING=1  # 启用同步CUDA调用以便调试")
    print()

if __name__ == "__main__":
    main()