#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 设备一致性检查和修复工具

import torch

def check_device_consistency():
    """检查CUDA设备状态"""
    print("=== GPU设备检查 ===")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA设备数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"当前CUDA设备: {torch.cuda.current_device()}")
    print()

def ensure_model_device(model, device):
    """确保模型在指定设备上"""
    model = model.to(device)
    print(f"模型已移动到设备: {device}")
    return model

def test_tensor_operations(device):
    """测试tensor操作"""
    print(f"=== 测试设备 {device} 上的tensor操作 ===")
    try:
        # 创建测试tensor
        x = torch.randn(2, 3).to(device)
        y = torch.randn(3, 4).to(device)
        
        # 执行操作
        z = torch.mm(x, y)
        print(f"tensor操作成功: {z.shape}")
        print(f"结果tensor设备: {z.device}")
        return True
    except Exception as e:
        print(f"tensor操作失败: {e}")
        return False

if __name__ == "__main__":
    check_device_consistency()
    
    # 测试不同设备
    devices_to_test = ['cpu']
    if torch.cuda.is_available():
        devices_to_test.append('cuda:0')
        if torch.cuda.device_count() > 1:
            devices_to_test.append('cuda:1')
    
    for device in devices_to_test:
        device_obj = torch.device(device)
        success = test_tensor_operations(device_obj)
        if not success:
            print(f"警告: 设备 {device} 可能有问题")
        print()
    
    print("=== 建议 ===")
    print("如果遇到设备不匹配错误，请:")
    print("1. 确保所有tensor和模型都在同一设备上")
    print("2. 使用 tensor.to(device) 或 model.to(device)")
    print("3. 避免在不同GPU之间混合使用tensor")
    print("4. 如果只有一个GPU，使用 --gpu 0")
    print("5. 如果使用CPU，使用 --gpu -1")