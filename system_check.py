#!/usr/bin/env python3
"""
系统环境检查工具
用于验证三模型联邦学习系统的环境是否正确配置
"""

def check_basic_imports():
    """检查基本库导入"""
    try:
        import sys
        print("✅ Python版本:", sys.version)
        return True
    except Exception as e:
        print("❌ Python基础模块导入失败:", e)
        return False

def check_scientific_libraries():
    """检查科学计算库"""
    results = {}
    
    # 检查numpy
    try:
        import numpy as np
        print(f"✅ NumPy版本: {np.__version__}")
        results['numpy'] = True
    except Exception as e:
        print(f"❌ NumPy导入失败: {e}")
        results['numpy'] = False
    
    # 检查pandas
    try:
        import pandas as pd
        print(f"✅ Pandas版本: {pd.__version__}")
        results['pandas'] = True
    except Exception as e:
        print(f"❌ Pandas导入失败: {e}")
        results['pandas'] = False
    
    # 检查matplotlib
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        print(f"✅ Matplotlib版本: {matplotlib.__version__}")
        results['matplotlib'] = True
    except Exception as e:
        print(f"❌ Matplotlib导入失败: {e}")
        results['matplotlib'] = False
    
    return results

def check_torch():
    """检查PyTorch"""
    try:
        import torch
        print(f"✅ PyTorch版本: {torch.__version__}")
        
        # 检查CUDA
        if torch.cuda.is_available():
            print(f"✅ CUDA可用，设备数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   设备{i}: {torch.cuda.get_device_name(i)}")
        else:
            print("⚠️  CUDA不可用，将使用CPU")
        
        return True
    except Exception as e:
        print(f"❌ PyTorch导入失败: {e}")
        return False

def check_sklearn():
    """检查scikit-learn"""
    try:
        import sklearn
        print(f"✅ Scikit-learn版本: {sklearn.__version__}")
        
        # 尝试导入spectral clustering
        from sklearn.cluster import SpectralClustering
        print("✅ SpectralClustering导入成功")
        
        return True
    except Exception as e:
        print(f"❌ Scikit-learn相关导入失败: {e}")
        return False

def check_project_modules():
    """检查项目模块"""
    import os
    import sys
    
    # 添加项目路径
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    results = {}
    
    # 检查主要模块
    modules_to_check = [
        'utils.options',
        'utils.data_partition', 
        'utils.sampling',
        'models.Nets',
        'models.Fed',
        'models.Update',
        'models.test',
        'models.cluster'
    ]
    
    for module_name in modules_to_check:
        try:
            __import__(module_name)
            print(f"✅ 模块 {module_name} 导入成功")
            results[module_name] = True
        except Exception as e:
            print(f"❌ 模块 {module_name} 导入失败: {e}")
            results[module_name] = False
    
    return results

def main():
    """主检查函数"""
    print("=" * 50)
    print("🔍 三模型联邦学习系统环境检查")
    print("=" * 50)
    
    # 基本检查
    print("\n📦 基本Python环境检查:")
    basic_ok = check_basic_imports()
    
    # 科学计算库检查
    print("\n🔬 科学计算库检查:")
    sci_results = check_scientific_libraries()
    
    # PyTorch检查
    print("\n🔥 PyTorch检查:")
    torch_ok = check_torch()
    
    # Scikit-learn检查
    print("\n🧮 Scikit-learn检查:")
    sklearn_ok = check_sklearn()
    
    # 项目模块检查
    print("\n📂 项目模块检查:")
    project_results = check_project_modules()
    
    # 总结
    print("\n" + "=" * 50)
    print("📋 检查总结:")
    print("=" * 50)
    
    all_good = True
    if not basic_ok:
        all_good = False
        print("❌ Python基础环境有问题")
    
    if not all(sci_results.values()):
        all_good = False
        print("❌ 科学计算库有问题")
    
    if not torch_ok:
        all_good = False
        print("❌ PyTorch有问题")
    
    if not sklearn_ok:
        all_good = False
        print("❌ Scikit-learn有问题")
    
    if not all(project_results.values()):
        all_good = False
        print("❌ 项目模块有问题")
    
    if all_good:
        print("🎉 所有检查通过！系统可以运行三模型联邦学习实验")
        print("\n💡 建议运行的测试命令:")
        print("python main_fed.py --dataset mnist --epochs 5 --num_channels 1 --model cnn --gpu 0 --num_users 10 --k2 1 --k3 1 --num_processes 2")
    else:
        print("⚠️  存在问题，请根据上述错误信息进行修复")
    
    return all_good

if __name__ == "__main__":
    main()