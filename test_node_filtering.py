#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试地理节点筛选功能的脚本

使用方法：
1. 不启用筛选：
   python test_node_filtering.py

2. 启用筛选，使用默认参数（半径100km，中心为图质心）：
   python test_node_filtering.py --enable_node_filter

3. 启用筛选，自定义半径：
   python test_node_filtering.py --enable_node_filter --filter_radius 50

4. 启用筛选，自定义中心点和半径：
   python test_node_filtering.py --enable_node_filter --filter_center_lat 39.9042 --filter_center_lon 116.4074 --filter_radius 200

5. 对比筛选前后的效果：
   python test_node_filtering.py --enable_node_filter --filter_radius 80
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.bipartite_bandwidth import run_bandwidth_allocation
from utils.options import args_parser
import argparse

def main():
    # 解析命令行参数
    args = args_parser()
    
    print("="*60)
    print("地理节点筛选功能测试")
    print("="*60)
    
    # 打印当前配置
    print(f"GraphML文件: {args.graphml_file}")
    print(f"边缘服务器比例: {args.es_ratio}")
    print(f"最大容量: {args.max_capacity}")
    
    if args.enable_node_filter:
        print(f"✅ 地理节点筛选: 已启用")
        print(f"   筛选半径: {args.filter_radius} km")
        if args.filter_center_lat is not None and args.filter_center_lon is not None:
            print(f"   筛选中心: ({args.filter_center_lat}, {args.filter_center_lon})")
        else:
            print(f"   筛选中心: 图质心（自动计算）")
    else:
        print(f"❌ 地理节点筛选: 未启用")
    
    print("\n" + "="*60)
    print("开始运行带宽分配算法...")
    print("="*60)
    
    # 运行带宽分配算法
    try:
        result = run_bandwidth_allocation(
            graphml_file=args.graphml_file,
            es_ratio=args.es_ratio,
            max_capacity=args.max_capacity,
            visualize=True
        )
        
        if result and result[0] is not None:
            bipartite_graph, client_nodes, active_es_nodes, association_matrix, r_client_to_es, r_es, r_es_to_cloud, r_client_to_cloud = result
            
            print("\n" + "="*60)
            print("运行结果摘要:")
            print("="*60)
            print(f"✅ 成功构建网络拓扑")
            print(f"   客户端节点数: {len(client_nodes)}")
            print(f"   活跃边缘服务器数: {len(active_es_nodes)}")
            print(f"   关联矩阵形状: {association_matrix.shape}")
            
            if r_client_to_es is not None:
                print(f"   客户端到ES传输速率矩阵形状: {r_client_to_es.shape}")
                mean_rate = r_client_to_es.mean()
                print(f"   平均传输速率: {mean_rate:.2e} bit/s ({mean_rate/1e6:.2f} Mbps)")
            
            if r_es is not None:
                print(f"   ES间传输速率矩阵形状: {r_es.shape}")
            
            print(f"\n📊 可视化图已保存到 ./save/ 目录")
            
        else:
            print("❌ 算法运行失败")
            
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()