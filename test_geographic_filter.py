#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试地理节点筛选功能的脚本
演示如何使用基于图范围比例的节点筛选
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.bipartite_bandwidth import run_bandwidth_allocation
from utils.options import args_parser

def test_geographic_filtering():
    """测试不同比例的地理筛选效果"""
    
    print("=" * 60)
    print("测试地理节点筛选功能")
    print("=" * 60)
    
    # 获取基本参数
    args = args_parser()
    graphml_file = "graph-example/Ulaknet.graphml"
    es_ratio = 0.34
    
    # 测试不同的筛选比例
    test_ratios = [0.1, 0.2, 0.3, 0.5, 0.8]
    
    print(f"使用图文件: {graphml_file}")
    print(f"边缘服务器比例: {es_ratio}")
    print(f"测试筛选比例: {test_ratios}")
    print()
    
    for ratio in test_ratios:
        print(f"\n{'='*50}")
        print(f"测试筛选比例: {ratio}")
        print(f"{'='*50}")
        
        # 临时修改args参数
        original_filter = getattr(args, 'enable_node_filter', False)
        original_ratio = getattr(args, 'filter_radius_ratio', 0.3)
        
        # 启用筛选并设置比例
        args.enable_node_filter = True
        args.filter_radius_ratio = ratio
        args.filter_center_lat = None
        args.filter_center_lon = None
        
        try:
            # 运行带宽分配算法
            result = run_bandwidth_allocation(
                graphml_file=graphml_file,
                es_ratio=es_ratio,
                max_capacity=0,  # 自动计算
                visualize=True   # 生成可视化图
            )
            
            if result is not None:
                bipartite_graph, client_nodes, active_es_nodes, association_matrix, r_client_to_es, r_es, r_es_to_cloud, r_client_to_cloud = result
                
                print(f"\n筛选结果统计 (比例 {ratio}):")
                print(f"  客户端节点数: {len(client_nodes)}")
                print(f"  活跃边缘服务器数: {len(active_es_nodes)}")
                print(f"  关联矩阵形状: {association_matrix.shape if association_matrix is not None else 'None'}")
                
                if r_client_to_es is not None:
                    import numpy as np
                    avg_rate = np.mean(r_client_to_es) / 1e6  # 转换为Mbps
                    print(f"  平均传输速率: {avg_rate:.2f} Mbps")
                
            else:
                print(f"筛选比例 {ratio} 运行失败")
                
        except Exception as e:
            print(f"筛选比例 {ratio} 出现错误: {e}")
        
        finally:
            # 恢复原始参数
            args.enable_node_filter = original_filter
            args.filter_radius_ratio = original_ratio
    
    print(f"\n{'='*60}")
    print("测试完成！可视化图已保存到 ./save/ 目录")
    print("可以对比不同比例下的节点分布效果")
    print(f"{'='*60}")

def test_specific_center():
    """测试指定中心点的筛选功能"""
    
    print("\n" + "=" * 60)
    print("测试指定中心点的地理筛选")
    print("=" * 60)
    
    args = args_parser()
    
    # 启用筛选并设置参数
    args.enable_node_filter = True
    args.filter_radius_ratio = 0.4
    # 设置一个特定的中心点（这里以土耳其的大致中心为例）
    args.filter_center_lat = 39.0
    args.filter_center_lon = 35.0
    
    try:
        result = run_bandwidth_allocation(
            graphml_file="graph-example/Ulaknet.graphml",
            es_ratio=0.34,
            max_capacity=0,
            visualize=True
        )
        
        if result is not None:
            bipartite_graph, client_nodes, active_es_nodes, association_matrix, r_client_to_es, r_es, r_es_to_cloud, r_client_to_cloud = result
            
            print(f"\n指定中心点筛选结果:")
            print(f"  中心点: ({args.filter_center_lat}, {args.filter_center_lon})")
            print(f"  筛选比例: {args.filter_radius_ratio}")
            print(f"  客户端节点数: {len(client_nodes)}")
            print(f"  活跃边缘服务器数: {len(active_es_nodes)}")
        else:
            print("指定中心点筛选运行失败")
            
    except Exception as e:
        print(f"指定中心点筛选出现错误: {e}")
    
    finally:
        # 重置参数
        args.enable_node_filter = False
        args.filter_center_lat = None
        args.filter_center_lon = None

if __name__ == "__main__":
    # 测试不同比例的筛选效果
    test_geographic_filtering()
    
    # 测试指定中心点的筛选
    test_specific_center()
    
    print("\n测试脚本执行完成！")