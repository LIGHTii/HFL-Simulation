#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试新的节点分布功能
"""

from utils.bipartite_bandwidth import run_bandwidth_allocation
import argparse

def test_node_distribution():
    """测试不同边缘服务器占比的节点分布"""
    
    # 测试不同的边缘服务器占比
    es_ratios = [0.1, 0.15, 0.2, 0.25]
    
    for es_ratio in es_ratios:
        print(f"\n{'='*60}")
        print(f"Testing ES Ratio: {es_ratio}")
        print(f"{'='*60}")
        
        try:
            # 运行带宽分配，生成可视化
            result = run_bandwidth_allocation(
                graphml_file="graph-example/Ulaknet.graphml", 
                model_size=1e7, 
                es_ratio=es_ratio, 
                visualize=True
            )
            
            if result[0] is not None:
                bipartite_graph, client_nodes, es_nodes = result[:3]
                print(f"✅ Success! Clients: {len(client_nodes)}, Edge Servers: {len(es_nodes)}")
                print(f"   Actual ES ratio: {len(es_nodes)/(len(client_nodes)+len(es_nodes)):.3f}")
            else:
                print("❌ Failed to build bipartite graph")
                
        except Exception as e:
            print(f"❌ Error with ES ratio {es_ratio}: {str(e)}")
    
    print(f"\n{'='*60}")
    print("Testing completed! Check generated PNG files for visualizations.")
    print(f"{'='*60}")

if __name__ == '__main__':
    test_node_distribution()