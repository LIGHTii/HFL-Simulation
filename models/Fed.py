#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn

from .shamir import use_shamir


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    # w_avg_shamir=use_shamir(w)
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
        """
        w_avg_shamir[k] = torch.div(w_avg_shamir[k], len(w))
    for key, value in w_avg.items():
        print(f"Aggregated {key}: {value}")
    print("————————————————————————\n")
    for key, value in w_avg_shamir.items():
        print(f"Aggregated {key}: {value}")
    """
    return w_avg

def FedAvg_layered(w,C):
    num_groups = len(C)
    grouped_w_avg = [None] * num_groups

    # 遍历C中的每一个分组
    for group_id, client_indices in C.items():
        # 如果分组为空，则跳过
        if not client_indices:
            continue

        # 获取当前分组的所有模型参数
        group_models = [w[i] for i in client_indices]

        # 深度拷贝第一个模型的参数作为初始值
        w_avg = copy.deepcopy(group_models[0])

        # 遍历模型参数的每一层
        for k in w_avg.keys():
            # 从第二个模型开始，累加所有参数
            for i in range(1, len(group_models)):
                w_avg[k] += group_models[i][k]

            # 除以该组客户端的数量，求得平均值
            w_avg[k] = torch.div(w_avg[k], len(group_models))

        # 将计算出的平均模型存入列表的对应位置
        if group_id < num_groups:
            grouped_w_avg[group_id] = w_avg
        else:
            # 如果group_id超出现有列表长度，这是一个意外情况
            # 可以在这里添加错误处理或扩展列表
            print(f"警告: 分组编号 {group_id} 超出预期范围，结果未被存储。")

    return grouped_w_avg

