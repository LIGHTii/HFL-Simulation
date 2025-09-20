#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


# from .shamir import use_shamir  # 如果不需要，应该删除这行


def FedAvg(w):
    # 过滤掉None值
    valid_w = [model for model in w if model is not None]

    # 如果没有有效模型，返回None
    if not valid_w:
        return None

    w_avg = copy.deepcopy(valid_w[0])
    num = len(valid_w)

    for k in w_avg.keys():
        for i in range(1, num):
            w_avg[k] += valid_w[i][k]

        w_avg[k] = torch.div(w_avg[k], num)

    return w_avg


def FedAvg_layered(w, C):
    num_groups = len(C)
    grouped_w_avg = [None] * num_groups

    # 遍历C中的每一个分组
    for group_id, client_indices in C.items():
        # 如果分组为空，则跳过
        if not client_indices:
            continue

        # 获取当前分组的所有模型参数，过滤掉None值
        group_models = [w[i] for i in client_indices if w[i] is not None]

        # 如果没有有效模型，跳过这个分组
        if not group_models:
            continue

        # 深度拷贝第一个模型的参数作为初始值
        w_avg = copy.deepcopy(group_models[0])
        num = len(group_models)

        # 遍历模型参数的每一层
        for k in w_avg.keys():
            # 从第二个模型开始，累加所有参数
            for i in range(1, num):
                w_avg[k] += group_models[i][k]

            # 除以该组客户端的数量，求得平均值
            w_avg[k] = torch.div(w_avg[k], num)

        # 将计算出的平均模型存入列表的对应位置
        if group_id < num_groups:
            grouped_w_avg[group_id] = w_avg
        else:
            # 如果group_id超出现有列表长度，扩展列表
            while len(grouped_w_avg) <= group_id:
                grouped_w_avg.append(None)
            grouped_w_avg[group_id] = w_avg

    return grouped_w_avg

