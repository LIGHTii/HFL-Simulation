#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


# from .shamir import use_shamir  # 如果不需要，应该删除这行


def FedAvg(w, weights=None):
    """
    联邦平均算法
    
    Args:
        w: 模型权重列表
        weights: 权重列表（数据量），如果为None则使用简单平均
    
    Returns:
        聚合后的模型权重
    """
    # 过滤掉None值
    valid_indices = [i for i, model in enumerate(w) if model is not None]
    valid_w = [w[i] for i in valid_indices]

    # 如果没有有效模型，返回None
    if not valid_w:
        return None

    # 如果没有提供权重，使用简单平均
    if weights is None:
        w_avg = copy.deepcopy(valid_w[0])
        num = len(valid_w)

        for k in w_avg.keys():
            for i in range(1, num):
                w_avg[k] += valid_w[i][k]
            w_avg[k] = torch.div(w_avg[k], num)

        return w_avg
    
    # 使用加权平均
    valid_weights = [weights[i] for i in valid_indices]
    total_weight = sum(valid_weights)
    
    # 如果总权重为0，回退到简单平均
    if total_weight == 0:
        return FedAvg(w, weights=None)
    
    w_avg = copy.deepcopy(valid_w[0])
    
    # 初始化为0
    for k in w_avg.keys():
        w_avg[k] = torch.zeros_like(w_avg[k])
    
    # 加权累加
    for i, model in enumerate(valid_w):
        weight_ratio = valid_weights[i] / total_weight
        for k in w_avg.keys():
            w_avg[k] += model[k] * weight_ratio

    return w_avg


def FedAvg_layered(w, C, weights=None):
    """
    分层联邦平均算法
    
    Args:
        w: 模型权重列表
        C: 分组字典 {group_id: [client_indices]}
        weights: 权重字典 {client_id: weight}，如果为None则使用简单平均
    
    Returns:
        分组聚合后的模型权重列表
    """
    num_groups = len(C)
    grouped_w_avg = [None] * num_groups

    # 遍历C中的每一个分组
    for group_id, client_indices in C.items():
        # 如果分组为空，则跳过
        if not client_indices:
            continue

        # 获取当前分组的所有模型参数，过滤掉None值
        valid_indices = [i for i in client_indices if w[i] is not None]
        group_models = [w[i] for i in valid_indices]

        # 如果没有有效模型，跳过这个分组
        if not group_models:
            continue

        # 如果没有提供权重，使用简单平均
        if weights is None:
            w_avg = copy.deepcopy(group_models[0])
            num = len(group_models)

            for k in w_avg.keys():
                for i in range(1, num):
                    w_avg[k] += group_models[i][k]
                w_avg[k] = torch.div(w_avg[k], num)
        else:
            # 使用加权平均
            group_weights = [weights.get(i, 0) for i in valid_indices]
            total_weight = sum(group_weights)
            
            # 如果总权重为0，回退到简单平均
            if total_weight == 0:
                w_avg = copy.deepcopy(group_models[0])
                num = len(group_models)
                for k in w_avg.keys():
                    for i in range(1, num):
                        w_avg[k] += group_models[i][k]
                    w_avg[k] = torch.div(w_avg[k], num)
            else:
                w_avg = copy.deepcopy(group_models[0])
                
                # 初始化为0
                for k in w_avg.keys():
                    w_avg[k] = torch.zeros_like(w_avg[k])
                
                # 加权累加
                for i, model in enumerate(group_models):
                    weight_ratio = group_weights[i] / total_weight
                    for k in w_avg.keys():
                        w_avg[k] += model[k] * weight_ratio

        # 将计算出的平均模型存入列表的对应位置
        if group_id < num_groups:
            grouped_w_avg[group_id] = w_avg
        else:
            # 如果group_id超出现有列表长度，扩展列表
            while len(grouped_w_avg) <= group_id:
                grouped_w_avg.append(None)
            grouped_w_avg[group_id] = w_avg

    return grouped_w_avg

