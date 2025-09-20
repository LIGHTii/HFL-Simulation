#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn

from .shamir import use_shamir

def FedAvg(w):
    if not w or all(w_i is None or not w_i for w_i in w):
        print("Error: No valid models for FedAvg aggregation")
        return None
    valid_models = [w_i for w_i in w if w_i is not None and w_i]
    if not valid_models:
        print("Error: No valid models after filtering")
        return None
    w_avg = copy.deepcopy(valid_models[0])
    for k in w_avg.keys():
        w_avg[k] = torch.zeros_like(w_avg[k])
        count = 0
        for i in range(len(valid_models)):
            if valid_models[i] is not None and k in valid_models[i]:
                w_avg[k] += valid_models[i][k]
                count += 1
        if count > 0:
            w_avg[k] = torch.div(w_avg[k], count)
        else:
            print(f"Warning: No valid models for key {k}")
    return w_avg

def FedAvg_layered(w, C):
    num_groups = len(C)
    grouped_w_avg = [None] * num_groups

    for group_id, client_indices in C.items():
        if not client_indices:
            print(f"Warning: Group {group_id} has no clients, skipping")
            continue
        group_models = [w[i] for i in client_indices if i < len(w) and w[i] is not None and w[i]]
        if not group_models:
            print(f"Error: No valid models for group {group_id}")
            continue
        w_avg = copy.deepcopy(group_models[0])
        for k in w_avg.keys():
            w_avg[k] = torch.zeros_like(w_avg[k])
            count = 0
            for i in range(len(group_models)):
                if group_models[i] is not None and k in group_models[i]:
                    w_avg[k] += group_models[i][k]
                    count += 1
            if count > 0:
                w_avg[k] = torch.div(w_avg[k], count)
            else:
                print(f"Warning: No valid models for key {k} in group {group_id}")
        if group_id < num_groups:
            grouped_w_avg[group_id] = w_avg
        else:
            print(f"Warning: Group ID {group_id} exceeds expected range")
    return grouped_w_avg