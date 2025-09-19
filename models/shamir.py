import random
import torch
import numpy as np
from functools import reduce

def generate_unique_random_numbers(n):
    unique_integers = random.sample(range(1, n+10), n)
    unique_floats = [num / 100 for num in unique_integers]

    return unique_floats


# 生成秘密的份额
def shamir_secret_sharing(w_secrets, w_shamirs, x_s):
    num_shares = len(x_s)
    threshold = len(x_s)
    for i in range(len(w_secrets)):
        w_secret=w_secrets[i]
        for key in w_secret.keys():
            tensor_secret = w_secret[key]
            for z in range(tensor_secret.size(0)):
                secret=tensor_secret[z]  ##遍历每一个秘密
                coefficients = [secret] + [np.random.uniform(0, 0.001) for _ in range(threshold - 1)]
                for j in range(len(x_s)):  ##生成第i个客户端的秘密碎片
                    y = sum(coeff * (x_s[j] ** n) for n, coeff in enumerate(coefficients))
                    w_shamir = w_shamirs[j]   ##传输给对应的第j个客户端
                    tensor_shamir = w_shamir[key]
                    tensor_shamir[z] =(tensor_shamir[z] + y)   ##将各个秘密碎片加


# 从给定的份额重建秘密
def reconstruct_secret(w_shamirs, w_encrypted, x_s):
    for i in range(len(x_s)):
        xi=x_s[i]
        num = 1
        denom = 1
        for j in range(len(x_s)):
            if i != j:
                num *= -x_s[j]
                denom *= (x_s[i] - x_s[j])
        if denom == 0:  # 确保不会出现零除情况
            continue
        denom_inv = 1 / denom  # 不再进行模逆运算
        w_shamir = w_shamirs[i]
        for key in w_shamir.keys():
            tensor_shamir = w_shamir[key]
            for z in range(tensor_shamir.size(0)):  # 每个客户端的拉格朗日插值相同
                y = tensor_shamir[z]
                tensor_encrypted = w_encrypted[key]
                tensor_encrypted[z] += y * num * denom_inv

# 使用 Shamir 的秘密共享方案
def use_shamir(w_locals):
    num_users = len(w_locals)
    x_s = generate_unique_random_numbers(num_users)
    example_w = w_locals[0]

    w_shamirs = [{} for _ in range(len(w_locals))]  # 初始化 w_shamirs 列表

    for key in example_w.keys():
        zero_tensor = torch.zeros_like(example_w[key])  # 创建与原张量相同形状的零张量
        for idx in range(len(w_shamirs)):
            w_shamirs[idx][key] = zero_tensor.clone()  # 将全零张量赋值到每个字典中

    w_encrypted = {key: torch.zeros_like(value) for key, value in w_locals[0].items()}  ##存储每个服务器收到的秘密碎片之和,即服务器收到的加密过的参数
    shamir_secret_sharing(w_locals, w_shamirs, x_s)
    reconstruct_secret(w_shamirs, w_encrypted, x_s)
    return w_encrypted

