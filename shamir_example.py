import random
import torch
import numpy as np
from functools import reduce

def generate_unique_random_numbers(n):
    unique_integers = random.sample(range(1, n+10), n)

    return unique_integers


# 生成秘密的份额
def shamir_secret_sharing(w_secrets, w_shamirs, x_s):
    num_shares = len(x_s)
    threshold = len(x_s)
    for i in range(len(w_secrets)):
        w_secret=w_secrets[i]
        ##coefficients = [0] + [np.random.randint(1, 20) for _ in range(threshold - 1)]
        for key in range(len(w_secret)):
            tensor_secret = w_secret[key]
            for z in range(len(tensor_secret)):
                secret=tensor_secret[z]  ##遍历每一个秘密
                coefficients = [secret] + [np.random.uniform(0, 10) for _ in range(threshold - 1)]
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
        for key in range(len(w_shamir)):
            tensor_shamir = w_shamir[key]
            for z in range(len(tensor_shamir)):  # 每个客户端的拉格朗日插值相同
                y = tensor_shamir[z]
                tensor_encrypted = w_encrypted[key]
                tensor_encrypted[z] += y * num * denom_inv


if __name__ == '__main__':
    w_locals=[[[1.1,1.2],[2.1,2.2],[3.1,3.2],[4.1,4.2],[5.1,5.2]],
              [[2.1,2.2],[3.1,3.2],[4.1,4.2],[5.1,5.2],[6.1,6.2]],
              [[3.1,3.2],[4.1,4.2],[5.1,5.2],[6.1,6.2],[7.1,7.2]]]
    num_users = len(w_locals)
    x_s = generate_unique_random_numbers(num_users)
    w_shamirs=[[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]
    w_encrypted=[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]

    shamir_secret_sharing(w_locals, w_shamirs, x_s)
    reconstruct_secret(w_shamirs, w_encrypted, x_s)
    print(w_encrypted)
