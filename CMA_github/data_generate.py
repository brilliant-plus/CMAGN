# -*- coding: utf-8 -*-

import random
import numpy as np

def get_all_samples(conjunction, random_seed=None):
    pos = []
    neg = []
    for index in range(conjunction.shape[0]):
        for col in range(conjunction.shape[1]):
            if conjunction[index, col] == 0:
                neg.append([index, col, 0])
            else:
                pos.append([index, col, 1])  # 非0值变成1，看做有关联，否则成多标签了
                # np.where(conjunction[index, col] != 0, 1, conjunction[index, col])
    if random_seed is not None:
        random.seed(random_seed)  # 有随机操作，已设置种子
    pos_len = len(pos)
    new_neg = random.sample(neg, pos_len)
    samples = pos + new_neg
    samples = random.sample(samples, len(samples))  # 打乱正负样本顺序
    # print("samples", samples)
    samples = np.array(samples)  # 这一步转换格式十分必要
    # print("samples", samples)
    # samples_path = f"samples/Samples.csv"
    # np.savetxt(samples_path, samples, fmt='%d, %d, %.8f', delimiter=',')
    return samples


# # TEST
# import pandas as pd
# association = pd.read_csv("data/matrix_AS_score.csv", header=None).to_numpy()
# seed = 12  # 选择一个种子值
#
# # 第一次执行
# samples = get_all_samples(association, seed)
# print(samples)
#
# # 第二次执行
# samples = get_all_samples(association, seed)
# print(samples)