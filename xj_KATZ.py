# -*- coding: utf-8 -*-
"""xujing"""
import numpy as np
from numpy.linalg import linalg


class KATZ(object):
    def __init__(self, index_similarity, column_similarity, adjacency_matrix):
        self.index_similarity = index_similarity
        self.column_similarity = column_similarity
        self.adjacency_matrix = adjacency_matrix
        self.index_num = self.index_similarity.shape[0]
        self.column_num = self.column_similarity.shape[0]
        self.first_matrix = self.matrix_A_start()

    def zeros_matrix(self):
        num = self.index_num + self.column_num
        return np.zeros((num, num))

    def matrix_A_start(self):
        matrix = self.zeros_matrix()
        matrix[:self.index_num, :self.index_num] = self.index_similarity
        matrix[:self.index_num, self.index_num:] = self.adjacency_matrix

        matrix[self.index_num:, :self.index_num] = self.adjacency_matrix.T
        matrix[self.index_num:, self.index_num:] = self.column_similarity

        return matrix

    def split(self, matrix_s):
        return matrix_s[:self.index_num, self.index_num:]

    def matrix_power(self, matrix, k):
        return linalg.matrix_power(matrix, k)

    def predict(self, k, beta):
        if k == 1:
            return beta*self.matrix_power(self.first_matrix, 1)
        else:
            return (beta**k)*self.matrix_power(self.first_matrix, k) + self.predict(k-1, beta)

    def get_matrix(self, k=2, beta=0.5):  # 终极奥  # 此处相较源代码增加了默认参数 k=2, beta=0.5
        matrix_s = self.predict(k, beta)
        return self.split(matrix_s)

# 自定义katz方法
def katz_similarity(SC, SD, AS, k=2, beta=0.5):
    """
    计算Katz相似性矩阵

    参数:
        SC (ndarray): circRNA相似性矩阵
        SD (ndarray): disease相似性矩阵
        AS (ndarray): circRNA-disease关联矩阵
        k (int): 路径长度参数
        beta (float): 衰减因子

    返回:
        S (ndarray): Katz相似性矩阵
    """

    # 构建异构网络的Katz相似性矩阵
    A = np.block([
        [SC, AS],
        [AS.T, SD]
    ])

    S = np.zeros_like(A)  # 创建一个与A相同大小的零矩阵

    for i in range(1, k+1):
        S += beta**i * np.linalg.matrix_power(A, i)  # 累加Katz相似性的后续项

    return S
# # 处理AS，用自定义katz，使用方式与徐京不同，如下
# jac_AS = katz_similarity(m_jaccard_similarity, c_jaccard_similarity, AS)[:821, 821:]
# GIP_AS = katz_similarity(m_GIP_similarity, c_GIP_similarity, AS)[:821, 821:]
# cos_AS = katz_similarity(m_cos_similarity, c_cos_similarity, AS)[:821, 821:]



