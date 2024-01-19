# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity, rbf_kernel


def sim_thresholding(matrix: np.ndarray, threshold):
    """
    根据阈值划分相似性网络
    @param matrix:
    @param threshold:
    @return:
    """
    matrix_copy = matrix.copy()
    # 将对角线元素设置为0
    np.fill_diagonal(matrix_copy, 0)

    matrix_copy[matrix_copy >= threshold] = 1
    matrix_copy[matrix_copy < threshold] = 0

    print(int(np.sum(np.sum(matrix_copy))), end='. ')
    return matrix_copy

def jac_similarity(AS, axis=0):
    if axis == 1:
        AS = AS.T
    # # 关联矩阵本身就是二进制矩阵了，所以用不到转换
    # # 将数据集转换为二进制矩阵
    AS_binary = np.where(AS > 0, 1, 0)
    # 计算 Jaccard 相似性矩阵
    similarity_matrix = 1 - pairwise_distances(AS, metric='jaccard')
    return similarity_matrix

def cos_similarity(AS, axis=0):
    """

    @param AS: 传入的embeddings
    @param axis: 值为0时整行为embedding，求原AS各行之间的相似性；为1时整列为embedding（因为转至了，相当于还是整行），求原AS各列之间的相似性
    @return:
    """
    if axis == 1:
        AS = AS.T
    similarity_matrix = cosine_similarity(AS)
    # 保存对角线元素
    diag_elements = np.diag(similarity_matrix)

    # 将对角线元素设置为1
    np.fill_diagonal(similarity_matrix, 1)

    # 将保存的对角线元素放回到矩阵中
    np.fill_diagonal(similarity_matrix, diag_elements)

    return similarity_matrix

def getGamma(AS):
    """

    :param AS:
    :return:
    """
    # 默认的gamma值是经过AS计算得来的
    n = AS.shape[0]
    sum = 0
    for i in range(n):
        x_norm = np.square(np.linalg.norm(AS[i, :]))
        sum = sum + x_norm
    r = n / sum
    # print(r)
    return r

def GIP_similarity(AS, axis=0, gamma=None):
    """

    :param AS:
    :param axis:
    :param gamma:
    :return:
    """
    if axis == 1:
        AS = AS.T
    # 计算高斯相似性矩阵
    # 使用默认gamma值，调用rbf_kernel函数时，你需要传递一个具体的gamma值，而不是一个函数,要获得具体的gamma值
    if gamma is None:
        gamma = getGamma(AS)
    similarity_matrix = rbf_kernel(AS, gamma=gamma)
    return similarity_matrix

def get_seqGIP_similarity(AS, axis, seq_sim):
    seqGIP_sim = np.zeros((AS.shape[axis], AS.shape[axis]))

    GIP_sim = GIP_similarity(AS, axis=axis)

    for i in range(AS.shape[axis]):
        for j in range(AS.shape[axis]):
                seqGIP_sim[i, j] = (seq_sim[i, j] + GIP_sim[i, j])/2

    return seqGIP_sim

def get_seqCOS_similarity(AS, axis, seq_sim):
    seqCOS_sim = np.zeros((AS.shape[axis], AS.shape[axis]))

    cos_sim = cos_similarity(AS, axis=axis)

    for i in range(AS.shape[axis]):
        for j in range(AS.shape[axis]):
            if cos_sim[i, j] == 0:
                seqCOS_sim[i, j] =seq_sim[i, j]
            else:
                seqCOS_sim[i, j] = (seq_sim[i, j] + cos_sim[i, j]) / 2

    return seqCOS_sim