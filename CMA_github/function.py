# -*- coding: utf-8 -*-
import numpy as np


def get_ColRow_AS(new_association):
    return new_association, new_association.T


def get_edgelist(association):
    """
    为当作GAT节点特征的AS进行node2vec提取特征，准备关联网络的edgelist形式
    @param association:
    @return:
    """
    pos = []
    for index in range(association.shape[0]):
        for col in range(association.shape[1]):
            if association[index, col] != 0:
                pos.append([index + 1, col + 821 + 1, association[index, col]])  # 设置了权值，注意节点ID从1开始
    pos_len = len(pos)  # 正样本数量
    pos = np.array(pos)  # 变成矩阵形式

    return pos