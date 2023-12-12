"""徐京"""
import pandas as pd
from sim_process import jac_similarity, cos_similarity, GIP_similarity
import numpy as np

class NCP(object):
    def __init__(self, drug_similarity, atc_similarity, adjacency_matrix):
        self.drug_similarity = drug_similarity
        self.atc_similarity = atc_similarity
        self.adjacency_matrix = adjacency_matrix

    # def ATCSP(self):
    #     eps = 1e-8
    #     temp_matrix = np.dot(self.adjacency_matrix, self.atc_similarity)
    #     modulus = np.linalg.norm(self.adjacency_matrix, axis=1).reshape(-1, 1) + eps
    #     # print(modulus)
    #     return temp_matrix / modulus
    def ATCSP(self):
        temp_matrix = np.dot(self.adjacency_matrix, self.atc_similarity)
        modulus = np.linalg.norm(self.adjacency_matrix, axis=1).reshape(-1, 1)
        # print(modulus)
        return temp_matrix / modulus

    # def DrugSP(self):
    #     eps = 1e-8
    #     temp_matrix = np.dot(self.drug_similarity, self.adjacency_matrix)
    #     modulus = np.linalg.norm(self.adjacency_matrix, axis=0).reshape(1, -1) + eps
    #     # print(modulus)
    #     return temp_matrix / modulus
    def DrugSP(self):
        temp_matrix = np.dot(self.drug_similarity, self.adjacency_matrix)
        modulus = np.linalg.norm(self.adjacency_matrix, axis=0).reshape(1, -1)
        # print(modulus)
        return temp_matrix / modulus


    def calculate_modulus_sum(self):
        index_modulus = np.linalg.norm(self.drug_similarity, axis=1).reshape(-1, 1)
        columns_modulus = np.linalg.norm(self.atc_similarity, axis=0).reshape(1, -1)
        return index_modulus + columns_modulus

    def network_NCP(self):
        # return (self.ATCSP() + self.DrugSP()) / self.calculate_modulus_sum()  # 使用eps
        result = np.nan_to_num((np.nan_to_num(self.ATCSP())+np.nan_to_num(self.DrugSP()))/self.calculate_modulus_sum())
        return result


def katz_similarity(SC, SM, AS, k=2, beta=0.5):
    """
    计算Katz相似性矩阵

    参数:
        SC (ndarray): circRNA相似性矩阵
        SM (ndarray): miRNA相似性矩阵
        AS (ndarray): circRNA-miRNA关联矩阵
        k (int): 路径长度参数
        beta (float): 衰减因子

    返回:
        S (ndarray): Katz相似性矩阵
    """

    # 构建异构网络的Katz相似性矩阵
    A = np.block([
        [SC, AS],
        [AS.T, SM]
    ])

    S = np.zeros_like(A)  # 创建一个与A相同大小的零矩阵

    for i in range(1, k+1):
        S += beta**i * np.linalg.matrix_power(A, i)  # 累加Katz相似性的后续项

    return S

# # TEST
# # get sim matrix and association
# seqSimMatrix_circRNA = pd.read_csv("data/seqSim_circRNA.csv", header=None, dtype=np.float32).to_numpy()
# seqSimMatrix_miRNA = pd.read_csv("data/seqSim_miRNA.csv", header=None, dtype=np.float32).to_numpy()
# association = pd.read_csv("data/matrix_AS.csv", header=None).to_numpy()
#
# result = NCP(seqSimMatrix_miRNA, seqSimMatrix_circRNA, association).network_NCP()
# print(result)

