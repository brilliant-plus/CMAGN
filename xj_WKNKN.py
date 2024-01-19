"""xujing"""
import numpy as np
import pandas as pd
from sim_process import jac_similarity, cos_similarity, GIP_similarity
import time
startTime = time.time()


class WKNKN(object):

    def __init__(self, index_similarity, column_similarity, adjacency_matrix, weight=1):
        self.index_similarity = index_similarity
        self.column_similarity = column_similarity
        self.adjacency_matrix = adjacency_matrix
        self.weight = weight

    def horizontal(self):
        all_scores = []
        for i in range(self.index_similarity.shape[0]):
            W = []
            sort_similarity = (-self.index_similarity[i]).argsort()
            for j in range(len(sort_similarity)):
                W.append((self.weight ** j) * self.index_similarity[i, sort_similarity[j]])
            Z_d = 1 / np.sum(self.index_similarity[i, sort_similarity])
            score = Z_d * np.dot(W, self.adjacency_matrix[sort_similarity, :])
            all_scores.append(list(score))

        return pd.DataFrame(all_scores).values

    def vertical(self):
        all_scores = []
        for i in range(self.column_similarity.shape[0]):
            W = []
            sort_similarity = (-self.column_similarity[i]).argsort()
            for j in range(len(sort_similarity)):
                W.append((self.weight ** j) * self.column_similarity[i, sort_similarity[j]])
            Z_d = 1 / np.sum(self.column_similarity[i, sort_similarity])
            score = Z_d * np.dot(self.adjacency_matrix[:, sort_similarity], np.array(W).T)
            all_scores.append(list(score))

        return pd.DataFrame(all_scores).values.T

    def get_scores(self):
        sum_matrix = (self.horizontal() + self.vertical()) / 2
        return np.where((sum_matrix + self.adjacency_matrix) > 1, 1, sum_matrix)


# m_embeddings = pd.read_csv("nodeEmb/from_seq_net/fea_AS_node2vec/new_m_embeddings_mth0.1-cth0.1-epochs200-lambda1_fold1.csv",
#     header=None).to_numpy()
# c_embeddings = pd.read_csv("nodeEmb/from_seq_net/fea_AS_node2vec/new_c_embeddings_mth0.1-cth0.1-epochs200-lambda1_fold1.csv",
#     header=None).to_numpy()
# m_jaccard_similarity, c_jaccard_similarity = jaccard_similarity(m_embeddings), jaccard_similarity(c_embeddings)
# AS = pd.read_csv(f"newAS/new_AS_fold_weightedIS1.csv", header=None, dtype=int).to_numpy()
# print(m_jaccard_similarity, c_jaccard_similarity, AS)
#
# wk = WKNKN(m_jaccard_similarity, c_jaccard_similarity, AS, 1)
# print(wk.get_scores())
# print(type(wk.get_scores()))




# data = pd.DataFrame([[0, 1, 0, 1, 0, 0, 1, 0],
#                      [1, 0, 0, 1, 1, 0, 0, 1],
#                      [1, 1, 0, 0, 1, 1, 0, 0],
#                      [0, 0, 1, 0, 0, 1, 1, 0],
#                      [0, 1, 1, 1, 0, 0, 0, 1]])
# index_similarity = pd.DataFrame([[1, 0.28, 0.28, 0.33, 0.57],
#                                [0.28, 1, 0.5, 0, 0.5],
#                                [0.28, 0.5, 1, 0.28, 0.25],
#                                [0.33, 0, 0.28, 1, 0.28],
#                                [0.57, 0.5, 0.25, 0.28, 1]])
#
# column_similarity = pd.DataFrame([[1, 0.41, 0, 0.41, 1, 0.5, 0, 0.5],
#                                 [0.41, 1, 0.41, 0.67, 0.41, 0.41, 0.4, 0.4],
#                                 [0, 0.41, 1, 0.41, 0, 0.5, 0.5, 0.5],
#                                 [0.41, 0.67, 0.41, 1, 0.41, 0, 0.41, 0.82],
#                                 [1, 0.41, 0, 0.41, 1, 0.5, 0, 0.5],
#                                 [0.5, 0.41, 0.5, 0, 0.5, 1, 0.5, 0],
#                                 [0, 0.41, 0.5, 0.41, 0, 0.5, 1, 0],
#                                 [0.5, 0.41, 0.5, 0.82, 0.5, 0, 0, 1]])
#
# wk = WKNKN(index_similarity.values, column_similarity.values, data.values, 1)
# print(wk.get_scores())
# print(type(wk.get_scores()))
#
#
#
# endTime = time.time()
# print(endTime - startTime)
