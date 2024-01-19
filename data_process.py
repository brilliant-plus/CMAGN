import pandas as pd
import numpy as np
import networkx as nx
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
import argparse

"""use in main.py"""
def single_generate_graph_adj_and_feature(network, feature):
    features = sp.csr_matrix(feature).tolil().todense()

    graph = nx.from_numpy_matrix(network)
    adj = nx.adjacency_matrix(graph)
    adj_COO = sp.coo_matrix(adj)

    return adj_COO, features


"""use in get_gate_feature.py"""
def prepare_graph_data(adj):
    # adapted from preprocess_adj_bias
    num_nodes = adj.shape[0]
    adj = adj + sp.eye(num_nodes)  # self-loop
    # print(adj, "-=================================")
    data = adj.tocoo().data
    # print(data, "+++++++++++++++++++++++++++++++++")
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = adj.astype(np.float32)
    indices = np.vstack((adj.col, adj.row)).transpose()
    return (indices, adj.data, adj.shape), adj.row, adj.col

def parse_args(epochs,l):
    '''
    Parses the arguments.
    '''
    parser = argparse.ArgumentParser(description="Run gate.")

    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate. Default is 0.001.')

    parser.add_argument('--n-epochs', default=epochs, type=int,
                        help='Number of epochs')

    parser.add_argument('--hidden-dims', type=list, nargs='+', default=[128,64],
                        help='Number of dimensions.')

    parser.add_argument('--lambda-', default=l, type=float,
                        help='Parameter controlling the contribution of graph structure reconstruction in the loss function.')

    parser.add_argument('--dropout', default=0.5, type=float,
                        help='Dropout.')

    parser.add_argument('--gradient_clipping', default=5.0, type=float,
                        help='gradient clipping')

    return parser.parse_args()


"""use in gate_trainer.py"""
def conver_sparse_tf2np(input):
    # Convert Tensorflow sparse matrix to Numpy sparse matrix
    return [sp.coo_matrix((input[layer][1], (input[layer][0][:, 0], input[layer][0][:, 1])),
                          shape=(input[layer][2][0], input[layer][2][1])) for layer in input]


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