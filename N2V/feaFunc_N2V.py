# coding=utf-8

import random
import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec

def N2V(edge_path, para_dim, random_seed):
    random.seed(random_seed)
    # 创建一个带权重的无向图
    G = nx.Graph()

    # 方法一
    # 从edge_path读取目标文件，获得节点对和权重信息
    with open(edge_path, 'r') as file:
        for line in file:
            parts = line.strip().split()  # 假设文件中的数据以空格分隔
            node1, node2, weight = parts[0], parts[1], float(parts[2])  # 假设权重为浮点数
            # 添加带权重的边到图
            G.add_edge(node1, node2, weight=weight)

    node2vec = Node2Vec(G, dimensions=para_dim, walk_length=80, num_walks=10, p=1, q=1, workers=1, seed=random_seed)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    node_emb = model.wv

    # 获取嵌入向量和节点ID
    embeddings = []
    node_ids = []
    for node in node_emb.index_to_key:
        node_ids.append(int(node))  # 明确将节点ID转换为整数类型
        embeddings.append(np.round(node_emb[node], 6))

    # 转换为 NumPy 数组
    embedding_matrix = np.column_stack((node_ids, np.array(embeddings)))
    return embedding_matrix

def emb_split(df): #, fold, m_fea_path, c_fea_path
    # 分割并排序整理emb
    # df = pd.read_csv(emb_path, skiprows=1, sep=" ", header=None).values  # 直接去掉首行标题取值变array


    # 针对如果有点孤立节点没计算到。没有相应节点的特征。补 全零向量
    sum_node = 2936
    dim = 128
    vec = np.zeros((1, dim+1))
    list = df[:, 0].tolist()
    for i in range(1, sum_node+1):
        if i not in list:
            print("第{}个节点没特征。即将补0".format(i))
            vec[0, 0] = i
            df = np.append(df, vec, axis=0)

    # 根据每行的第一个数据进行排序，并保持相同格式
    sorted_array = np.array(sorted(df, key=lambda x: x[0]))
    # # 复制排序后的数据到另一个变量 data1，以保持相同的格式
    # print(sorted_array)

    m_fea_node2vec = sorted_array[0:821, 1:]
    c_fea_node2vec = sorted_array[821:, 1:]
    return m_fea_node2vec, c_fea_node2vec


# # TEST
# for i in range(1, 6):
#     edge_path = f"graph/edge_fold{i}.edgelist"
#
#     # G = nx.read_edgelist(edge_path, create_using=nx.Graph, nodetype=None,
#     #                      data=[('weight', float)])
#
#     # 创建一个带权重的无向图
#     G = nx.Graph()
#     # 从文件中读取节点对和权重信息
#     with open(edge_path, 'r') as file:
#         for line in file:
#             parts = line.strip().split()  # 假设文件中的数据以空格分隔
#             node1, node2, weight = parts[0], parts[1], float(parts[2])  # 假设权重为浮点数
#
#             # 添加带权重的边到图
#             G.add_edge(node1, node2, weight=weight)
#
#     N2V = Node2Vec(G, dimensions=64, walk_length=80, num_walks=10, p=1, q=1, workers=1, seed=random_seed)
#     model = N2V.fit(window=10, min_count=1, batch_words=4)
#     model.wv.save_word2vec_format('emb/emb_fold{}.emb'.format(i))
#     print("第{}折运行完毕".format(i))