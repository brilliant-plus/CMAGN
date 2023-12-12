# -*- coding: utf-8 -*-
import time
import warnings
import pandas as pd
import numpy as np
import itertools
from sklearn import metrics
from sklearn.model_selection import KFold
from data_generate import get_all_samples
from sim_process import sim_thresholding
from data_process import single_generate_graph_adj_and_feature
from get_gate_feature import get_gate_feature
from network_analyzer import reconstruct_similarity_network, network_analyzer
from pltPicture import plt_fold_ROC
from N2V.feaFunc_N2V import N2V, emb_split
from function import get_ColRow_AS, get_edgelist
warnings.filterwarnings("ignore")
starttime = time.time()

# 相似性矩阵 关联矩阵
seqSimMatrix_circRNA = pd.read_csv("data/seqSim_circRNA.csv", header=None, dtype=np.float32).to_numpy()
seqSimMatrix_miRNA = pd.read_csv("data/seqSim_miRNA.csv", header=None, dtype=np.float32).to_numpy()
association = pd.read_csv("data/matrix_AS.csv", header=None).to_numpy()  # 关联网络中的值为1  # ok

# 超参数
seed, n_splits = 10, 5
c_threshold, d_threshold, epochs, reconstruct_similarity, analyzer = [0.8], [0.8], [200], ["cos"], ["xj-NCP"]

for s in itertools.product(c_threshold, d_threshold, epochs, reconstruct_similarity, analyzer):
    all_act, all_pre, fpr_list, tpr_list, auroc_list= [], [], [], [], []

    # 从关联矩阵中获取所有样本
    samples = get_all_samples(conjunction=association, random_seed=seed)

    # 从所有样本samples划分5折，记录对应每折训练集测试集的索引
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = 0
    for train_index, val_index in kf.split(samples):
        fold = fold + 1
        print(f"======================== {fold} folds test start ========================")

        # 根据索引划分为训练集、测试集
        train_samples = samples[train_index, :]
        val_samples = samples[val_index, :]
        # 保存操作——训练集、测试集样本
        train_samples_path = f"samples/trainSamples_fold{fold}.csv"
        np.savetxt(train_samples_path, train_samples, fmt='%d, %d, %.8f', delimiter=',')
        val_samples_path = f"samples/valSamples_fold{fold}.csv"
        np.savetxt(val_samples_path, val_samples, fmt='%d, %d, %.6f', delimiter=',')

        # 遍历测试集去除关联得到新关联
        new_association = association.copy()
        for row in val_samples:
            new_association[int(row[0]), int(row[1])] = 0
        print("去掉测试集关联后，miRNA与circRNA之间关联个数为：", np.count_nonzero(new_association))
        # 保存操作——去掉测试集中关联信息的新关联
        new_association_path = f"samples/new_association_fold{fold}.csv";
        np.savetxt(new_association_path, new_association, fmt='%.8f', delimiter=',')

        # GATE处理AS生成特征
        # 选择相似性矩阵生成网络
        print("使用的相似性网络是seq")
        m_sim = seqSimMatrix_miRNA
        c_sim = seqSimMatrix_circRNA

        # 处理边：阈值将关联网络二值化
        print("miRNA、circRNA阈值划分的网络边数目分别为:", end='')
        m_net = sim_thresholding(m_sim, s[1])
        c_net = sim_thresholding(c_sim, s[0])
        print(" ")

        # 选择AS行或列作为节点特征
        m_fea, c_fea = get_ColRow_AS(new_association)
        print("miRNA、circRNA网络节点特征矩阵维度分别为:", m_fea.shape, c_fea.shape)

        # 将网络边和节点特征处理成适合GATE输入的形式, 然后带入GATE生成特征
        m_networks, m_features = single_generate_graph_adj_and_feature(m_net, m_fea)
        c_networks, c_features = single_generate_graph_adj_and_feature(c_net, c_fea)
        m_embeddings = get_gate_feature(net=m_networks, fea=m_features, epochs=s[2], l=1)
        c_embeddings = get_gate_feature(net=c_networks, fea=c_features, epochs=s[2], l=1)
        print("GATE生成mi特征成功...\nGATE生成circ特征成功...")

        # 保存操作——将GATE生成特征进行保存
        m_embeddings_path = f"embeddings/m_embeddings-cth{s[0]}-mth{s[1]}_fold{fold}.csv"
        c_embeddings_path = f"embeddings/c_embeddings_cth{s[0]}-mth{s[1]}_fold{fold}.csv"
        np.savetxt(m_embeddings_path, m_embeddings, fmt="%.8f", delimiter=",")
        np.savetxt(c_embeddings_path, c_embeddings, fmt="%.8f", delimiter=",")

        # # N2V处理AS生成特征
        dim = 128

        new_ass_N2V = association.copy()
        for row in val_samples:
            new_ass_N2V[int(row[0]), int(row[1])] = 1e-6

        new_edgelist = get_edgelist(new_ass_N2V)
        edge_path = f"N2V/graph/edge_fold{fold}.edgelist"
        np.savetxt(edge_path, new_edgelist, fmt="%d %d %f", delimiter=' ')

        emb_matrix = N2V(edge_path=edge_path, para_dim=dim, random_seed=seed)

        # 保存操作——整体emb_matrix
        emb_path = f"N2V/emb/emb_fold{fold}.emb"
        np.savetxt(emb_path, emb_matrix, delimiter=' ')
        m_fea, c_fea = emb_split(emb_matrix)  # 【参数】是emb矩阵，切割作为最终fea，并保存（每折也在变化）

        # 保存操作——N2V生成的特征
        m_fea_path = f"N2V/emb/m_emb_fold{fold}.csv"
        c_fea_path = f"N2V/emb/c_emb_fold{fold}.csv"
        np.savetxt(m_fea_path, m_fea, fmt='%.8f', delimiter=',')
        np.savetxt(c_fea_path, c_fea, fmt='%.8f', delimiter=',')

        # 跑完第一次完整五折直接保存fea，以后直接读取fea文件
        m_fea_path = f"N2V/emb/m_emb_fold{fold}.csv"
        c_fea_path = f"N2V/emb/c_emb_fold{fold}.csv"
        m_fea_N2V = pd.read_csv(m_fea_path, header=None)
        c_fea_N2V = pd.read_csv(c_fea_path, header=None)

        # 结合GATE和N2V生成的特征表示
        m_embeddings = np.concatenate((m_embeddings, m_fea_N2V), axis=1)
        c_embeddings = np.concatenate((c_embeddings, c_fea_N2V), axis=1)

        # 重构相似性网络
        m_reconstruct_similarity, c_reconstruct_similarity = reconstruct_similarity_network(m_embeddings, c_embeddings, s[3])
        m_sim, c_sim = m_reconstruct_similarity, c_reconstruct_similarity

        # 生成推荐系统
        predictedMatrix = network_analyzer(m_sim, c_sim, new_association, s[4])

        # 保存操作——推荐矩阵
        predictedMatrix_path = f"result/predictedMatrix{fold}-{s[0]}-{s[1]}-{s[2]}-{s[3]}-{s[4]}.csv"
        np.savetxt(predictedMatrix_path, predictedMatrix, delimiter=',', fmt="%.8f")

        # 根据推荐矩阵和原有关联矩阵，取测试集对应元素组成实际值和预测值列表，对比得出性能
        # actual value
        act = val_samples[:, 2]
        print("here is actual value: \n", act)
        all_act = np.concatenate((all_act, act))  # 将每一折act汇总，为后面计算mean值准备数据

        # predict value
        pre = np.array([])
        for row in val_samples:
            i = int(row[0])
            j = int(row[1])
            pre = np.append(pre, predictedMatrix[i, j])
        print("here is predict value: \n", pre)
        all_pre = np.concatenate((all_pre, pre))  # 将每一折pre汇总，为后面计算mean值准备数据

        # 每折ROC曲线和AUC值相关数据
        FPR, TPR, thresholds = metrics.roc_curve(act, pre)
        AUC = metrics.auc(FPR, TPR)
        print(f"第{fold}折的AUC值为:", AUC)

        fpr_list.append(list(FPR))
        tpr_list.append(list(TPR))
        auroc_list.append(AUC)

    """根据全部测试集的预测值、实际值，计算每折性能并记录,为画图做准备"""
    mean_FPR, mean_TPR, mean_thresholds = metrics.roc_curve(all_act, all_pre)
    mean_AUROC = metrics.auc(mean_FPR, mean_TPR)
    auroc_list.append(mean_AUROC)

    # 画图ROC曲线和AUC值
    ROC_path = f"result/ROC.jpg"
    plt_fold_ROC(curveName=ROC_path, x_foldList=fpr_list, y_foldList=tpr_list, auc_foldList=auroc_list, x_mean=mean_FPR, y_mean=mean_TPR, auc_mean=mean_AUROC)

    # 记录主要程序耗时
    endtime = time.time()
    TIME = endtime-starttime
    print("程序耗时：", TIME)

    # 写入结果到文件中
    resultContent = f"{mean_AUROC:.4f}, {s[0]}, {s[1]}, {s[2]}, {s[3]}, {s[4]}, {TIME}"
    print("结果如下：\n", "mean_AUROC, cth, mth, reconstruct_similarity, analyzer, time")
    print(resultContent, "\n===========================================================================")