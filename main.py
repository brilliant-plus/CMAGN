# -*- coding: utf-8 -*-
import time
import warnings
import pandas as pd
import numpy as np
import itertools
from sklearn import metrics
from sklearn.model_selection import KFold
from sim_process import sim_thresholding
from data_process import single_generate_graph_adj_and_feature, get_ColRow_AS, get_edgelist
from data_generate import get_all_samples
from GATE.get_gate_feature import get_gate_feature
from N2V.feaFunc_N2V import N2V, emb_split
from net_analyzer import reconstruct_similarity_network, network_analyzer
from plt_picture import plt_fold_ROC

warnings.filterwarnings("ignore")
starttime = time.time()

# data
seqSimMatrix_circRNA = pd.read_csv("data/seqSim_circRNA.csv", header=None, dtype=np.float32).to_numpy()
seqSimMatrix_miRNA = pd.read_csv("data/seqSim_miRNA.csv", header=None, dtype=np.float32).to_numpy()
association = pd.read_csv("data/matrix_AS.csv", header=None).to_numpy()

seed, n_splits = 10, 5
c_threshold, d_threshold, epochs, reconstruct_similarity, analyzer = [0.8], [0.8], [200], ["cos"], ["xj-NCP"]

for s in itertools.product(c_threshold, d_threshold, epochs, reconstruct_similarity, analyzer):
    all_act, all_pre, fpr_list, tpr_list, auroc_list= [], [], [], [], []

    # Get all samples from the correlation matrix.
    samples = get_all_samples(conjunction=association, random_seed=seed)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = 0
    for train_index, val_index in kf.split(samples):
        fold = fold + 1
        print(f"===================================== {fold} folds test start =====================================")

        train_samples = samples[train_index, :]
        val_samples = samples[val_index, :]
        # Save operations - training set, test set samples.
        train_samples_path = f"samples/trainSamples_fold{fold}.csv"
        np.savetxt(train_samples_path, train_samples, fmt='%d, %d, %.8f', delimiter=',')
        val_samples_path = f"samples/valSamples_fold{fold}.csv"
        np.savetxt(val_samples_path, val_samples, fmt='%d, %d, %.6f', delimiter=',')

        # Iterate through the test set to remove associations to get new associations.
        new_association = association.copy()
        for row in val_samples:
            new_association[int(row[0]), int(row[1])] = 0
        # Save operation - new association.
        new_association_path = f"samples/new_association_fold{fold}.csv"
        np.savetxt(new_association_path, new_association, fmt='%.8f', delimiter=',')

        # Generates features by GATE
        print("The similarity network used is: seq")
        m_sim = seqSimMatrix_miRNA
        c_sim = seqSimMatrix_circRNA

        # Processing edges: thresholding binarizes the similarity network.
        print("The number of similarity network edges divided by miRNA, circRNA thresholds are: ", end='')
        m_net = sim_thresholding(m_sim, s[1])
        c_net = sim_thresholding(c_sim, s[0])
        print(" ")

        # Processing features: select AS rows or columns as node features.
        m_fea, c_fea = get_ColRow_AS(new_association)
        print("The miRNA, circRNA similarity network node feature matrix dimensions are : ", m_fea.shape, c_fea.shape)

        # The network edges and node features are processed into a form suitable for GATE input
        m_networks, m_features = single_generate_graph_adj_and_feature(m_net, m_fea)
        c_networks, c_features = single_generate_graph_adj_and_feature(c_net, c_fea)
        m_embeddings = get_gate_feature(net=m_networks, fea=m_features, epochs=s[2], l=1)
        c_embeddings = get_gate_feature(net=c_networks, fea=c_features, epochs=s[2], l=1)
        print("GATE generates representation of miRNAs successfully... "
              "\nGATE generates representation of circRNAs  successfully...")
        # Save operation - the GATE-generated features
        m_embeddings_path = f"GATE/emb/m_embeddings-cth{s[0]}-mth{s[1]}_fold{fold}.csv"
        c_embeddings_path = f"GATE/emb/c_embeddings_cth{s[0]}-mth{s[1]}_fold{fold}.csv"
        np.savetxt(m_embeddings_path, m_embeddings, fmt="%.8f", delimiter=",")
        np.savetxt(c_embeddings_path, c_embeddings, fmt="%.8f", delimiter=",")

        # Generates features by node2vec
        dim = 128

        new_ass_N2V = association.copy()
        for row in val_samples:
            new_ass_N2V[int(row[0]), int(row[1])] = 1e-6

        new_edgelist = get_edgelist(new_ass_N2V)
        edge_path = f"N2V/graph/edge_fold{fold}.edgelist"
        np.savetxt(edge_path, new_edgelist, fmt="%d %d %f", delimiter=' ')

        emb_matrix = N2V(edge_path=edge_path, para_dim=dim, random_seed=seed)

        # Save operation - overall emb_matrix
        emb_path = f"N2V/emb/emb_fold{fold}.emb"
        np.savetxt(emb_path, emb_matrix, delimiter=' ')
        m_fea, c_fea = emb_split(emb_matrix)

        # Save operation - the no2vec-generated features
        m_fea_path = f"N2V/emb/m_emb_fold{fold}.csv"
        c_fea_path = f"N2V/emb/c_emb_fold{fold}.csv"
        np.savetxt(m_fea_path, m_fea, fmt='%.8f', delimiter=',')
        np.savetxt(c_fea_path, c_fea, fmt='%.8f', delimiter=',')

        # Optional simplified operation - Save the fea directly after the first full five cross validation runs.
        m_fea_path = f"N2V/emb/m_emb_fold{fold}.csv"
        c_fea_path = f"N2V/emb/c_emb_fold{fold}.csv"
        m_fea_N2V = pd.read_csv(m_fea_path, header=None)
        c_fea_N2V = pd.read_csv(c_fea_path, header=None)

        # Combining GATE and N2V generated feature representation
        m_embeddings = np.concatenate((m_embeddings, m_fea_N2V), axis=1)
        c_embeddings = np.concatenate((c_embeddings, c_fea_N2V), axis=1)

        # Reconstructing similarity networks
        m_reconstruct_similarity, c_reconstruct_similarity = reconstruct_similarity_network(m_embeddings, c_embeddings, s[3])
        m_sim, c_sim = m_reconstruct_similarity, c_reconstruct_similarity

        # Generating recommender systems
        predictedMatrix = network_analyzer(m_sim, c_sim, new_association, s[4])

        # Save operation - recommended matrix
        predictedMatrix_path = f"result/predictedMatrix{fold}-{s[0]}-{s[1]}-{s[2]}-{s[3]}-{s[4]}.csv"
        np.savetxt(predictedMatrix_path, predictedMatrix, delimiter=',', fmt="%.8f")

        # Computational performance
        # actual value
        act = val_samples[:, 2]
        all_act = np.concatenate((all_act, act))

        # predicted value
        pre = np.array([])
        for row in val_samples:
            i = int(row[0])
            j = int(row[1])
            pre = np.append(pre, predictedMatrix[i, j])
        all_pre = np.concatenate((all_pre, pre))

        # Data related to per-fold ROC curves and AUC values
        FPR, TPR, thresholds = metrics.roc_curve(act, pre)
        AUC = metrics.auc(FPR, TPR)
        print(f"The AUC values for the {fold} fold are.", AUC)

        fpr_list.append(list(FPR))
        tpr_list.append(list(TPR))
        auroc_list.append(AUC)

    """Calculate the performance of each fold and record it in preparation for the graph."""
    mean_FPR, mean_TPR, mean_thresholds = metrics.roc_curve(all_act, all_pre)
    mean_AUROC = metrics.auc(mean_FPR, mean_TPR)
    auroc_list.append(mean_AUROC)

    # Drawing ROC curves and AUC values
    ROC_path = f"result/ROC"
    plt_fold_ROC(curveName=ROC_path, x_foldList=fpr_list, y_foldList=tpr_list, auc_foldList=auroc_list, x_mean=mean_FPR, y_mean=mean_TPR, auc_mean=mean_AUROC)

    # Recording of time spent on major procedures
    endtime = time.time()
    TIME = endtime-starttime
    print("The program is time-consuming : ", TIME)

    # Results
    resultContent = f"{mean_AUROC:.4f}, {s[0]}, {s[1]}, {s[2]}, {s[3]}, {s[4]}, {TIME}"
    print("The results are as follows : \n", "mean_AUROC, cth, mth, epochs, reconstruct_similarity, analyzer, time")
    print(resultContent, "\n===========================================================================")