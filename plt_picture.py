# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plt_fold_ROC(curveName, x_foldList, y_foldList, auc_foldList, x_mean, y_mean, auc_mean):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'

    x_foldList = np.array(x_foldList, dtype=object)
    y_foldList = np.array(y_foldList, dtype=object)
    auc_foldList = np.array(auc_foldList, dtype=object)

    fig_1, ax_1 = plt.subplots(1, 1)

    colors = ['C1', 'C2', 'C3', 'C4', 'C5']
    for i in range(5):
        ax_1.plot(x_foldList[i], y_foldList[i], label="ROC fold {} (AUROC={:.4f})".format(i, auc_foldList[i])
                  , color=colors[i])

    ax_1.plot(x_mean, y_mean, label="Mean ROC (AUROC={:.4f})".format(auc_mean), color='red')

    # para
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(' Receiver Operating Characteristic Curves')
    plt.savefig(f'{curveName}.pdf', format='pdf', dpi=500)
    plt.show()