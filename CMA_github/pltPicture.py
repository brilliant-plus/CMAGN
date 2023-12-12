# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def plt_fold_ROC(curveName, x_foldList, y_foldList, auc_foldList, x_mean, y_mean, auc_mean):
    # """画图ROC曲线和AUC值-放大镜"""
    # 调整数据格式
    x_foldList = np.array(x_foldList, dtype=object)
    y_foldList = np.array(y_foldList, dtype=object)
    auc_foldList = np.array(auc_foldList, dtype=object)

    # 设置画布和实例
    fig_1, ax_1 = plt.subplots(1, 1)

    # region 绘制母图曲线
    # 每一折结果曲线 —— 五分之一数据集
    colors = ['C1', 'C2', 'C3', 'C4', 'C5']
    for i in range(5):
        ax_1.plot(x_foldList[i], y_foldList[i], label="ROC fold {}(AUROC={:.4f})".format(i, auc_foldList[i])
                  , color=colors[i], linestyle='--')
    # 平均结果曲线 —— 全部数据集
    ax_1.plot(x_mean, y_mean, label="Mean ROC(AUROC={:.4f})".format(auc_mean), color='red', linestyle='-')

    # 设置母图参数
    plt.legend()  # 添加图例
    plt.xlabel('False Positive Rate')  # 添加横轴标签
    plt.ylabel('True Positive Rate')  # 添加纵轴标签
    plt.title(' Receiver Operating Characteristic Curves')  # 添加图像标题
    # endregion 绘制母图曲线

    # region 保存并显示图片
    plt.savefig(f'{curveName}', dpi=500)  # 保存图片
    print(f"结果保存到图片{curveName}成功")
    plt.show()  # 显示绘制的图像
    # endregion 保存并显示图片

