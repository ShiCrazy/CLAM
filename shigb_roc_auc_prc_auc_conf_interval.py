import argparse

import pandas as pd
import os
import glob
from numpy import interp
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.metrics import precision_recall_curve, auc


def plot_threshold_curves(dir_path, suffix, threshold_value):
    plt.figure(figsize=(20, 10))

    # 定义通用阈值
    common_thresholds = np.linspace(0, 1, 1000)

    # 存储插值精度和召回率值的列表
    interp_precisions = []
    interp_recalls = []
    interp_specificities = []
    interp_accuracies = []

    # 存储每个文件的精度、召回率、准确率、特异性
    precisions = []
    recalls = []
    accuracies = []
    specificities = []
    TPs = []
    TNs = []
    FPs = []
    FNs = []

    csv_files = sorted([f for f in glob.glob(os.path.join(dir_path, '*.csv'))
                        if re.match(r'.*fold_\d+\.csv$', f)])

    for csv_file in csv_files:
        data = pd.read_csv(csv_file)
        Y = data['Y']
        probas_ = data['p_1']

        precision, recall, thresholds = precision_recall_curve(Y, probas_)
        thresholds = np.append(thresholds, 1)

        # 计算 TP, TN, FP, FN
        TP = np.sum((Y == 1) & (probas_ >= threshold_value))
        TN = np.sum((Y == 0) & (probas_ < threshold_value))
        FP = np.sum((Y == 0) & (probas_ >= threshold_value))
        FN = np.sum((Y == 1) & (probas_ < threshold_value))

        TPs.append(TP)
        TNs.append(TN)
        FPs.append(FP)
        FNs.append(FN)

        # 将精度、召回率、准确率、特异性插值到通用阈值上
        interp_precision = np.interp(common_thresholds, thresholds, precision)
        interp_recall = np.interp(common_thresholds, thresholds, recall)
        interp_accuracy = [np.mean(Y == (probas_ >= t)) for t in common_thresholds]
        interp_specificity = [(np.sum((Y == 0) & (probas_ < t)) / np.sum(Y == 0)) for t in common_thresholds]

        interp_precisions.append(interp_precision)
        interp_recalls.append(interp_recall)
        interp_accuracies.append(interp_accuracy)
        interp_specificities.append(interp_specificity)

        # 在指定阈值下的指标
        idx = np.where(thresholds >= threshold_value)[0][0]
        precisions.append(precision[idx])
        recalls.append(recall[idx])
        accuracies.append(np.mean(Y == (probas_ >= threshold_value)))
        specificities.append(np.sum((Y == 0) & (probas_ < threshold_value)) / np.sum(Y == 0))

    # 计算插值精度和召回率的均值
    mean_precision = np.mean(interp_precisions, axis=0)
    mean_recall = np.mean(interp_recalls, axis=0)
    mean_accuracy = np.mean(interp_accuracies, axis=0)
    mean_specificity = np.mean(interp_specificities, axis=0)

    # 绘制精度-阈值曲线
    plt.subplot(2, 2, 1)
    plt.plot(common_thresholds, mean_precision, color='blue', lw=2, label='Mean Precision')
    plt.title(f'Precision-Threshold Curve on {suffix}')
    plt.xlabel('Threshold')
    plt.ylabel('Precision')
    plt.legend()

    # 绘制召回率-阈值曲线
    plt.subplot(2, 2, 2)
    plt.plot(common_thresholds, mean_recall, color='red', lw=2, label='Mean Recall/Sensitivity')
    plt.title(f'Recall/Sensitivity-Threshold Curve on {suffix}')
    plt.xlabel('Threshold')
    plt.ylabel('Recall/Sensitivity')
    plt.legend()

    # 绘制准确率-阈值曲线
    plt.subplot(2, 2, 3)
    plt.plot(common_thresholds, mean_accuracy, color='green', lw=2, label='Mean Accuracy')
    plt.title(f'Accuracy-Threshold Curve on {suffix}')
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.legend()

    # 绘制特异性-阈值曲线
    plt.subplot(2, 2, 4)
    plt.plot(common_thresholds, mean_specificity, color='purple', lw=2, label='Mean Specificity')
    plt.title(f'Specificity-Threshold Curve on {suffix}')
    plt.xlabel('Threshold')
    plt.ylabel('Specificity')
    plt.legend()

    # 保存图形
    plt.savefig(os.path.join(dir_path, 'threshold_curves.png'), bbox_inches='tight')
    plt.show()

    # 输出给定阈值下的平均指标
    print(f"Average Precision at threshold {threshold_value}: {np.mean(precisions)}")
    print(f"Average Recall at threshold {threshold_value}: {np.mean(recalls)}")
    print(f"Average Accuracy at threshold {threshold_value}: {np.mean(accuracies)}")
    print(f"Average Specificity at threshold {threshold_value}: {np.mean(specificities)}")

    print(f"Average TP at threshold {threshold_value}: {np.mean(TPs)}")
    print(f"Average TN at threshold {threshold_value}: {np.mean(TNs)}")
    print(f"Average FP at threshold {threshold_value}: {np.mean(FPs)}")
    print(f"Average FN at threshold {threshold_value}: {np.mean(FNs)}")

def plot_pr_from_dir(dir_path, suffix, given_threshold):
    precs = []
    recs = []  # 存储召回率
    thresholds_list = []
    aucs = []
    mean_recall = np.linspace(0, 1, 100)
    plt.figure(figsize=(10, 10))

    csv_files = sorted([f for f in glob.glob(os.path.join(dir_path, '*.csv'))
                        if re.match(r'.*fold_\d+\.csv$', f)])

    precs_at_threshold = []

    for i, csv_file in enumerate(csv_files):
        data = pd.read_csv(csv_file)
        Y = data['Y']
        probas_ = data['p_1']

        precision, recall, thresholds = precision_recall_curve(Y, probas_)
        reversed_recall = recall[::-1]
        reversed_precision = precision[::-1]
        reversed_thresholds = np.r_[thresholds[::-1], thresholds[-1]]
        thresholds_list.append(reversed_thresholds)

        prec = np.interp(mean_recall, reversed_recall, reversed_precision)
        prec[0] = 1.0
        precs.append(prec)
        pr_auc = auc(recall, precision)
        aucs.append(pr_auc)
        plt.plot(recall, precision, lw=1, alpha=0.3, label=f'PR fold {i + 1} (AUC = {pr_auc:.2f})')

        index = np.searchsorted(thresholds, given_threshold, side='right') - 1
        recall_at_threshold = recall[index]
        precision_at_threshold = precision[index]

        recs.append(recall_at_threshold)
        precs_at_threshold.append(precision_at_threshold)

    # Plot the average PR curve
    mean_prec = np.mean(precs, axis=0)
    mean_auc = auc(mean_recall, mean_prec)
    std_auc = np.std(aucs)
    plt.plot(mean_recall, mean_prec, color='b', label=f'Mean PR (AUC = {mean_auc:.2f} ± {std_auc:.2f})', lw=2, alpha=.8)

    # Plot the confidence intervals
    std_prec = np.std(precs, axis=0)
    precs_upper = np.minimum(mean_prec + 1.96 * std_prec, 1)
    precs_lower = np.maximum(mean_prec - 1.96 * std_prec, 0)
    plt.fill_between(mean_recall, precs_lower, precs_upper, color='grey', alpha=.2, label='± 95% CI')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve on {suffix}')
    plt.legend(loc="upper right", prop={'size': 10})


    plt.savefig(os.path.join(dir_path, 'pr_plot.png'), bbox_inches='tight')

    mean_recall_at_threshold = np.mean(recs)
    mean_precision_at_threshold = np.mean(precs_at_threshold)
    print(f'Mean recall at threshold {given_threshold}: {mean_recall_at_threshold}')
    print(f'Mean precision at threshold {given_threshold}: {mean_precision_at_threshold}')

# def plot_pr_from_dir(dir_path, suffix, threshold_value):
 
#     precs = []
#     recs = []
#     aucs = []
#     common_thresholds = np.linspace(0, 1, 1000)
#     plt.figure(figsize=(10, 10))

#     csv_files = sorted([f for f in glob.glob(os.path.join(dir_path, '*.csv'))
#                         if re.match(r'.*fold_\d+\.csv$', f)])

#     for i, csv_file in enumerate(csv_files):
#         data = pd.read_csv(csv_file)
#         Y = data['Y']
#         probas_ = data['p_1']

#         # Compute PR curve and area under the curve
#         precision, recall, thresholds = precision_recall_curve(Y, probas_)
#         # Append max threshold to make it of the same length as precision and recall
#         thresholds = np.append(thresholds, 1)

#         # Interpolate precision and recall for common thresholds
#         interp_prec = np.interp(common_thresholds, thresholds, precision)
#         interp_recall = np.interp(common_thresholds, thresholds, recall)

#         precs.append(interp_prec)
#         recs.append(interp_recall)
#         pr_auc = auc(interp_recall, interp_prec)
#         aucs.append(pr_auc)
#         plt.plot(interp_recall, interp_prec, lw=1, alpha=0.3, label=f'PR fold {i + 1} (AUC = {pr_auc:.2f})')

#     # Calculate the mean and standard deviation for precision and recall
#     mean_prec = np.mean(precs, axis=0)
#     mean_rec = np.mean(recs, axis=0)
#     std_prec = np.std(precs, axis=0)
#     std_rec = np.std(recs, axis=0)

#     # Plot the average PR curve and the 95% confidence intervals
#     mean_auc = auc(mean_rec, mean_prec)
#     std_auc = np.std(aucs)
#     plt.plot(mean_rec, mean_prec, color='b', label=f'Mean PR (AUC = {mean_auc:.2f} ± {std_auc:.2f})', lw=2, alpha=.8)

#     prec_upper = np.minimum(mean_prec + 1.96 * std_prec, 1)
#     prec_lower = np.maximum(mean_prec - 1.96 * std_prec, 0)
#     plt.fill_between(mean_rec, prec_lower, prec_upper, color='grey', alpha=.2, label='± 95% CI')

#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.title(f'Precision-Recall Curve on {suffix}')
#     plt.legend(loc="upper right", prop={'size': 10})

#     # Save the figure before showing it
#     plt.savefig(os.path.join(dir_path, 'pr_plot.png'), bbox_inches='tight')

#     # Find index of the specified threshold and calculate mean precision and recall at this threshold
#     index = np.argmin(np.abs(common_thresholds - threshold_value))
#     mean_precision_at_threshold = np.mean([p[index] for p in precs])
#     mean_recall_at_threshold = np.mean([r[index] for r in recs])
#     print(f'Mean precision at threshold {threshold_value}: {mean_precision_at_threshold}')
#     print(f'Mean recall at threshold {threshold_value}: {mean_recall_at_threshold}')


def plot_roc_from_dir(dir_path, suffix):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    youden_indexes_list = []
    thresholds_list = []

    plt.figure(figsize=(10, 10))

    csv_files = sorted([f for f in glob.glob(os.path.join(dir_path, '*.csv'))
                        if re.match(r'.*fold_\d+\.csv$', f)])

    for i, csv_file in enumerate(csv_files):
        data = pd.read_csv(csv_file)
        Y = data['Y']
        probas_ = data['p_1']

        # Compute ROC curve and area under the curve
        fpr, tpr, thresholds = roc_curve(Y, probas_)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0

        youden_indexes = tpr - fpr
        youden_indexes_list.append(interp(mean_fpr, fpr, youden_indexes))
        thresholds_list.append(thresholds)

        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'ROC fold {i + 1} (AUC = {roc_auc:.2f})')

    # Plot the average ROC curve
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})', lw=2, alpha=.8)

    # Plot the confidence intervals
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + 1.96 * std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - 1.96 * std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label='± 95% CI')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic on {suffix}')
    plt.legend(loc="lower right", prop={'size': 10})
    plt.savefig(os.path.join(dir_path, 'roc_plot.png'), bbox_inches='tight')

    mean_youden_indexes = np.mean(youden_indexes_list, axis=0)
    max_mean_youden_index_at = np.argmax(mean_youden_indexes)
    max_mean_youden_index_thresholds = [t[max_mean_youden_index_at] if max_mean_youden_index_at < len(t) else t[-1] for
                                        t in thresholds_list]
    max_mean_youden_index_threshold = np.mean(max_mean_youden_index_thresholds)
    print(f'Max mean Youden index threshold: {max_mean_youden_index_threshold}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser("快速根据eval的k折交叉验证的结果绘制ROC曲线及置信区间")
    parser.add_argument("--eval_dir_path", type=str, help="fold_*.csv文件所在的目录的路径")
    parser.add_argument("--given_threshold", type=float, default=0.5, help="给出一个阈值，之后会根据这个阈值计算对应精度和召回率，默认值为0.5")
    parser.add_argument("--suffix", type=str, default=None, help="标题后缀")
    args = parser.parse_args()
    plot_roc_from_dir(args.eval_dir_path, args.suffix)
    plot_pr_from_dir(args.eval_dir_path, args.suffix, args.given_threshold)
    plot_threshold_curves(args.eval_dir_path, args.suffix, args.given_threshold)
