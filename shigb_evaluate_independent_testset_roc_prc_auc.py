import pandas as pd
import numpy as np
from sklearn.metrics import auc, accuracy_score, precision_score, recall_score, roc_curve, roc_auc_score, \
    average_precision_score, precision_recall_curve, f1_score, confusion_matrix, matthews_corrcoef
import argparse
import os
import sys
import matplotlib.pyplot as plt


def read_prediction_or_true_results_csv(path, threshold):
    """
    读取用RUN_MODEL.py运行后的预测结果的csv文件
    csv文件的格式如下：

    slide_id,p_0,p_1,cutoff threshold,Final Prediction Result
    R19B3204R1L1_2019_06_17_10_45_26,0.922411054,0.07758893,0.3,FGFR-WT
    ...,...,...,...,...

    :param path: 待读取的csv文件路径，类型是str
    :return probability: 模型对样本队列的预测概率(p_1)，类型是dataframe
    """

    path = str(path)
    if os.path.isdir(path):

        sys.exit("所提供的路径是目录路径，请提供csv文件的路径")
    elif os.path.isfile(path):
        if os.path.splitext(path)[-1] == ".csv":
            prediction_results = pd.read_csv(path)
        elif os.path.splitext(path)[-1] == ".xlsx":
            prediction_results = pd.read_excel(path)
        else:
            sys.exit("请重新提供结果文件路径，需要是csv或xlsx格式")

    else:
        sys.exit("所提供的既不是目录路径也不是文件路径，请重新确认")

    if 'p_1' in prediction_results.columns:
        sample_prob_df = prediction_results[["slide_id", "p_1"]]
    else:
        sample_prob_df = prediction_results[["slide_id", "label"]]
        sample_prob_df['p_1'] = sample_prob_df['label'].apply(lambda x: 1 if x == 'FGFR_onco' else 0 if x == 'FGFR_WT' else 'other')

    sample_prob_df['label'] = ['FGFR_onco' if x > threshold else 'FGFR_WT' for x in sample_prob_df['p_1']]

    return sample_prob_df


def plot_ROC_and_calculate_AUC(sample_prob_df, sample_label_df, eval_results_dir_path):
    """
    用来绘制独立测试集的ROC曲线并绘制曲线下面积
    :param sample_prob_df: 样本预测概率dataframe
    :param sample_label_df: 样本实际标签dataframe
    :param eval_results_dir_path:  输出结果的所在目录
    :return: ROC曲线下面积
    """
    y_pred = np.array(sample_prob_df["p_1"])
    y_true = np.array(sample_label_df["p_1"])
    roc_auc_score(y_true, y_pred)

    fpr, tpr, thread = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC-AUC = %0.3f' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.title('Receiver operating characteristic (ROC)')
    if not os.path.exists(eval_results_dir_path):
        os.makedirs(eval_results_dir_path)
    plt.savefig(os.path.join(eval_results_dir_path, 'ROC curve.png'))

    return roc_auc


def plot_PRC_and_calculate_AUC(sample_prob_df, sample_label_df, eval_results_dir_path):
    """
    用来绘制独立测试集的PR曲线并绘制曲线下面积
    :param sample_prob_df: 样本预测概率dataframe
    :param sample_label_df: 样本实际标签dataframe
    :param eval_results_dir_path: 输出结果的所在目录
    :return: PRC曲线下面积
    """
    y_pred = np.array(sample_prob_df["p_1"])
    y_true = np.array(sample_label_df["p_1"])
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    prc_auc = auc(recall, precision)
    plt.figure()
    lw = 2
    plt.plot(recall, precision, color='green',
             lw=lw, label='PR-curve of class 1 (area = %0.3f)' % prc_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall (P-R) curve')
    plt.legend(loc="lower right")
    if not os.path.exists(eval_results_dir_path):
        os.makedirs(eval_results_dir_path)
    plt.savefig(os.path.join(eval_results_dir_path, 'PRC curve.png'))

    return prc_auc


def calculate_average_precision(sample_prob_df, sample_label_df):
    """
    计算独立测试集的平均精度
    :param sample_prob_df: 样本预测概率dataframe
    :param sample_label_df: 样本实际标签dataframe
    :return: 平均精度
    """
    y_pred = np.array(sample_prob_df["p_1"])
    y_true = np.array(sample_label_df["p_1"])
    ap = average_precision_score(y_true, y_pred)
    return ap


def calculate_accuracy_precision_recall_f1_score(sample_prob_df, sample_label_df, threshold):
    """
    计算独立测试集的准确率、精度、召回率、f1分数
    :param sample_prob_df: 样本预测概率dataframe
    :param sample_label_df: 样本实际标签dataframe
    :param threshold: 模型区分阳性阴性样本的阈值
    :return: 准确率，精度，召回率，F1分数
    """
    y_pred = np.array(sample_prob_df["p_1"])
    y_true = np.array(sample_label_df["p_1"])
    y_pred = [1 if score >= threshold else 0 for score in y_pred]
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, precision, recall, f1


def calculate_TPR_FPR_TNR_FNR_MCC(sample_prob_df, sample_label_df, threshold):
    y_pred = np.array(sample_prob_df["p_1"])
    y_true = np.array(sample_label_df["p_1"])
    y_pred = [1 if score >= threshold else 0 for score in y_pred]
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    TPR_sensitivity = TP / (TP + FN)
    FPR = FP / (FP + TN)
    TNR_specificity = TN / (TN + FP)
    FNR = FN / (FN + TP)
    mcc = matthews_corrcoef(y_true, y_pred)

    return TPR_sensitivity, FPR, TNR_specificity, FNR, mcc


def main(parameters):
    sample_prob_df = read_prediction_or_true_results_csv(parameters.prediction_results_csv_path, parameters.threshold)
    sample_label_df = read_prediction_or_true_results_csv(parameters.true_results_csv_path, parameters.threshold)
    desired_order = sample_prob_df['slide_id']
    sample_label_df.set_index('slide_id', inplace=True)
    sample_label_df = sample_label_df.reindex(desired_order)
    sample_label_df.reset_index(inplace=True)

    ROC_AUC = plot_ROC_and_calculate_AUC(sample_prob_df, sample_label_df, parameters.eval_results_dir_path)
    PRC_AUC = plot_PRC_and_calculate_AUC(sample_prob_df, sample_label_df, parameters.eval_results_dir_path)
    average_precision = calculate_average_precision(sample_prob_df, sample_label_df)
    accuracy, precision, recall, f1 = calculate_accuracy_precision_recall_f1_score(sample_prob_df, sample_label_df,
                                                                                   parameters.threshold)
    TPR_sensitivity, FPR, TNR_specificity, FNR, mcc = calculate_TPR_FPR_TNR_FNR_MCC(sample_prob_df, sample_label_df,
                                                                                    parameters.threshold)

    print("预测表格")
    print(sample_prob_df)
    print("实际结果")
    print(sample_label_df)
    print()
    print("ROC_AUC:", ROC_AUC)
    print("PRC_AUC:", PRC_AUC)
    print("average_precision:", average_precision)

    print()
    print("THRESHOLD:", parameters.threshold)
    print("accuracy:", accuracy)
    print("precision", precision)
    print("recall:", recall)
    print("f1", f1)
    print("TPR_sensitivity:", TPR_sensitivity)
    print("FPR:", FPR)
    print("TNR_specificity:", TNR_specificity)
    print("FNR:", FNR)
    print("mcc:", mcc)

    print(f"ROC曲线和PR曲线的图像已经存储到{os.path.abspath(parameters.eval_results_dir_path)}中")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="本程序可以根据RUN_MODEL的预测结果来进行性能评估")
    parser.add_argument("--prediction_results_csv_path", type=str, required=True, help="请输入RUN_MODEL.py预测结果的csv文件路径")
    parser.add_argument("--true_results_csv_path", type=str, required=True, help="请输入真实结果的csv文件路径")
    parser.add_argument("--eval_results_dir_path", type=str, default="./eval_results",
                        help="请输入评估结果的存放路径，默认为当前路径下的eval_results目录，即'./eval_results'")
    parser.add_argument("--ROC", default=False, action="store_true", help="若需要绘图ROC并计算曲线下面积，请添加该参数")
    parser.add_argument("--PRC", default=False, action="store_true", help="若需要绘图PRC并计算曲线下面积，请添加该参数")
    parser.add_argument("--average_precision", default=False, action="store_true", help="若需要计算平均精度，请添加该参数")
    parser.add_argument("--threshold", type=float, default=0.5, help="请给一个默认的模型区分阳性阴性样本的阈值，以便计算准确率，精度，召回率等指标，默认值为0.5")
    parser.add_argument("--threshold_related_metrics_1", default=False, action="store_true",
                        help="若需要计算准确率、精度、召回率、F1分数指标，请添加该参数")
    parser.add_argument("--threshold_related_metrics_2", default=False, action="store_true",
                        help="若需要计算真阳性率TPR/灵敏度、假阳性率FPR、真阴性率TNR/特异度、假阴性率FNR、MCC指标，请添加该参数")
    # parser.add_argument("--other_metrics", default=False, action="store_true", help="若需要计算其他指标，请添加该参数")
    args = parser.parse_args()

    main(args)
