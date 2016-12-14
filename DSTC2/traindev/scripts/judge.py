# -*- coding:utf-8 -*-  
import numpy as np

__author__ = "JOHNKYON"


# def recall(y_true, y_pred):
#     TP = 0
#     FN = 0
#     for n in len(y_true):
#         if y_true[n][0] == 1 and y_pred[n][0] == 0:
#             FN += 1
#         elif y_true[n][0] == 1 and y_pred[n][0] == 1:
#             TP += 1
#     rc = TP/(TP+FN)
#     return rc

def recall_precision_F(y_true, y_pred, threshold):
    for n in range(0, len(y_pred)):
        for m in range(0, len(y_pred[0])):
            if y_pred[n][m] > threshold:
                y_pred[n][m] = 1
            else:
                y_pred[n][m] = 0
    TP = float(0)
    FP = float(0)
    FN = float(0)
    TN = float(0)
    # 对每一条act都认为是一次独立的判断
    for n in range(0, len(y_true)):
        for m in range(0, len(y_true[n])):
            if y_true[n][m] == 1 and y_pred[n][m] == 0:
                FN += 1
            elif y_true[n][m] == 1 and y_pred[n][m] == 1:
                TP += 1
            elif y_true[n][m] == 0 and y_pred[n][m] == 1:
                FP += 1
            elif y_true[n][m] == 0 and y_pred[n][m] == 0:
                TN += 1


    # # 对一个句子总体认为一次判断，要求act全对才算正确
    # for n in range(0, len(y_true)):
    #     for m in range(0, len(y_true[n])):
    #         if y_true[n][m] == 1 and y_pred[n][m] == 0:
    #             FN += 1
    #         elif y_true[n][m] == 1 and y_pred[n][m] == 1:
    #             TP += 1
    #         elif y_true[n][m] == 0 and y_pred[n][m] == 1:
    #             FP += 1

    if (TP+FP) != 0:
        recall = TP/(TP+FN)
    else:
        recall = 'NaN'
    if (TP+FN) != 0:
        precision = TP/(TP+FP)
    else:
        precision = 'NaN'
    if recall == 'NaN' or precision == 'NaN':
        f_measure = 'NaN'
    else:
        f_measure = 2 * precision * recall / (precision + recall)
    accuracy = (TP + TN)/(TP + TN + FP + FN)
    return accuracy, recall, precision, f_measure
