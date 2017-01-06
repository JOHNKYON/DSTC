# -*- coding:utf-8 -*-  
import numpy as np
import random
from DSTC2.traindev.scripts.util.utils import dict_verse

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

    if (TP+FN) != 0:
        recall = TP/(TP+FN)
    else:
        recall = 'NaN'
    if (TP+FP) != 0:
        precision = TP/(TP+FP)
    else:
        precision = 'NaN'
    if recall == 'NaN' or precision == 'NaN':
        f_measure = 'NaN'
    else:
        f_measure = 2 * precision * recall / (precision + recall)
    accuracy = (TP + TN)/(TP + TN + FP + FN)
    return accuracy, recall, precision, f_measure


def reduction(dictionary, x, y_true, y_pred, y_dic):
    """
    Show vectorized x and y into sentence and real act
    :param dictionary:
    :param x:
    :param y_true:
    :param y_pred:
    :param y_dic:
    :return:
    """
    for n in range(0, 10):
        index = random.randint(0, len(y_true))
        # TODO: id2token not exist
        id2token = dict_verse(dictionary.token2id)
        sentence = ""
        for word in x[index]:
            sentence += id2token[word[0]]+" "
        print(sentence)

        # Reduct y
        y_index_true = []
        y_index_pred = []
        for n in range(0, len(y_true[index])):
            if y_true[index][n] == 1:
                y_index_true.append(n)
            if y_pred[index][n] == 1:
                y_index_pred.append(n)
        print("True")
        for key in y_dic:
            if y_dic[key] in y_index_true:
                print(key)

        print("Pred")
        for key in y_dic:
            if y_dic[key] in y_index_pred:
                print(key)