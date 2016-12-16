# -*- coding:utf-8 -*-  

from __future__ import print_function
from __future__ import print_function
from sklearn.cross_validation import train_test_split

from DSTC2.traindev.scripts import myLogger
from DSTC2.traindev.scripts.model import LSTM
from traindev.scripts import file_reader
from traindev.scripts import initializer
from traindev.scripts.initializer import Set
from traindev.scripts.judge import recall_precision_F
import codecs
import matplotlib.pyplot as plt

__author__ = "JOHNKYON"

global logger

if __name__ == "__main__":
    # Choose mode
    # 1:act 2:slot
    mode = 1

    global logger
    logger = myLogger.myLogger("basic")
    logger.info("Starting basic")
    record_file = codecs.open("LSTM_threshold_record.txt", 'wb+', encoding='utf8')
    threshold = 0.1
    plt.figure(1)
    # ready to plot
    thres = []
    f_m = []
    # 选择模式
    roll = range(0, 16)
    for n in roll:
        dataset = file_reader.get_dataset("dstc2_debug")
        logger.info("token check test begin")
        raw = initializer.raw_initializer(dataset)
        # Build token and dictionary
        token = initializer.token_initializer(raw["input"])
        dictionary = initializer.dictionary_initializer(token)
        # Build input vector
        one_set = Set(token, dictionary, raw["output"], mode)
        input_mtr, output_mtr = LSTM.basic_LSTM_init(one_set.input_mtr, one_set.output_mtr)


        # TODO:设计降维算法，需要将输出的维度降低。
        # get model
        # model = LSTM.get_basic_LSTM(one_set.sentence_length, len(one_set.act_dict))
        model = LSTM.get_LSTM(one_set.sentence_length, len(one_set.act_dict))


        # train
        X_train, X_test, y_train, y_test = train_test_split(input_mtr, output_mtr, test_size=0.2, random_state=42)
        model.fit(X_train, y_train, batch_size=16, nb_epoch=12)


        # test
        print(model.evaluate(X_test, y_test, batch_size=2))
        y_pre = model.predict(X_test, batch_size=16)
        # print("[recall: {0},\tprecision: {1},\tf_measure: {2}]".format(recall_precision_F(y_test, y_pre)))


        f_measure_old = 0
        accuracy, recall, precision, f_measure = recall_precision_F(y_test, y_pre, threshold)
        f_m.append(f_measure)
        thres.append(threshold)
        record_file.write('threshold: {0}'.format(threshold))
        threshold += (f_measure - f_measure_old) / 200


        print("[accuracy: {4}, recall: {0},\tprecision: {1},\tf_measure: {2}, threshold: {3}]".format(recall, precision, f_measure, threshold, accuracy))
        record_file.write("[accuracy: {3}, recall: {0},\tprecision: {1},\tf_measure: {2}]\n\n".format(recall, precision, f_measure, accuracy))
        print("--------------------------------------------------------------")
    plt.plot(roll, f_m, 'b', roll, thres, 'r')
    plt.savefig('figure.jpg')
    record_file.close()

