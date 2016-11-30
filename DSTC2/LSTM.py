# -*- coding:utf-8 -*-  

from sklearn.cross_validation import train_test_split

from DSTC2.traindev.scripts import myLogger
from DSTC2.traindev.scripts.model import LSTM
from traindev.scripts import file_reader
from traindev.scripts import initializer
from traindev.scripts.initializer import Set
from traindev.scripts.judge import recall_precision_F

__author__ = "JOHNKYON"

global logger

if __name__ == "__main__":
    global logger
    logger = myLogger.myLogger("basic")
    logger.info("Starting basic")
    # 选择模式
    dataset = file_reader.get_dataset("dstc2_debug")
    logger.info("token check test begin")
    raw = initializer.raw_initializer(dataset)
    # Build token and dictionary
    token = initializer.token_initializer(raw["input"])
    dictionary = initializer.dictionary_initializer(token)
    # Build input vector
    one_set = Set(token, dictionary, raw["output"])
    input_mtr, output_mtr = LSTM.basic_LSTM_init(one_set.input_mtr, one_set.output_mtr)
    # TODO:设计降维算法，需要将输出的维度降低。
    # get model
    # model = LSTM.get_basic_LSTM(one_set.sentence_length, len(one_set.act_dict))
    model = LSTM.get_LSTM(one_set.sentence_length, len(one_set.act_dict))
    # train
    X_train, X_test, y_train, y_test = train_test_split(input_mtr, output_mtr, test_size=0.2, random_state=True)
    model.fit(X_train, y_train, batch_size=16, nb_epoch=8)
    # test
    print model.evaluate(X_test, y_test, batch_size=2)
    y_pre = model.predict(X_test, batch_size=16)
    print("[recall: %s,\tprecision: %s,\tf_measure: %s]", recall_precision_F(y_test, y_pre))

