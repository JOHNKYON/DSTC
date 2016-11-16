# -*- coding:utf-8 -*-  

from sklearn.cross_validation import train_test_split

from DSTC2.traindev.scripts import myLogger
from DSTC2.traindev.scripts.model import bp
from traindev.scripts import file_reader
from traindev.scripts import initializer
from traindev.scripts.initializer import Set

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
    # get model
    model = bp.bp_builder(one_set.dimension * one_set.sentence_dim, len(one_set.act_dict) * one_set.sentence_dim)
    # train
    X_train, X_test, y_train, y_test = train_test_split(one_set.input_mtr, one_set.output_mtr, test_size=0.2)
    model.fit(X_train, y_train, batch_size=2, nb_epoch=5)
    # test
    print model.evaluate(X_test, y_test, batch_size=2)
