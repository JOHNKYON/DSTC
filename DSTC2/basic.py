# -*- coding:utf-8 -*-  

# import keras.utils.visualize_util
# from keras.models import Sequential
# from keras.layers import Dense, Activation
from DSTC2.traindev.scripts import myLogger
from traindev.scripts import file_reader
from traindev.scripts import initializer
from traindev.scripts.initializer import InputRaw

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
    input_raw = InputRaw(token, dictionary)
    for call in input_raw:
        print call

