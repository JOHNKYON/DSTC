# -*- coding:utf-8 -*-  

# import keras.utils.visualize_util
# from keras.models import Sequential
# from keras.layers import Dense, Activation
from DSTC2.traindev.scripts import myLogger
from traindev.scripts import corpus
from traindev.scripts import file_reader

__author__ = "JOHNKYON"

global logger

if __name__ == "__main__":
    global logger
    logger = myLogger.myLogger("basic")
    logger.info("Starting basic")
    # 选择模式
    dataset = file_reader.get_dataset("dstc2_debug")
    logger.info("token check test begin")
    for call in dataset:
        # if call.log["session-id"] == "voip-f32f2cfdae-130328_192703":
        for turn, label in call:
            corpus.input_tokenize(turn["output"]["transcript"])
