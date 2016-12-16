# -*- coding:utf-8 -*-  
from keras.utils.visualize_util import plot
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Lambda
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
import numpy as np
from keras import backend as K
from keras.engine.topology import Layer
from DSTC2.traindev.scripts import myLogger


__author__ = "JOHNKYON"


def get_LSTM(sentence_length, output_dimension):
    """
    build and get a LSTM based model
    :param sentence_length:
    :param output_dimension:
    :return:
    """
    logger = myLogger.myLogger('LSTM')
    logger.info('Building LSTM model')
    layer = 3
    hidden_size = 32
    model = Sequential()
    # model.add(LSTM(output_dimension, input_dim=sentence_length, input_length=3, dropout_U=0.1, dropout_W=0.1, return_sequences=True))
    model.add(LSTM(hidden_size, input_dim=1, input_length=sentence_length, dropout_U=0.1, dropout_W=0.1, return_sequences=True))
    model.add(LSTM(hidden_size, dropout_W=0.2, dropout_U=0.2, return_sequences=True))
    model.add(TimeDistributed(Dense(hidden_size)))
    model.add(Activation('softmax'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(output_dimension))
    # model.add(Lambda(lambda x: 1 * (x > 0.45)))
    # model.add(Thresholded(theta=0.45))
    # sigmoid函数反而降低了各节点之间的差异性
    # model.add(Activation('sigmoid'))
    # model.add(Dense(output_dimension))
    # model.compile(loss="categorical_crossentropy", optimizer='sgd', metrics=['accuracy'])
    model.compile(loss="msle", optimizer='RMSprop', metrics=['accuracy', 'sparse_categorical_accuracy'])
    plot(model, to_file="lstm.png")
    logger.info('Building finished')
    return model


def get_basic_LSTM(sentence_length, output_dimension):
    """
    test of build and get a basic LSTM base model
    :param sentence_length:
    :param output_dimension:
    :return:
    """
    model = Sequential()
    # model.add(LSTM(output_dimension, input_dim=sentence_length, input_length=3, dropout_U=0.1, dropout_W=0.1))
    model.add(LSTM(output_dimension, input_dim=1, input_length=sentence_length, dropout_U=0.1, dropout_W=0.1))
    model.compile(loss="mse", optimizer='RMSprop', metrics=['accuracy'])
    # model.compile(loss="'mean_squared_error", optimizer='sgd', metrics=['accuracy'])
    plot(model, to_file="lstm.png")
    return model


def basic_LSTM_init(input_mtr, output_mtr):
    input_mtr = reduce(lambda session1, session2: np.vstack((session1, session2)), input_mtr)
    input_mtr = np.array(map(lambda sentence: np.array(map(lambda word: np.array([word]), sentence)), input_mtr))
    # input_mtr = np.array(map(lambda s: np.vstack((s, s, s)), input_mtr))
    output_mtr = reduce(lambda session1, session2: np.vstack((session1, session2)), output_mtr)
    return input_mtr, output_mtr


class Thresholded(Layer):
    '''Threshold Unit:
    `f(x) = 1 for x > theta`
    `f(x) = 0 otherwise`.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as the input.

    # Arguments
        theta: float >= 0. Threshold location of activation.

    '''
    def __init__(self, theta=1.0, **kwargs):
        self.supports_masking = True
        self.theta = K.cast_to_floatx(theta)
        super(Thresholded, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return 1 * K.cast(x > self.theta, K.floatx())

    def get_config(self):
        config = {'theta': float(self.theta)}
        base_config = super(Thresholded, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))