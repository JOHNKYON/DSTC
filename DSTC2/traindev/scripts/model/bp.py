# -*- coding:utf-8 -*-  

from keras.utils.visualize_util import plot
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np


__author__ = "JOHNKYON"


def bp_builder(input_dimension, output_dimension):
    model = Sequential()
    model.add(Dense(output_dim=2048, input_dim=input_dimension))
    model.add(Activation("relu"))
    model.add(Dense(output_dim=1024))
    model.add(Activation("relu"))
    model.add(Dense(output_dim=512))
    model.add(Activation("relu"))
    model.add(Dense(output_dim=output_dimension))
    model.add(Activation("softmax"))
    model.compile(loss="categorical_crossentropy", optimizer='sgd', metrics=['accuracy'])
    plot(model, to_file="bp.png")
    return model


def bp_train(x_train, y_train, model):
    model.fit(x_train, y_train, nb_epoch=5, batch_size=32)
    return model


def bp_initialize(input_mtr, output_mtr):
    input_mtr = np.array(map(lambda session: reduce(lambda sentence1, sentense2: np.hstack((sentence1, sentense2)), session), input_mtr))
    output_mtr = np.array(map(lambda session: reduce(lambda sentence1, sentense2: np.hstack((sentence1, sentense2)), session), output_mtr))
    return input_mtr, output_mtr
