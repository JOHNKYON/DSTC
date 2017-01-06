# -*- coding:utf-8 -*-  
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.recurrent import LSTM
from DSTC2.traindev.scripts import myLogger
from keras.layers.pooling import GlobalMaxPooling1D
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers import Embedding
from keras.datasets import imdb
from keras.utils.visualize_util import plot
import numpy as np

__author__ = "JOHNKYON"


def basic_cnn_LSTM_init(input_mtr, output_mtr):
    return input_mtr, output_mtr


def output_shape(y_train, y_test):
    y_train = np.array(map(lambda session: reduce(lambda sentence1, sentence2: np.hstack((sentence1, sentence2)), session), y_train))
    y_test = np.array(map(lambda session: reduce(lambda sentence1, sentence2: np.hstack((sentence1, sentence2)), session), y_test))
    return y_train, y_test


def get_mixed(input_shape, output_dimension):
    logger = myLogger.myLogger('mixed')
    logger.info('Building mixed model')

    # Convolution
    filter_length = 5
    nb_filter = 16
    pool_length = 4

    # LSTM
    lstm_output_size = 1024

    # Training

    layer = 3
    hidden_size = 1024
    model = Sequential()
    model.add(Convolution1D(nb_filter=nb_filter,
                            filter_length=filter_length,
                            border_mode='valid',
                            activation='relu',
                            subsample_length=1,
                            input_shape=input_shape))
    model.add(MaxPooling1D(pool_length=pool_length))
    model.add(LSTM(lstm_output_size, dropout_W=0.2, dropout_U=0.2))
    model.add(Dense(output_dimension))

    model.compile(loss='binary_crossentropy',
                  optimizer='RMSprop',
                  metrics=['accuracy'])
    plot(model, to_file="cnn_lstm.jpg")
    return model


def sample_mixed():
    # Embedding
    max_features = 20000
    maxlen = 100
    embedding_size = 128

    # Convolution
    filter_length = 5
    nb_filter = 64
    pool_length = 4

    # LSTM
    lstm_output_size = 70

    # Training
    batch_size = 30
    nb_epoch = 2

    '''
    Note:
    batch_size is highly sensitive.
    Only 2 epochs are needed as the dataset is very small.
    '''

    print('Loading data...')
    (X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features)
    print(len(X_train), 'train sequences')
    print(len(X_test), 'test sequences')

    print('Pad sequences (samples x time)')
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    print('Build model...')

    model = Sequential()
    model.add(Embedding(max_features, embedding_size, input_length=maxlen))
    model.add(Dropout(0.25))
    model.add(Convolution1D(nb_filter=nb_filter,
                            filter_length=filter_length,
                            border_mode='valid',
                            activation='relu',
                            subsample_length=1))
    model.add(MaxPooling1D(pool_length=pool_length))
    model.add(LSTM(lstm_output_size))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='cosine_proximity',
                  optimizer='adam',
                  metrics=['accuracy'])

    print('Train...')
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              validation_data=(X_test, y_test))
    score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)
