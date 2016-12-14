# -*- coding:utf-8 -*-

import nltk
from gensim import corpora
import myLogger

__author__ = "JOHNKYON"


def input_tokenize(sentence):
    logger = myLogger.myLogger("tokenize test")
    tokens = nltk.word_tokenize(sentence)
    dictionary = corpora.Dictionary(tokens)
    print dictionary.token2id
    logger.info("dictionary test finished")


