# -*- coding:utf-8 -*-
from gensim import corpora
import nltk
import myLogger
import numpy as np

__author__ = "JOHNKYON"


def raw_initializer(dataset):
    """
    Read input and output information from json object
    :param dataset:
    :return:
    """
    logger = myLogger.myLogger("initializer")
    logger.info("Starting raw initializing")
    input_raw = []
    output_raw = []
    for call in dataset:
        input_row = []
        output_row = []
        # if call.log["session-id"] == "voip-f32f2cfdae-130328_192703":
        for turn, label in call:
            input_row.append(turn["output"]["transcript"].lower())
            output_row.append(turn["output"]["dialog-acts"])
            input_row.append(label["transcription"].lower())
            output_row.append(label["semantics"]["json"])
        input_raw.append(input_row)
        output_raw.append(output_row)
    logger.info("Finish raw initializing")
    print(len(input_raw))
    return {"input": input_raw, "output": output_raw}


def token_initializer(data):
    """
    Translate text from input into token
    :param data:
    :return:
    """
    logger = myLogger.myLogger("Token initializer")
    logger.info("Starting tokenizing")
    token = map(lambda element: map(lambda x: nltk.word_tokenize(x.lower()), element), data)
    logger.info("Tokenizing finished")
    return token


def dictionary_initializer(token):
    """
    Build dictionary with token
    :param token:
    :return:
    """
    logger = myLogger.myLogger("Dictionary initializer")
    logger.info("Starting building dictionary")
    raw = map(lambda element: reduce(lambda x, y: x + y, element), token)
    dictionary = corpora.Dictionary(raw)
    logger.info("Finish building dictionary")
    return dictionary


def label_dict(output):
    """
    Use output to create label dictionary and output vector
    :param output:
    :return:
    """
    act_dict = {}
    slot_dict = {}
    label = []
    act_count = 0
    slot_count = 0
    for session in output:
        act_session = []
        slot_session = []
        for sentence in session:
            act_sentence = []
            slot_sentence = []
            for dic in sentence:
                if act_dict.has_key(dic["act"]):
                    act_sentence.append(act_dict[dic["act"]])
                else:
                    act_sentence.append(act_count)
                    act_dict[dic["act"]] = act_count
                    act_count += 1
                for slot in dic["slots"]:
                    if slot_dict.has_key(tuple(slot)):
                        slot_sentence.append(slot_dict[tuple(slot)])
                    else:
                        slot_sentence.append(slot_count)
                        slot_dict[tuple(slot)] = slot_count
                        slot_count += 1
            act_session.append(act_sentence)
            slot_session.append(act_sentence)
        label.append({"act": act_session, "slot": slot_session})
    return label, act_dict, slot_dict

class Set:
    def __init__(self, token, dictionary, output, mode):
        logger = myLogger.myLogger("Input layer initializer")
        logger.info("Initializing input raw")
        self.input = map(lambda call: map(lambda sentence: map(lambda x: dictionary.token2id[x], sentence), call), token)
        self.output, self.act_dict, self.slot_dict = label_dict(output)
        self.sentence_length = 0
        self.sentence_count = 0
        for session in self.input:
            self.sentence_count = max(self.sentence_count, len(session))
            for sentence in session:
                self.sentence_length = max(self.sentence_length, len(sentence))

        # 初始化ndarray
        self.input_mtr = np.zeros((len(self.input), self.sentence_count, self.sentence_length))
        self.output_mtr = np.zeros((len(self.input), self.sentence_count, len(self.act_dict)))
        for session_index in range(0, len(self.input)):
            for sentence_index in range(0, len(self.input[session_index])):
                # 此处仅记录act的label
                for n in range(0, len(self.input[session_index][sentence_index])):
                    self.input_mtr[session_index][sentence_index][n] = self.input[session_index][sentence_index][n]
                if mode == 1:
                    for n in self.output[session_index]["act"][sentence_index]:
                        self.output_mtr[session_index][sentence_index][n] = 1
                elif mode == 2:
                    for n in self.output[session_index]["slot"][sentence_index]:
                        self.output_mtr[session_index][sentence_index][n] = 1

    def __iter__(self):
        for session_index in range(0, len(self.input)):
            for sentence_index in range(0, len(self.input[session_index])):
                vector = {"input": self.input_mtr[session_index, sentence_index], "output": self.output_mtr[session_index, sentence_index]}
                yield vector