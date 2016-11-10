# -*- coding:utf-8 -*-  

import logging

__author__ = "JOHNKYON"


def myLogger(name):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s:%(msecs)05.1f pid:%(process)d [%(levelname)s] (%(funcName)s) %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=name+'.log',
        filemode='a+')
    return logging.getLogger()
