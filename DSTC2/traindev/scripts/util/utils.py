# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import nested_scopes
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement


def dict_verse(dictionary):
    """
    Transe verse a dictionary
    Every value in the dictionary should be unique
    :param dictionary:
    :return:
    """
    verse = dict()
    for key in dictionary:
        verse[dictionary[key]] = key
    return verse
