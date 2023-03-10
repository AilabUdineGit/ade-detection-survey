#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Edoardo Lenzi, Beatrice Portelli, Simone Scaboro'
__version__ = '1.0'
__license__ = '???'
__copyright__ = '???'


from ade_detection.domain.enums import *


class ExporterConfig(object):


    def __init__(self, dataset:CORPUS, split: SPLIT, path: str, 
                 train_percentage: float, test_percentage: float):
        self.dataset = dataset
        self.path = path
        self.train_percentage = train_percentage
        self.test_percentage = test_percentage
           

    def __eq__(self, other):
        return self.train_percentage == other.train_percentage and \
               self.test_percentage == other.test_percentage and \
               self.path == other.path and \
               self.dataset == other.dataset