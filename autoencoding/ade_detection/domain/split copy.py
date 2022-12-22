#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Edoardo Lenzi, Beatrice Portelli, Simone Scaboro'
__version__ = '1.0'
__license__ = '???'
__copyright__ = '???'


from ade_detection.domain.train_config import TrainConfig
from ade_detection.domain.enums import *


class Split(object):

    train: list #<SubDocuments> 
    test: list #<SubDocuments> 
    validation: list #<SubDocuments> 