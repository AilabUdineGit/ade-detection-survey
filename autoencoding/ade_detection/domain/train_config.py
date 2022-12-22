#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Edoardo Lenzi, Beatrice Portelli, Simone Scaboro'
__version__ = '1.0'
__license__ = '???'
__copyright__ = '???'


from ade_detection.domain.enums import *


class TrainConfig(object):


    def __init__(self, max_patience: int, learning_rate: float, dropout: float,
                 epochs: int, batch_size: int, random_seed: float, epsilon: float = 1e-8):
        self.max_patience = max_patience
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.epochs = epochs
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.random_seed = random_seed
           

    def __eq__(self, other):
        return self.max_patience == other.max_patience and \
               self.learning_rate == other.learning_rate and \
               self.dropout == other.dropout and \
               self.epochs == other.epochs and \
               self.epsilon == other.epsilon and \
               self.batch_size == other.batch_size and \
               self.random_seed == other.random_seed 