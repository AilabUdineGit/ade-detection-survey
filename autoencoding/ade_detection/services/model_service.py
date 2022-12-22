#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Edoardo Lenzi, Beatrice Portelli, Simone Scaboro'
__version__ = '1.0'
__license__ = '???'
__copyright__ = '???'


from ade_detection.utils.logger import Logger
LOG = Logger.getLogger(__name__)
from transformers import AutoConfig
from transformers import AutoTokenizer
from os import path
import os
os.environ['TRANSFORMERS_CACHE'] = '/mnt/HDD/transformers/'

from ade_detection.utils.env import Env
from ade_detection.domain.task import Task
from ade_detection.domain.train_config import TrainConfig
import ade_detection.utils.file_manager as fm
from ade_detection.domain.enums import *


class ModelService(object):


    def __init__(self, task:Task):
        self.task = task


    def get_config(self):
        num_labels = (2 * len(self.task.goal)) + 1
        if self.task.notation == NOTATION.BINARY:
            num_labels = 2
        overrides = {
            #'model_max_len': 512,
            #'max_len': 512,
            'num_labels': num_labels,
            'output_attentions' : False,
            'output_hidden_states' : False,
            'hidden_dropout_prob' : self.task.train_config.dropout,
            'attention_probs_dropout_prob' :  self.task.train_config.dropout
        } # note: max_position_embeddings = 64 causes model issues
        config = AutoConfig.from_pretrained(self.task.model.value, **overrides)
        #print(config)
        config.dropout = self.task.train_config.dropout
        config.num_labels = num_labels
        config.batch_size = self.task.train_config.batch_size #BATCH_SIZE[self.task.corpus]
        if hasattr(config, 'model'):
            config.base_model = config.model # ADDED FOR JUST_TESTING
        else:
            config.base_model = self.task.model.value
        config.model = self.task.model.value
        # MODIFICATO per BERTWeet
        #config.model_max_len=512
        #config.max_len=512
        #print(config)
        #print(config)
        return config


    def get_tokenizer(self):
        config = self.get_config()
        # original
        #return AutoTokenizer.from_pretrained(self.task.model.value, config=self.get_config(),cache_dir='transformers_cache/')
        # MODIFIED FOR JUST_TESTING
        # MODIFICATO per BERTWeet
        tok = AutoTokenizer.from_pretrained(config.base_model, config=config, cache_dir='transformers_cache/')
        return tok

    @staticmethod
    def get_bio_git_model():
        zip_path = loc.abs_path([loc.TMP, loc.BIO_BERT_ZIP])
        if not path.exists(zip_path):
            #LOG.info('Model download in progress...')
            fm.wget_with_progressbar(loc.BIO_BERT_GIT_LINK, zip_path)
            #LOG.info('Model download completed!')
        #LOG.info('Model decompression in progress...')
        fm.decompress_zip(zip_path)
        #LOG.info('Model decompression completed!')
