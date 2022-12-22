#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Edoardo Lenzi, Beatrice Portelli, Simone Scaboro'
__version__ = '1.0'
__license__ = '???'
__copyright__ = '???'


from torch.utils.data import TensorDataset
from torch import LongTensor
import torch

from ade_detection.domain.enums import *


class Split(object):


    def __init__(self, train: list, test: list, validation: list,
                 train_tensor = None, test_tensor = None, validation_tensor = None):
        self.train = train
        self.test = test
        self.validation = validation
        
        self.train_tensor = train_tensor
        self.test_tensor = test_tensor
        self.validation_tensor = validation_tensor
           

    def to_tensor_dataset(self, sdocs, binary=False):
        dataset = None
        if binary:
            dataset = TensorDataset(
                LongTensor([x.num_subtokens for x in sdocs]),
                LongTensor([x.attention_mask for x in sdocs]),
                LongTensor([1 if 1 in x.num_tags else 0 for x in sdocs]),
                LongTensor([x.id for x in sdocs])
            )
        else:
            dataset = TensorDataset(
                LongTensor([x.num_subtokens for x in sdocs]),
                LongTensor([x.attention_mask for x in sdocs]),
                LongTensor([x.num_tags for x in sdocs]),
                LongTensor([x.id for x in sdocs])
            )
        return dataset

    def to_tensor_dataset_t5(self, sdocs, tokenizer, t5=False):
        """
            TODO: add t5 correct label
        """
        dataset = None
        labels = []
        for doc in sdocs:
            last_is_ent = False
            sub_labels = []
            for label,ids in zip(doc.num_tags, doc.num_subtokens):
                #print(tokenizer.convert_tokens_to_ids("; "))
                if label == 0 and last_is_ent:
                    sub_labels.append(tokenizer.convert_tokens_to_ids("; "))
                    last_is_ent = False
                if label != 0:
                    last_is_ent = True
                    sub_labels.append(ids)
                else:
                    last_is_ent = False
            sub_labels.extend([tokenizer.encode(tokenizer.pad_token) for _ in range(len(doc.num_tags)+1-len(sub_labels))])
            labels.append(sub_labels)
        
        prefix_ids = [tokenizer.convert_tokens_to_ids("ade ner: ")] if t5 else []
        prefix_mask = [1]*len(prefix_ids)
        #print(set([len(prefix_ids + x.num_subtokens) for x in sdocs]))
        #print(set([len(prefix_mask + x.attention_mask) for x in sdocs]))    
        #print(set([len(label) for label in labels]))    
        
        dataset = TensorDataset(
            LongTensor([prefix_ids + x.num_subtokens for x in sdocs]),
            LongTensor([prefix_mask + x.attention_mask for x in sdocs]),
            LongTensor(labels),
            LongTensor([x.id for x in sdocs])
        )

        return dataset

    def __eq__(self, other):
        return self.train == other.train and \
               self.test == other.test and \
               self.validation == other.validation
