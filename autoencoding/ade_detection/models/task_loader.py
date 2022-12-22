#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Edoardo Lenzi, Beatrice Portelli'
__version__ = '1.0'
__license__ = '???'
__copyright__ = '???'


from ade_detection.utils.logger import Logger
LOG = Logger.getLogger(__name__)
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import precision_recall_fscore_support
from transformers import get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertConfig
from transformers import AutoTokenizer, AutoConfig
from transformers import AdamW
from os import path
import pandas as pd
import numpy as np
import random
import pickle
from torch import LongTensor
import torch
import gc
import os


import ade_detection.utils.localizations as loc
from ade_detection.domain.enums import *
import ade_detection.utils.file_manager as fm
from ade_detection.domain.task import Task
from ade_detection.domain.train_config import TrainConfig
from ade_detection.domain.span import Span
from ade_detection.domain.subtoken import SubToken
from ade_detection.domain.document import Document
from ade_detection.services.database_service import DatabaseService
from ade_detection.services.split_service import SplitService
from ade_detection.services.query_service import QueryService
from ade_detection.services.model_service import ModelService

class TaskLoader(object):


    def __init__(self, task: Task):
        self.task = task
        self.max_seq_len = MAX_SEQ_LEN[self.task.corpus]
        split_svc = SplitService()
        self.task.split = split_svc.load_split(self.task)
        
        model_svc = ModelService(task)

        self.tokenizer = model_svc.get_tokenizer()
        self.model_name = task.model
        
        #t = self.TMP(task.split.train)
        #t1 = self.TMP(task.split.test)
        #t2 = self.TMP(task.split.validation)
        #a = pd.concat((t,t1))
        #a = pd.concat((a,t2))
        #a.to_pickle("SMM4H.pkl")
        #print("FINISHED")
        #assert False
        self.task.split.train = self.load(task.split.train)
        self.task.split.test = self.load(task.split.test)
        self.task.split.validation = self.load(task.split.validation)

    def TMP(self, sdocs):
        """ DELETE THIS FUNCTION """
        print("Fixing spans...")
        for annotation_type in self.task.goal:
            if None in self.task.tidy_modes:
                break
            elif TIDY_MODE.MERGE_OVERLAPS in self.task.tidy_modes and \
               TIDY_MODE.SOLVE_DISCONTINUOUS in self.task.tidy_modes:
                
                sdocs = self.merge_discontinuous_overlaps(sdocs, annotation_type)
                sdocs = self.solve_discontinuous(sdocs, annotation_type)
                sdocs = self.merge_overlaps(sdocs, annotation_type)
            else:
                raise NotImplementedError('tidy mode combination not implemented') 
        print("Spans fixed!")
        i = 0
        while i < len(sdocs):
            sdoc = sdocs[i]
            if sdoc.subtokens == None:
                sdoc = self.subtokenize(sdoc, self.task)
                sdoc = self.biluo_tagging(sdoc, self.task)
                if self.task.notation == NOTATION.IOB:
                    sdoc.tags = self.biluo_to_iob(sdoc.tags)
                if self.task.notation == NOTATION.IO:
                    sdoc.tags = self.biluo_to_io(sdoc.tags)  
                if self.task.notation == NOTATION.BINARY:
                    sdoc.tags = self.biluo_to_io(sdoc.tags) 

            if len(sdoc.subtokens) > self.max_seq_len - 2: #TODO
                new_sdoc = sdoc.copy()
                new_sdoc.id = sdocs[-1].id + 1 
                (char_index, subtoken_index) = self.find_split_index(sdoc) 
                sdoc.subtokens = sdoc.subtokens[:subtoken_index]
                new_sdoc.subtokens = new_sdoc.subtokens[subtoken_index:]
                sdoc.tags = sdoc.tags[:subtoken_index]
                new_sdoc.tags = new_sdoc.tags[subtoken_index:]
                sdoc.text = sdoc.text[:char_index]
                new_sdoc.text = new_sdoc.text[char_index:]
                sdocs.append(new_sdoc)
                assert len(sdoc.subtokens) <= self.max_seq_len - 2
            i += 1

        df = {'text': [], 'token': [], 'spans':[], 'id': []}
        for _doc in sdocs:
            spans, tokens = [], []
            last_is_i = False
            last_is_b = False
            last_end = None
            for idx, (token,label) in enumerate(zip(_doc.subtokens, _doc.tags)):
                if idx == 0:
                    start = 0 if token.token.begin == 0 else token.token.begin-1
                if label[0] == 'O':
                    if last_is_i or last_is_b:
                        e = last_end
                        text = _doc.doc.text[s:e]
                        spans.append((s,e))
                        tokens.append(text)
                        last_is_b,last_is_i = False, False
                if label[0] == "B":
                    if last_is_b or last_is_i:
                        e = last_end
                        text = _doc.doc.text[s:e]
                        spans.append((s,e))
                        tokens.append(text)
                        last_is_b,last_is_i = False, False
                    s = token.token.begin-start
                    last_is_b = True
                    last_end = token.token.end-start
                if label[0] == "I":
                    last_is_i = True
                    last_end = token.token.end-start
            if last_is_b or last_is_i:
                e = last_end
                text = _doc.doc.text[s:e]
                spans.append((s,e))
                tokens.append(text)
            spans = list(set(spans))
            tokens = list(set(tokens))
            df['id'].append(str(_doc.doc.external_id))
            df['text'].append(_doc.text)
            df['token'].append("; ".join(tokens))
            df['spans'].append(spans)
        df = pd.DataFrame(df)
        df.set_index("id",inplace=True)
        return df

    def load(self, sdocs):
        print("Fixing spans...")
        for annotation_type in self.task.goal:
            if None in self.task.tidy_modes:
                break
            elif TIDY_MODE.MERGE_OVERLAPS in self.task.tidy_modes and \
               TIDY_MODE.SOLVE_DISCONTINUOUS in self.task.tidy_modes:
                sdocs = self.merge_discontinuous_overlaps(sdocs, annotation_type)
                sdocs = self.solve_discontinuous(sdocs, annotation_type)
                sdocs = self.merge_overlaps(sdocs, annotation_type)
            else:
                raise NotImplementedError('tidy mode combination not implemented') 
        print("Spans fixed!")
        i = 0
        while i < len(sdocs):
            sdoc = sdocs[i]
            if sdoc.subtokens == None:
                sdoc = self.subtokenize(sdoc, self.task)
                sdoc = self.biluo_tagging(sdoc, self.task)
                if self.task.notation == NOTATION.IOB:
                    sdoc.tags = self.biluo_to_iob(sdoc.tags)
                if self.task.notation == NOTATION.IO:
                    sdoc.tags = self.biluo_to_io(sdoc.tags)  
                if self.task.notation == NOTATION.BINARY:
                    sdoc.tags = self.biluo_to_io(sdoc.tags) 

            if len(sdoc.subtokens) > self.max_seq_len - 2: #TODO
                new_sdoc = sdoc.copy()
                new_sdoc.id = sdocs[-1].id + 1 
                (char_index, subtoken_index) = self.find_split_index(sdoc) 
                sdoc.subtokens = sdoc.subtokens[:subtoken_index]
                new_sdoc.subtokens = new_sdoc.subtokens[subtoken_index:]
                sdoc.tags = sdoc.tags[:subtoken_index]
                new_sdoc.tags = new_sdoc.tags[subtoken_index:]
                sdoc.text = sdoc.text[:char_index]
                new_sdoc.text = new_sdoc.text[char_index:]
                sdocs.append(new_sdoc)
                assert len(sdoc.subtokens) <= self.max_seq_len - 2
            i += 1
        
        for sdoc in sdocs:
            PAD = self.tokenizer.pad_token
            SEP = self.tokenizer.sep_token
            CLS = self.tokenizer.cls_token
            sdoc.subtokens.insert(0, SubToken(None, CLS))
            sdoc.tags.insert(0, 'O')
            sdoc.subtokens.append(SubToken(None, SEP))
            sdoc.tags.append('O')
            
            sdoc.subtokens.extend([SubToken(None, PAD)] * (self.max_seq_len - len(sdoc.subtokens)))
            sdoc.tags.extend(['O'] * (self.max_seq_len - len(sdoc.tags)))
            sdoc.attention_mask = [0 if x.text == PAD else 1 for x in sdoc.subtokens]
            sdoc.num_subtokens = self.tokenizer.convert_tokens_to_ids([x.text for x in sdoc.subtokens])

            sdoc.num_tags = self.convert_iob_tags_to_ids(sdoc.tags, sdoc.subtokens) #TODO
            
            #print(" ".join([t.text for t in sdoc.subtokens]))
            #print(sdoc.tags)
            #print(sdoc.num_tags)
            #print("--------------")
            
            assert len(sdoc.subtokens) == len(sdoc.tags)
            assert len(sdoc.attention_mask) == len(sdoc.tags)
            assert len(sdoc.num_tags) == len(sdoc.tags)
            assert len(sdoc.num_subtokens) == len(sdoc.subtokens)
        return sdocs


    def convert_iob_tags_to_ids(self, tags, subtokens):
        num_tags = []
        for i, t in enumerate(tags):
            if subtokens[i].text == self.tokenizer.pad_token:
                num_tags.append(-1)
            else:
                if t == 'O':
                    num_tags.append(0)
                else:
                    annotation = t[2:]
                    index = self.index_by_annotation(annotation)
                    if t[0] == 'B':
                        num_tags.append(index * 2 + 2)
                    elif t[0] == 'I':
                        num_tags.append(index * 2 + 1)
        return num_tags


    def index_by_annotation(self, annotation):
        for i, a in enumerate(self.task.goal):
            if a.name == annotation:
                return i 
        assert False


    def find_split_index(self, sdoc):
        max_seq_len = self.max_seq_len - 2
        split_candidate = sdoc.subtokens[max_seq_len]
        ends = list(filter(lambda x: x <= split_candidate.token.end, [x.end for x in sdoc.doc.sentences]))
        end = max(ends) if len(ends) > 0 else split_candidate.token.end 
        subtoken_index = self.nearest_subtoken(sdoc.subtokens, end)
        
        if max_seq_len >= subtoken_index and subtoken_index > 5: #and max_seq_len >= (len(sdoc.subtokens) - subtoken_index):
            pass
        else:
            print("@@@@ fixing")
            subtoken_index = max_seq_len
        
        if sdoc.doc.external_id == "1009625926571704321" and subtoken_index == 0:
            print("@@@@@@@@@@@@@@@")
            
            print("max_seq_len", max_seq_len)
            print("split_candidate", split_candidate.text)
            
            print(sdoc.doc.external_id)
            print(sdoc.text)
            
            print([t.text for t in sdoc.subtokens])
            #print(sdoc.tags)
            
            print(ends)
            print(end)
            print("---------------")
            print(subtoken_index)
            print("@@@@@@@@@@@@@@@")
            #exit()
        
        
            
        return (sdoc.subtokens[subtoken_index].token.end, subtoken_index)


    def nearest_subtoken(self, array, value):
            winner = 0
            best_delta = abs(array[0].token.end - value)
            for i, s in enumerate(array):
                delta = abs(s.token.end - value)
                if delta <= best_delta:
                    best_delta = delta
                    winner = i 
            return winner


    def biluo_tagging(self, sdoc, task):
        tags = ['O'] * len(sdoc.subtokens) 
        for span in sdoc.doc.spans:            
            for annotation_type in task.goal:
                if span.contains_annotation(annotation_type):
                    begin = span.tokens[0].subtokens_interval[0]
                    end = span.tokens[-1].subtokens_interval[1]
                    if begin == end - 1:
                        tags[begin] = 'U-' + annotation_type.name
                    else:
                        tags[begin] = 'B-' + annotation_type.name 
                        tags[end - 1] = 'L-' + annotation_type.name 
                        for i in range(begin + 1, end - 1):
                            tags[i] = 'I-' + annotation_type.name
        sdoc.tags = tags
        return sdoc


    def biluo_to_iob(self, biluo):
        for i, t in enumerate(biluo):
            if t[0:1] == 'U':
                biluo[i] = t.replace('U', 'B', 1)
            elif t[0:1] == 'L':
                biluo[i] = t.replace('L', 'I', 1)
        return biluo 


    def biluo_to_io(self, biluo):
        for i, t in enumerate(biluo):
            if t[0:1] == 'U':
                biluo[i] = t.replace('U', 'I', 1)
            elif t[0:1] == 'L':
                biluo[i] = t.replace('L', 'I', 1)
            elif t[0:1] == 'B':
                biluo[i] = t.replace('B', 'I', 1)
        return biluo 


    def subtokenize(self, sdoc, task):
        subtokens = []
        for token in sdoc.doc.tokens:
            begin = max(0, len(subtokens))
            for chunk in self.tokenizer.tokenize(token.text):
                subtokens.append(SubToken(token, chunk))
            end = max(0, len(subtokens))
            token.subtokens_interval = [begin, end]
        sdoc.subtokens = subtokens
        return sdoc


    def merge_discontinuous_overlaps(self, sdocs, annotation_type):
        #LOG.info("Merge discontinuous overlaps")
        for sdoc in sdocs:
            for span in sdoc.doc.spans:
                if span.contains_annotation(annotation_type):
                    for i in span.intervals:
                        for j in span.intervals:
                            if i != j and i.overlaps(j):
                                i.begin = min(i.begin, j.begin)
                                i.end = max(i.end, j.end)
                                span.intervals.remove(j)
        return sdocs


    def merge_overlaps(self, sdocs, annotation_type):
        #LOG.info("Merge overlaps")
        for sdoc in sdocs:
            for i in sdoc.doc.spans:
                for j in sdoc.doc.spans:
                    if i != j and i.contains_annotation(annotation_type) \
                              and j.contains_annotation(annotation_type) \
                              and len(i.intervals) == 1 \
                              and len(j.intervals) == 1:
                        if i != j and i.intervals[0].overlaps(j.intervals[0]):
                            i.intervals[0].begin = min(i.intervals[0].begin, j.intervals[0].begin)
                            i.intervals[0].end = max(i.intervals[0].end, j.intervals[0].end)
                            sdoc.doc.spans.remove(j)
        return sdocs


    def solve_discontinuous(self, sdocs, annotation_type):
        #LOG.info("Solve discontinuous")
        for sdoc in sdocs:
            for span in sdoc.doc.spans:
                if span.contains_annotation(annotation_type) and \
                   len(span.intervals) > 1:
                    new_spans = []
                    for interval in span.intervals:
                        new_spans.append(Span( document = span.document,
                                               document_id = span.document_id,
                                               annotations = span.annotations,
                                               intervals = [interval],
                                               tokens = span.document.tokens_in(interval) )) #TODO
                    sdoc.doc.spans.remove(span)
                    sdoc.doc.spans.extend(new_spans)
        return sdocs
