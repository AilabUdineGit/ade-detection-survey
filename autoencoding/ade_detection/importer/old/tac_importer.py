#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Edoardo Lenzi, Beatrice Portelli, Simone Scaboro'
__version__ = '1.0'
__license__ = '???'
__copyright__ = '???'


from ade_detection.utils.logger import Logger
LOG = Logger.getLogger(__name__)
from pandas.core.frame import DataFrame
from tqdm import tqdm
import pandas as pd 
import numpy as np
from os import path
import os

from ade_detection.services.database_service import DatabaseService
from ade_detection.services.twitter_service import TwitterService
from ade_detection.importer.base_importer import BaseImporter
from ade_detection.domain.annotation import Annotation
from ade_detection.domain.attribute import Attribute
from ade_detection.domain.document import Document
from ade_detection.domain.interval import Interval
from ade_detection.domain.span import Span
import ade_detection.utils.localizations as loc
import ade_detection.utils.file_manager as fm
from ade_detection.domain.enums import *
from ade_detection.utils.env import Env


class TacImporter(BaseImporter):

    '''Importer script for the ASU-TAC dataset
    see also http://diego.asu.edu/downloads/twitter_annotated_corpus/
    '''

    def __init__(self):
        self.ts = TwitterService()
        db = DatabaseService()
        
        self.decompress_dataset(loc.TAC_ZIP_PATH, loc.TAC_ARCHIVE_PATH)
        (train_corpus, test_corpus, 
         train_annotations, test_annotations) = self.load_dataset()
        
        #LOG.info('split in progress...')
        if not path.exists(loc.TAC_SPLIT_PATH):
            os.mkdir(loc.TAC_SPLIT_PATH)
        train_ids = train_corpus.tweet_id.unique()
        test_ids = test_corpus.tweet_id.unique()
        split_idx = int(len(train_ids) * 0.7)
        fm.to_id(train_ids[:split_idx], path.join(loc.TAC_SPLIT_PATH, loc.TRAIN_ID))
        fm.to_id(test_ids, path.join(loc.TAC_SPLIT_PATH, loc.TEST_ID))
        fm.to_id(train_ids[split_idx:], path.join(loc.TAC_SPLIT_PATH, loc.VALIDATION_ID))

        #LOG.info('split saved successfully!')
        corpus = pd.concat([train_corpus, test_corpus]).reset_index(drop=True)
        annotations = pd.concat([train_annotations, test_annotations]).reset_index(drop=True)
        session = db.new_session()
        documents = self.encode_dataset(corpus, annotations)
        session.add_all(documents)
        session.commit()
        #LOG.info('dataset stored in the database successfully!...')


    def encode_dataset(self, corpus, annotations):
        documents = []
        #LOG.info('dataset serialization in progress...')
        for _, row in tqdm(corpus.iterrows()):
            docs = list(filter(lambda x: x.external_id == row.tweet_id, documents))
            if len(docs) == 0:
                doc = Document(external_id = row.tweet_id, 
                               text = row.text, 
                               corpus = CORPUS.TAC,
                               attributes = [Attribute(key='user_id', value=row.user_id),
                                             Attribute(key='text_id', value=row.text_id)])
                documents.append(doc)
            else: 
                doc = docs[0]

            for _, span in annotations[annotations.text_id == row.text_id].iterrows():
                interval = Interval(begin = span.begin, end = span.end)
                span_annotations = [Annotation(key = annotation_by_name(span.type), value = span.span),
                                    Annotation(key = annotation_by_name('related_drug'), value = span.related_drug),
                                    Annotation(key = annotation_by_name('target_drug'), value = span.target_drug)]
                doc.spans.append(Span(intervals = [interval], annotations=span_annotations))

        return documents


    def load_dataset(self):
        #LOG.info('dataset loading in progress...')
        (train_annotations, test_annotations) = self.load_annotations()
        (train_corpus, test_corpus) = self.load_corpus()
        if Env.get_value(Env.TAC_SOURCE) == Env.TWITTER:
            train_corpus = self.load_text(train_corpus)
            test_corpus = self.load_text(test_corpus)
        else:
            train_corpus = self.load_text(train_corpus, fm.from_pickle(loc.OLD_TAC_TRAIN_PICKLE_PATH))
            test_corpus = self.load_text(test_corpus, fm.from_pickle(loc.OLD_TAC_TEST_PICKLE_PATH))
        #LOG.info('dataset loaded successfully!...')
        return (train_corpus, test_corpus, train_annotations, test_annotations)


    def load_annotations(self):
        names = ['text_id', 'begin', 'end', 'type', 'span', 'related_drug', 'target_drug']
        train_annotations = pd.read_csv(loc.TAC_TRAIN_ANNOTATIONS_PATH, sep='\t', 
                                        header=None, names = names) 
        test_annotations = pd.read_csv(loc.TAC_TEST_ANNOTATIONS_PATH, sep='\t', 
                                       header=None, names = names) 
        return (train_annotations, test_annotations)


    def load_corpus(self):
        dtype = {'tweet_id': object, 'user_id' : object, 'text_id': object}
        names = ['tweet_id', 'user_id', 'text_id']
        train_corpus = pd.read_csv(loc.TAC_TRAIN_IDS_PATH, sep='\t', header=None,
                                    dtype = dtype, names = names) 
        test_corpus = pd.read_csv(loc.TAC_TEST_IDS_PATH, sep='\t', header=None, 
                                  dtype = dtype, names = names) 
        return (train_corpus, test_corpus)


    def load_text(self, corpus, old_corpus = None):
        for index, row in tqdm(corpus.iterrows()):
            if old_corpus is None:
                text = self.ts.get_text(str(row.tweet_id))
                text = np.nan if text is None else text
            else:
                text = old_corpus[old_corpus.doc_id == row.text_id]
                text = np.nan if text.empty else text.iloc[0].doc_text
            corpus.loc[index, 'text'] = text
            
        return corpus.dropna().reset_index(drop=True)