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
from ade_detection.importer.base_importer import BaseImporter
import ade_detection.utils.localizations as loc
import ade_detection.utils.file_manager as fm
from ade_detection.domain.enums import *
from ade_detection.domain.annotation import Annotation
from ade_detection.domain.attribute import Attribute
from ade_detection.domain.document import Document
from ade_detection.domain.interval import Interval
from ade_detection.domain.span import Span


class SMM4H20Importer(BaseImporter):

    '''Importer script for the SMM4H19 - task2 dataset
    see also https://healthlanguageprocessing.org/smm4h19/challenge/
    '''

    def __init__(self):        
        db = DatabaseService()

        #self.decompress_dataset(loc.SMM4H19_ZIP_PATH, 
        #                        loc.SMM4H19_ARCHIVE_PATH)
        dataset = self.load_dataset()
        session = db.new_session()
        documents = self.encode_dataset(dataset)
        session.add_all(documents)
        session.commit()
        #LOG.info('dataset stored in the database successfully!...')


    def encode_dataset(self, dataset):
        documents = []
        #LOG.info('dataset serialization in progress...')
        for _, row in dataset.iterrows():
            docs = list(filter(lambda x: x.external_id == row.tweet_id, documents))
            if len(docs) == 0:
                doc = Document(external_id = row.tweet_id, 
                               text = row.tweet, 
                               corpus = CORPUS.SMM4H20,
                               attributes = [Attribute(key='drug', value=row.drug)])
                documents.append(doc)
            else: 
                doc = docs[0]

            if not pd.isnull(row.begin) and not pd.isnull(row.end):
                interval = Interval(begin = int(row.begin), end = int(row.end))
                annotations = [Annotation(key = annotation_by_name(row.type), value = row.extraction)]
                if not pd.isnull(row.meddra_code):
                    annotations.append(Annotation(key = annotation_by_name('meddra_code'), 
                                                  value = row.meddra_code))
                if not pd.isnull(row.meddra_term):
                    annotations.append(Annotation(key = annotation_by_name('meddra_term'), 
                                                  value = row.meddra_term))
                doc.spans.append(Span(intervals = [interval], annotations=annotations))
        
        print("SMM4H22 correctly encoded")
        return documents


    def load_dataset(self):
        #LOG.info('dataset loading in progress...')
        dtype = {'tweet_id': object, 'begin': object, 'end': object, 
                 'type': object, 'extraction': object, 'drug': object, 
                 'tweet': object, 'meddra_code': object, 'meddra_term': object}

        dataset_location = loc.SMM4H20_CLEAN_TRAIN_PATH
        print(f"Loading from: {dataset_location}")
        df = pd.read_csv(dataset_location, sep='\t', dtype=dtype) 
        df.index = range(len(df))
        dup = df.duplicated(subset=None, keep='first')
        return df[~dup]
