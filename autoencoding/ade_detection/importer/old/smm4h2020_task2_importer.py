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


class SMM4H2020T2Importer(BaseImporter):

    '''Importer script for the SMM4H19 - task2 dataset
    see also https://healthlanguageprocessing.org/smm4h19/challenge/
    '''

    def __init__(self):        
        db = DatabaseService()
        
        dataset = self.load_dataset()
        session = db.new_session()
        documents = self.encode_dataset(dataset)
        session.add_all(documents)
        session.commit()
        #LOG.info('dataset stored in the database successfully!...')


    def encode_dataset(self, dataset):
        documents = []
        #LOG.info('dataset serialization in progress...')
        for _, row in tqdm(dataset.iterrows(), total=len(dataset)):
            docs = list(filter(lambda x: x.external_id == row.tweet_id, documents))
            if len(docs) == 0:
                doc = Document(external_id = row.tweet_id, 
                               text = row.text, 
                               corpus = CORPUS.SMM4H20_TASK2,
                               attributes = [Attribute(key='is_ade', value=str(row.is_ade))])
                documents.append(doc)
            else: 
                doc = docs[0]
                
            if row.is_ade == 1:
                interval = Interval(begin = 0, end = len(row.text))
                annotations = [Annotation(key = annotation_by_name("ADR"), value = row.text)]
                doc.spans.append(Span(intervals = [interval], annotations=annotations))

        return documents


    def load_dataset(self):
        #LOG.info('dataset loading in progress...')

        df1 = pd.read_pickle("assets/datasets/2020_task2/new_samples_2020.pickle") 
        df2 = pd.read_pickle("assets/datasets/2020_task2/unlabeled_samples_2020.pickle") 
        df2["text"] = df2["tweet"]
        df2.drop("tweet", axis=1, inplace=True)
        df2["is_ade"] = -1
        
        df = pd.concat((df1, df2))
            
        return df
