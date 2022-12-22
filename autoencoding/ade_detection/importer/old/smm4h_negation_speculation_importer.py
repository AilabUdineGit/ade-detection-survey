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


class SMM4H19NegSpecImporter(BaseImporter):

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
        for _, row in tqdm(dataset.iterrows()):
            docs = list(filter(lambda x: x.external_id == row.tweet_id, documents))
            if len(docs) == 0:
                doc = Document(external_id = row.tweet_id, 
                               text = row.tweet, 
                               corpus = CORPUS.SMM4H19_NEGSPEC,
                               attributes = [])
                documents.append(doc)
            else: 
                doc = docs[0]

            if not pd.isnull(row.begin) and not pd.isnull(row.end):
                interval = Interval(begin = int(row.begin), end = int(row.end))
                annotations = [Annotation(key = annotation_by_name("ADR"), value = row.extraction)]
                doc.spans.append(Span(intervals = [interval], annotations=annotations))

        return documents


    def load_dataset(self):
        #LOG.info('dataset loading in progress...')

        df_original = pd.read_pickle(loc.SMM4H19_NEG_SPEC_ORIGINAL_PATH) 
        df_neg = pd.read_pickle(loc.SMM4H19_NEG_SPEC_NEGATION_PATH) 
        df_spec = pd.read_pickle(loc.SMM4H19_NEG_SPEC_SPECULATION_PATH)

        dfs = [df_original, df_neg, df_spec]

        for idx, name in enumerate(["O", "N", "S"]):
            df = dfs[idx]
            df.index = range(len(df))
            df["begin"] = np.nan
            df["end"] = np.nan
            df["extraction"] = ""
            dup = df.duplicated(subset=None, keep='first')
            df = df[~dup]
            df.tweet_id = [name+"_"+str(i) for i in df.tweet_id]
            dfs[idx] = df

        del df_original, df_neg, df_spec

        df = pd.concat(dfs)
        df.index = range(len(df))
        df.to_pickle(loc.SMM4H19_NEG_SPEC_FULL_PATH)
            
        return df
