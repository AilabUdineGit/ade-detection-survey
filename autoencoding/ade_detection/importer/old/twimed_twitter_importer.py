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
import glob
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


class TwimedTwitterImporter(BaseImporter):

    '''Importer script for the Twimed-Twitter dataset
    '''

    def __init__(self):
        self.ts = TwitterService()
        db = DatabaseService()
        if not os.path.exists(loc.TWIMED_ZIP_PATH):
            fm.wget_with_progressbar(loc.TWIMED_GIT_LINK, loc.TWIMED_ZIP_PATH)
        self.decompress_dataset(loc.TWIMED_ZIP_PATH, loc.TWIMED_ARCHIVE_PATH)
        
        (corpus, annotations) = self.load_dataset()

        session = db.new_session()
        documents = self.encode_dataset(corpus, annotations)
        session.add_all(documents)
        session.commit()
        #LOG.info('dataset stored in the database successfully!...')
        

    def encode_dataset(self, corpus, annotations):
        documents = []
        #LOG.info('dataset serialization in progress...')
        for _, row in tqdm(corpus.iterrows()):
            docs = list(filter(lambda x: x.external_id == row.text_id, documents))
            if len(docs) == 0:
                doc = Document(external_id = row.text_id, 
                               text = row.text, 
                               corpus = CORPUS.TWIMED_TWITTER)
                documents.append(doc)
            else: 
                doc = docs[0]

            for _, span in annotations[annotations.text_id == row.text_id].iterrows():
                annotation = span.annotation.split(' ')
                interval = Interval(begin = annotation[1], end = annotation[2])
                span_annotations = [Annotation(key = annotation[0], value = span.span),
                                    Annotation(key = 'id', value = span.id)]
                doc.spans.append(Span(intervals = [interval], annotations=span_annotations))

        return documents


    def load_dataset(self):
        #LOG.info('dataset loading in progress...')
        filenames = glob.glob(loc.TWIMED_TWITTER_QUERY)
        corpus = annotations = pd.DataFrame({})

        for filename in tqdm(filenames):
            text_id = os.path.basename(filename).replace('.ann', '')
            text = self.load_text(text_id)
            if text is not None:
                corpus = pd.concat([corpus, text], axis=0).reset_index(drop=True)
                df = self.load_annotations(filename, text_id)
                if not df.empty:
                    annotations = pd.concat([annotations, df], axis=0).reset_index(drop=True)
        #LOG.info('dataset loaded successfully!')
        return (corpus, annotations) 


    def load_annotations(self, filename, text_id):
        annotations = pd.read_csv(filename, sep='\t', header=None, 
                                  names=["id", "annotation", "span"])
        annotations['text_id'] = text_id
        return annotations


    def load_text(self, tweet_id):
        df = pd.DataFrame({'text_id': [''], 'text': ['']})
        if Env.get_value(Env.TWIMED_SOURCE) == 'Twitter':
            text = self.ts.get_text(tweet_id)
        elif Env.get_value(Env.TWIMED_SOURCE) == 'LocalStorage':
            text_path = loc.abs_path([loc.ASSETS, loc.CORPUSS, loc.TWIMED, 
                                      loc.TWIMED_TWITTER_TEXTS, tweet_id+'.txt'])
            if os.path.exists(text_path):
                text = fm.from_txt(text_path)
            else: 
                return None
        if text is None:
            return None
        df.iloc[0] = [tweet_id, text.replace('\n', ' ')]
        return df
