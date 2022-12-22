#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Edoardo Lenzi, Beatrice Portelli, Simone Scaboro'
__version__ = '1.0'
__license__ = '???'
__copyright__ = '???'


from ade_detection.utils.logger import Logger
LOG = Logger.getLogger(__name__)
from tqdm import tqdm

from ade_detection.services.database_service import DatabaseService
from ade_detection.domain.annotation import Annotation
from ade_detection.domain.attribute import Attribute
from ade_detection.domain.document import Document
from ade_detection.domain.interval import Interval
from ade_detection.domain.span import Span
import ade_detection.utils.localizations as loc
from ade_detection.domain.enums import *
from ade_detection.utils.env import Env


class TensorExporter(object):

    '''Exporter script to tensor dataset
    '''

    def __init__(self):
        #db = DatabaseService()
        pass

    def tagging(self, text, annotations):
        tags = []
        for i, t in text.iterrows():
            a = annotations[['begin','end','type']].loc[annotations['text_id']==t['text_id']]
            (tokens, biluo_tags) = self.biluo_tagging(t['text'], [tuple(x) for x in a.to_numpy()])
            tag = Tag(i, tokens, biluo_tags.copy())
            tag.iob = self.biluo_to_iob(biluo_tags.copy())
            tag.io = self.biluo_to_io(biluo_tags.copy())
            tags.append(tag)

        text['tokens'] = text['biluo_tags'] = text['io_tags'] = text['iob_tags'] = np.nan
        for t in tags:
            text['tokens'].iloc[t.text_id] = t.tokens
            text['biluo_tags'].iloc[t.text_id] = t.biluo
            text['iob_tags'].iloc[t.text_id] = t.iob
            text['io_tags'].iloc[t.text_id] = t.io
        return text


    def biluo_tagging(self, text, annotations):
        doc = self.nlp(text)
        annotations = self.trim_annotations(annotations, text)
        biluo_tags = biluo_tags_from_offsets(doc, annotations)
        tokens = [str(token) for token in doc]
        (tokens, biluo_tags) = self.postprocessing_0W30(tokens, biluo_tags, text, annotations)
        tokens = self.postprocessing_meaningless_tags(tokens)
        return (tokens, biluo_tags)


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