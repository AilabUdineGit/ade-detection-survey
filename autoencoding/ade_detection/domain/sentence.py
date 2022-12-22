#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Edoardo Lenzi, Beatrice Portelli, Simone Scaboro'
__version__ = '1.0'
__license__ = '???'
__copyright__ = '???'


from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy import Column, ForeignKey, Integer, String, Text
from ade_detection.services import Base


class Sentence(Base):
    __tablename__ = 'sentences'
    id = Column(Integer, primary_key=True, autoincrement=True)

    document_id = Column(Integer, ForeignKey('documents.id'))
    document = relationship('Document', back_populates='sentences')
    
    begin = Column(Integer)
    end = Column(Integer)

    def __eq__(self, other):
        return self.id == other.id and \
               self.document_id == other.document_id and \
               self.document == other.document and \
               self.begin == other.begin and \
               self.end == other.end 