#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Edoardo Lenzi, Beatrice Portelli, Simone Scaboro'
__version__ = '1.0'
__license__ = '???'
__copyright__ = '???'


class TaggingRules(object):
    #self.notation2num = { 'O': 0, 'I': 1, 'B': 2, 'L': 3, 'U': 4 }

    @staticmethod
    def iob_adr_nonadr_to_index(tag):
        '''Numericalize string tag in the following way
        0 O I-Drug B-Disease ...
        1 I-ADR
        2 B-ADR'''
        if tag == 'I-ADR':
            return 2 
        elif tag == 'B-ADR':
            return 1 
        else:
            return 0


    @staticmethod
    def iob_adr_nonadr_to_tag(index):
        '''Reverse iob_adr_nonadr_to_index()'''
        if index == 2:
            return 'I-ADR' 
        elif index == 1:
            return 'B-ADR' 
        else:
            return 'O'


    @staticmethod
    def iob_drug_nondrug_to_index(tag):
        '''Numericalize string tag in the following way
        0 O I-ADR B-Disease ...
        1 I-Drug
        2 B-Drug'''
        if tag == 'I-Drug':
            return 2 
        elif tag == 'B-Drug':
            return 1 
        else:
            return 0


    @staticmethod
    def iob_drug_nonadr_to_tag(index):
        '''Reverse iob_drug_nondrug_to_index()'''
        if index == 2:
            return 'I-Drug' 
        elif index == 1:
            return 'B-Drug' 
        else:
            return 'O'