#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Edoardo Lenzi, Beatrice Portelli, Simone Scaboro'
__version__ = '1.0'
__license__ = '???'
__copyright__ = '???'


from ade_detection.domain.enums import *
from ade_detection.domain.token import Token


class SubToken(object):


    def __init__(self, token: Token, text: str):
        self.token = token
        self.text = text
           

    def __str__(self):
        return f"Tok: {self.text}({self.token})"
    def __eq__(self, other):
        return self.token == other.token and \
               self.text == other.text 