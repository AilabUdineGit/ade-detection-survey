#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Edoardo Lenzi, Beatrice Portelli, Simone Scaboro'
__version__ = '1.0'
__license__ = '???'
__copyright__ = '???'


from ade_detection.utils.logger import Logger
LOG = Logger.getLogger(__name__)
import zipfile
import pandas as pd
import numpy as np
import tweepy
import os

from ade_detection.utils.env import Env


class TwitterService(object):


    def __init__(self):
        credentials = Env.load_credentials()

        # Autenticate
        auth = tweepy.OAuthHandler(credentials['CONSUMER_KEY'], credentials['CONSUMER_SECRET'])
        auth.set_access_token(credentials['ACCESS_KEY'], credentials['ACCESS_SECRET'])
        self.api = tweepy.API(auth,wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

        if(self.api.verify_credentials):
            pass #LOG.info('Logged in successfully')
        else:
            raise PermissionError('Twitter rejects your credentials')


    def get_text(self, tweet_id: str) -> str:
        try:
           return self.api.get_status(tweet_id).text
        except Exception as e:
            # private tweet
            return None