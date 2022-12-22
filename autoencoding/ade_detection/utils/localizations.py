#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Edoardo Lenzi, Beatrice Portelli, Simone Scaboro'
__version__ = '1.0'
__license__ = '???'
__copyright__ = '???'


import os
import json


'''Localization keys and files'''


def abs_path(relative_path: list) -> str:

    ''' given an array of localizations (relative path) 
    returns and absolute path starting from cwd '''

    abs_path = os.getcwd()
    for p in relative_path:
        abs_path = os.path.join(abs_path, p)
    if os.path.isdir(abs_path) and not os.path.exists(abs_path):
        os.mkdir(abs_path)
    return abs_path


# Localizations for files

TWITTER_API_KEY = 'twitter_dev_api_key.json'
BOT = 'bot.pickle'
CHATS = 'chats.json'
ADMINS = 'admins.json'

CADEC_ZIP = 'CADEC.v2.zip'
TAC_ZIP = 'download_tweets.zip'
TWIMED_ZIP = 'twimed.zip'
BIO_BERT_ZIP = 'biobert.zip'
SMM4H19_ZIP = 'SMM4H_2019_task2_dataset.zip'

TAC_TEST_ANNOTATIONS = 'test_tweet_annotations.tsv'
TAC_TRAIN_ANNOTATIONS = 'train_tweet_annotations.tsv'
TAC_TEST_IDS = 'test_tweet_ids.tsv'
TAC_TRAIN_IDS = 'train_tweet_ids.tsv'

OLD_TAC_TRAIN_PICKLE = 'old_tac_train.pkl'
OLD_TAC_TEST_PICKLE = 'old_tac_test.pkl'


# Localizations for links

BIO_BERT_GIT_LINK = 'https://github.com/vthost/biobert-pretrained-pytorch/releases/download/v1.1-pubmed/biobert_v1.1_pubmed.zip'
TWIMED_GIT_LINK = 'https://github.com/nestoralvaro/TwiMed/archive/master.zip'


# Localizations for folders

ASSETS = 'assets'
CREDENTIALS = 'credentials'
TMP = 'tmp'
DATASETS = 'datasets'
MODELS = 'models'
SPLITS = 'splits'
RUNS = 'runs'

BIO_BERT_GIT = 'biobert_v1.1_pubmed'

TAC = 'tac'
SMM4H19 = 'smm4h19'
SMM4H20 = 'smm4h20'
TWIMED = 'twimed'
OLD_TAC = 'old_tac'
TAC_ARCHIVE = 'download_tweets'
CADEC_ARCHIVE = 'cadec'
SMM4H19_ARCHIVE = 'SMM4H_2019_task2_dataset'
CLEAN_SMM4H19_CSV = 'CleanFixedFullDataSet.csv'
CLEAN_SMM4H20_CSV = 'CleanFixedFullDataSet.csv'
TWIMED_ARCHIVE = 'TwiMed-master'
CADEC = 'cadec'
MEDDRA = 'meddra'
ORIGINAL = 'original'
SCT = 'sct'
TEXT = 'text'

OBJ_LEX = 'obj_lex'
DIRKSON = 'dirkson'

TWIMED_GOLD_CONFLATED = 'gold_conflated'
TWITTER = 'twitter'
PUBMED = 'pubmed'
TWIMED_TWITTER_TEXTS = 'twimed_twitter_texts'
TRAIN_DATA_1 = 'TrainData1.tsv'
TRAIN_DATA_2 = 'TrainData1.tsv'
TRAIN_DATA_3 = 'TrainData1.tsv'
TRAIN_DATA_4 = 'TrainData1.tsv'

# Localizations for paths

TAC_ZIP_PATH = abs_path([ASSETS, DATASETS, TAC, TAC_ZIP])
CADEC_ZIP_PATH = abs_path([ASSETS, DATASETS, CADEC, CADEC_ZIP])
SMM4H19_ZIP_PATH = abs_path([ASSETS, DATASETS, SMM4H19, SMM4H19_ZIP])
TWIMED_ZIP_PATH = abs_path([ASSETS, DATASETS, TWIMED, TWIMED_ZIP])

TAC_ARCHIVE_PATH = abs_path([TMP, TAC_ARCHIVE])
CADEC_ARCHIVE_PATH = abs_path([TMP, CADEC_ARCHIVE])
SMM4H19_ARCHIVE_PATH = abs_path([TMP, SMM4H19_ARCHIVE])
SMM4H_ORIGINAL_DATA_PATH = abs_path([ASSETS, DATASETS,"task1_2020_2019" ,"smm4h_original_data.csv"])
TWIMED_ARCHIVE_PATH = abs_path([TMP, TWIMED_ARCHIVE])
TMP_PATH = abs_path([TMP])

TAC_TEST_ANNOTATIONS_PATH = abs_path([TMP, TAC_ARCHIVE, TAC_TEST_ANNOTATIONS])
TAC_TRAIN_ANNOTATIONS_PATH = abs_path([TMP, TAC_ARCHIVE, TAC_TRAIN_ANNOTATIONS])
TAC_TEST_IDS_PATH = abs_path([TMP, TAC_ARCHIVE, TAC_TEST_IDS])
TAC_TRAIN_IDS_PATH = abs_path([TMP, TAC_ARCHIVE, TAC_TRAIN_IDS])

OLD_TAC_TRAIN_PICKLE_PATH = abs_path([ASSETS, DATASETS, OLD_TAC, OLD_TAC_TRAIN_PICKLE])
OLD_TAC_TEST_PICKLE_PATH = abs_path([ASSETS, DATASETS, OLD_TAC, OLD_TAC_TEST_PICKLE])

SMM4H19_TRAIN = 'TrainData{0}.tsv'
SMM4H19_TRAIN_PATH_1 = abs_path([TMP, SMM4H19_TRAIN.format(1)])
SMM4H19_TRAIN_PATH_2 = abs_path([TMP, SMM4H19_TRAIN.format(2)])
SMM4H19_TRAIN_PATH_3 = abs_path([TMP, SMM4H19_TRAIN.format(3)])
SMM4H19_TRAIN_PATH_4 = abs_path([TMP, SMM4H19_TRAIN.format(4)])

SMM4H19_CLEAN_TRAIN_PATH = abs_path([ASSETS, DATASETS, SMM4H19, CLEAN_SMM4H19_CSV])
SMM4H20_CLEAN_TRAIN_PATH = abs_path([ASSETS, DATASETS, SMM4H20, CLEAN_SMM4H20_CSV])
SMM4H19_BLIND_TEST_PATH = abs_path([ASSETS, DATASETS, SMM4H19, "testDataST23_participants.txt"])
SMM4H19_NEG_SPEC_ORIGINAL_PATH = abs_path([ASSETS, DATASETS, "smm4h19_negation_speculation/smm4h19__ORIGINAL__to_evaluate.pickle"])
SMM4H19_NEG_SPEC_NEGATION_PATH = abs_path([ASSETS, DATASETS, "smm4h19_negation_speculation/smm4h19__NEGATION__to_evaluate.pickle"])
SMM4H19_NEG_SPEC_SPECULATION_PATH = abs_path([ASSETS, DATASETS, "smm4h19_negation_speculation/smm4h19__SPECULATION__to_evaluate.pickle"])
SMM4H19_NEG_SPEC_FULL_PATH = abs_path([ASSETS, DATASETS, "smm4h19_negation_speculation/smm4h19__FULL__to_evaluate.pickle"])

CADEC_TEXTS_QUERY = abs_path([TMP, CADEC, TEXT, '*.txt'])
TWIMED_TWITTER_QUERY = abs_path([TMP, TWIMED_ARCHIVE, TWIMED_GOLD_CONFLATED, TWITTER, '*.ann'])


# Connection Strings

DB_CONNECTION_STRING = 'sqlite:///assets/db.sqlite'
TEST_DB_CONNECTION_STRING = 'sqlite:///tmp/test_db.sqlite'
INTEGRATION_TEST_DB_CONNECTION_STRING = 'sqlite:///tmp/integration_test_db.sqlite'
DB = 'db.sqlite'
DBSMM4H = 'db_smm4h.sqlite'
DBCADEC = 'db_cadec.sqlite'
TEST_DB = 'test_db.sqlite'
INTEGRATION_DB = 'integration_test_db.sqlite'
DB_PATH = abs_path([ASSETS, DB])
DB_CADEC_PATH = abs_path([ASSETS, DBCADEC])
DB_SMM4H_PATH = abs_path([ASSETS, DBSMM4H])
TEST_DB_PATH = abs_path([TMP, TEST_DB])
INTEGRATION_DB_PATH = abs_path([TMP, INTEGRATION_DB])


# Localizations for exceptions

FAIL_ENGINE_CREATION_EXCEPTION = 'Fail to create db engine' 


# Splits

TEST_ID = 'test.id'
TRAIN_ID = 'train.id'
VALIDATION_ID = 'validation.id'

CADEC_SPLIT = 'cadec_article'
SMM4H19_SPLIT = 'smm4h19_task2'
TAC_SPLIT = 'tac'
SMM4H19_SPLIT_PATH = abs_path([ASSETS, SPLITS, SMM4H19_SPLIT])
TAC_SPLIT_PATH = abs_path([ASSETS, SPLITS, TAC_SPLIT])