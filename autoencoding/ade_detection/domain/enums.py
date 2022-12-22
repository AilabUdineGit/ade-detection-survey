#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Edoardo Lenzi, Beatrice Portelli, Simone Scaboro'
__version__ = '1.0'
__license__ = '???'
__copyright__ = '???'


import enum

from ade_detection.utils.tagging_rules import TaggingRules
import ade_detection.utils.localizations as loc


def enum_by_name(enum, name: str):
        for i in enum:
                if i.name == name:
                        return i 
        return None


def enums_by_list(enum, array: list):
        return [enum_by_name(enum,i) for i in array]


class CORPUS(enum.Enum):
        CADEC = 1
        TAC = 2
        TWIMED_TWITTER = 3
        TWIMED_PUBMED = 4
        SMM4H19_TASK1 = 5
        SMM4H19_TASK2 = 6
        PSY_TAR = 7
        BIO_SCOPE = 8
        SMM4H19_NEGSPEC = 9
        SMM4H20_TASK2 = 10
        SMM4H20_TASK3 = 11
        SMM4H_ORIGINAL_DATA = 12
        BAYER = 13
        SMM4H19_NEG_SPEC = 14
        SMM4H22_TASK1A = 15
        SMM4H22_TASK1B = 16
        SMM4H20 = 17


class PARTITION_TYPE(enum.Enum):
        TRAIN = 1
        VALIDATION = 2
        TEST = 3


class NOTATION(enum.Enum):
        IO = 1
        IOB = 2
        BILUO = 3
        BINARY = 4


class MODEL(str, enum.Enum):
        SPANBERT = 'SpanBERT/spanbert-base-cased'
        BIOBERT = "dmis-lab/biobert-v1.1"#'monologg/biobert_v1.1_pubmed'
        #BIO_BERT_GIT = loc.abs_path([loc.TMP, loc.BIO_BERT_GIT])
        SCIBERT = 'allenai/scibert_scivocab_cased'
        BIOCLINICALBERT = 'emilyalsentzer/Bio_ClinicalBERT'
        BERTWEET = 'vinai/bertweet-large'
        PUBMEDBERT = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
        BERT = 'bert-base-uncased'

# new models
        ROBERTA                         = "roberta-base"
        XLNET                           = "xlnet-base-cased"
        ELECTRA                         = "google/electra-base-discriminator"
        BIOROBERTA                      = "allenai/biomed_roberta_base"
        BIOCLINICALROBERTA      = "simonlevine/bioclinical-roberta-long"
        BIOELECTRA                      = "kamalkraj/bioelectra-base-discriminator-pubmed"

        ALBERT                          = "albert-large-v2" #albert-base-v2
        BIOALBERT                       = "sultan/BioM-ALBERT-xxlarge"
        DISTILBERT                      = "distilbert-base-uncased"
        ENDRBERT                        = "cimm-kzn/endr-bert"
        BLUEBERT                        = "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12"
        COVIDBERT                       = "digitalepidemiologylab/covid-twitter-bert-v2"
        # autoregressive and t2t
        T5 = "t5-base"
        GPT2 = "gpt2"
        GPTNERO = "EleutherAI/gpt-neo-125M"
        BART = "facebook/bart-base"

        FINETUNED_BERT            = "tmp/models/__BERT_seed20"
        FINETUNED_BERTCRF         = "tmp/models/__BERTCRF_seed42"
        FINETUNED_SPANBERT        = "tmp/models/__SpanBERT_seed10"
        FINETUNED_SPANBERTCRF = "tmp/models/__SpanBERTCRF_seed10"
        FINETUNED_PUBMEDBERT  = "tmp/models/__PUBMEDBERT_seed_30"
        # bert
        BERT_SPLIT_0 = "tmp/models/bert_split_0"
        BERT_SPLIT_1 = "tmp/models/bert_split_1"
        BERT_SPLIT_2 = "tmp/models/bert_split_2"
        BERT_SPLIT_3 = "tmp/models/bert_split_3"
        BERT_SPLIT_4 = "tmp/models/bert_split_4"
        BERT_SPLIT_5 = "tmp/models/__BERT_seed20"
        BERT_SPLIT_6 = "tmp/models/bert_split_6"

        # spanbert
        SPANBERT_SPLIT_0 = "tmp/models/spanbert_split_0"
        SPANBERT_SPLIT_1 = "tmp/models/spanbert_split_1"
        SPANBERT_SPLIT_2 = "tmp/models/spanbert_split_2"
        SPANBERT_SPLIT_3 = "tmp/models/spanbert_split_3"
        SPANBERT_SPLIT_4 = "tmp/models/spanbert_split_4"
        SPANBERT_SPLIT_5 = "tmp/models/__SpanBERT_seed10"
        SPANBERT_SPLIT_6 = "tmp/models/spanbert_split_6"

        # pubmedbert
        PUBMEDBERT_SPLIT_0 = "tmp/models/pubmedbert_split_0"
        PUBMEDBERT_SPLIT_1 = "tmp/models/pubmedbert_split_1"
        PUBMEDBERT_SPLIT_2 = "tmp/models/pubmedbert_split_2"
        PUBMEDBERT_SPLIT_3 = "tmp/models/pubmedbert_split_3"    
        PUBMEDBERT_SPLIT_4 = "tmp/models/pubmedbert_split_4"
        PUBMEDBERT_SPLIT_5 = "tmp/models/__PUBMEDBERT_seed_30"
        PUBMEDBERT_SPLIT_6 = "tmp/models/pubmedbert_split_6"

        #OTHER = "/mnt/HDD/bportelli/GitHub/ADE_twimed_backup/tmp/models/__BERTCRF_seed42"


class ARCHITECTURE(enum.Enum):
        BERT_WRAPPER = 1
        BERT_CRF = 2
        BERT_LSTM = 3
        DUAL_BERT = 4
        T5 = 5
        AUTOREGRESSIVE = 6
        T2T = 7


class TIDY_MODE(enum.Enum):
        MERGE_OVERLAPS = 1
        MERGE_ADJACENT = 2
        SOLVE_DISCONTINUOUS = 3


class TRAIN_MODE(enum.Enum):
        VALIDATION = 1
        TESTING = 2
        JUST_TESTING = 3


class ANNOTATION_TYPE(enum.Enum):
        ADR = 1
        Drug = 2
        Disease = 3
        Indication = 4
        Symptom = 5
        Finding = 6
        related_drug = 7
        target_drug = 8
        meddra_code = 9
        meddra_term = 10


def annotation_by_name(name:str):
        for annotation in ANNOTATION_TYPE:
                if annotation.name == name:
                        return annotation 
        raise ValueError("Unknown annotation_type: " + name)


#BATCH_SIZE = { CORPUS.CADEC : 8, 
#                           CORPUS.TAC : 32, 
#                           CORPUS.TWIMED_TWITTER : 32, 
#                           CORPUS.TWIMED_PUBMED : 8, 
#                           CORPUS.SMM4H19_TASK2 : 32,
#                           CORPUS.SMM4H19_TASK1 : 32,
#                           CORPUS.SMM4H19_NEGSPEC : 32,
#                           CORPUS.SMM4H20_TASK2: 32,
#                           CORPUS.SMM4H20_TASK3: 32,
#                           CORPUS.SMM4H_ORIGINAL_DATA: 32,
#                           CORPUS.BAYER: 32,
#                           CORPUS.SMM4H19_NEG_SPEC:32,
#                           CORPUS.SMM4H22_TASK1A:32,
#                           CORPUS.SMM4H22_TASK1B:32,
#                           CORPUS.SMM4H20:32
#                         }


MAX_SEQ_LEN = { CORPUS.CADEC : 512, 
                                CORPUS.TAC : 64, 
                                CORPUS.TWIMED_TWITTER : 64, 
                                CORPUS.TWIMED_PUBMED : 512, 
                                CORPUS.SMM4H19_TASK2 : 64,
                                CORPUS.SMM4H19_TASK1 : 128,
                                CORPUS.SMM4H19_NEGSPEC: 64,
                                CORPUS.SMM4H20_TASK2: 64,
                                CORPUS.SMM4H20_TASK3: 64,
                                CORPUS.SMM4H_ORIGINAL_DATA: 64,
                                CORPUS.BAYER: 512,
                                CORPUS.SMM4H19_NEG_SPEC: 64,
                                CORPUS.SMM4H22_TASK1A: 64,
                                CORPUS.SMM4H22_TASK1B: 64,
                                CORPUS.SMM4H20: 64
                          }


class COMPARATOR_MODE(enum.Enum):
        STRICT = 1
        PARTIAL = 2


'''
TO_INDEX_RULE = { GOAL.IOB_ADR_NONADR : TaggingRules.iob_adr_nonadr_to_index,
                                  GOAL.IO_ADR_NONADR : TaggingRules.iob_adr_nonadr_to_index,
                                  GOAL.IOB_DRUG_NONDRUG : TaggingRules.iob_drug_nondrug_to_index }


TO_TAG_RULE = { GOAL.IOB_ADR_NONADR : TaggingRules.iob_adr_nonadr_to_tag,
                                GOAL.IO_ADR_NONADR : TaggingRules.iob_adr_nonadr_to_tag,
                                GOAL.IOB_DRUG_NONDRUG : TaggingRules.iob_drug_nonadr_to_tag }


DETOK_PRIORITY = { GOAL.IOB_ADR_NONADR : { 0: 0, 1: 1, 2: 2 }, # {0: O, 1: I-ADR, 2: B-ADR} 
                                   GOAL.IOB_DRUG_NONDRUG : { 0: 0, 1: 1, 2: 2 } # {0: O, 1: I-Drug, 2: B-Drug} 
                                 }
'''
