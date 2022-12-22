#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Edoardo Lenzi, Beatrice Portelli, Simone Scaboro'
__version__ = '1.0'
__license__ = '???'
__copyright__ = '???'

# load .env configs
from ade_detection.utils.env import Env
Env.load()
from ade_detection.utils.logger import Logger
LOG = Logger.getLogger(__name__)

import os 
import sys
import shutil
from os import path
from datetime import datetime
from ade_detection.services.tokenization_service import TokenizationService
#from ade_detection.importer.task1_2019_2020_importer import Task120192020Importer
#from ade_detection.importer.smm4h2020_task2_importer import SMM4H2020T2Importer
#from ade_detection.importer.smm4h2020_task3_importer import SMM4H2020T3Importer
#from ade_detection.importer.smm4h19_task1_importer import SMM4H19Task1Importer
#from ade_detection.importer.bayer_importer import BayerImporter
from ade_detection.importer.smm4h20_importer import SMM4H20Importer
from ade_detection.importer.cadec_importer import CadecImporter
#from ade_detection.importer.smm4h19_neg_spec_importer import SMM4H19NegSpecImporter
#from ade_detection.importer.smm4h19_blind_importer import SMM4H19BlindImporter
#from ade_detection.importer.smm4h_original_data_importer import SMM4HOriginalDataImporter
#from ade_detection.importer.smm4h_negation_speculation_importer import SMM4H19NegSpecImporter
from ade_detection.services.database_service import DatabaseService
# from ade_detection.importer.smm4h22_task1a_importer import SMM4H22Task1aImporter
# from ade_detection.importer.smm4h22_task1b_importer import SMM4H22Task1bImporter
from ade_detection.services.model_service import ModelService
from ade_detection.domain.train_config import TrainConfig
from ade_detection.models.task_loader import TaskLoader
from ade_detection.models.trainer import Trainer
from ade_detection.models.trainer_binary import TrainerBinary
import ade_detection.utils.localizations as loc
import ade_detection.utils.file_manager as fm
from ade_detection.domain.task import Task
from ade_detection.domain.enums import *
from ade_detection.models.comparator import Comparator

import numpy as np
import torch
import random

from ade_detection.utils.grid_search import GridSearch
from ade_detection.utils.full_test import FullTest

class CliHandler(object):

    '''Cli business logic, given the arguments typed
    calls the right handlers/procedures of the pipeline'''


    def __init__(self, args):
        if not path.exists(loc.abs_path([loc.TMP, loc.BIO_BERT_GIT])):
            pass
            #ModelService.get_bio_git_model()
        if args.import_ds:
            self.import_handler()
        if args.run is not None:
            if args.cadec and args.gs:
                Env.DB = "DBCADEC"
            elif args.smm4h and args.gs:
                Env.DB = "DBSMM4H"
            elif args.cadec and args.ft:
                Env.DB = "DBCADECFT"
            elif args.smm4h and args.ft:
                Env.DB = "DBSMM4HFT"
            else:
                assert False, "Smm4h or cadec"
            self.run_handler(args)
        elif args.clean:
            self.clean_handler()
        else:
            self.default_handler()


    # Command Handlers 

    def default_handler(self):
        pass
        

    def clean_handler(self):
        #LOG.info('clean')
        fm.rmdir(loc.TMP_PATH)

    
    def import_handler(self):
        if os.path.exists(loc.DB_PATH):
            os.remove(loc.DB_PATH)
        DB = DatabaseService()
        DB.create_all()
        
        #CadecImporter()
        #SMM4H20Importer()
        #TokenizationService(CORPUS.CADEC)
        #TokenizationService(CORPUS.SMM4H20)
        # SMM4H19NegSpecImporter()
        # SMM4H22Task1aImporter()
        # SMM4H22Task1bImporter()
        # TokenizationService(CORPUS.SMM4H22_TASK1A)
        # TokenizationService(CORPUS.SMM4H2_TASK1B)
        # TokenizationService(CORPUS.SMM4H19_NEG_SPEC)

    def set_all_seed(self, seed):
        #LOG.info(f"random seed {seed}")
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        # if you are using GPU
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def get_architecture(self, args):
        """ get architecture name """
        if args.lstm:
            return "BERT_LSTM"
        if args.crf:
            return "BERT_CRF"
        if args.wrapper:
            return "BERT_WRAPPER"
        raise Exception("You have to specity the architecture type: crf, wrapper, lstm")

    def run_handler(self,args):
        _path = ""
        if args.cadec:
            _CORPUS = "CADEC"
            _SPLIT = "cadec"
        elif args.smm4h:
            _CORPUS = "SMM4H20"
            _SPLIT = "smm4h20"
        else:
            pass
            #raise Exception("You have to specify the dataset: -cadec or -smm4h")

        if args.gs:
            grid_search = GridSearch(_SPLIT, _CORPUS, self.get_architecture(args))
            grid_search.generate_run()
            _path = grid_search.get_run_path()

        elif args.ft:
            full_test = FullTest(_SPLIT, _CORPUS, args.express, self.get_architecture(args))
            full_test.create_best_runs()
            _path = full_test.get_run_path()

        else:
            _path = ""
            
        json = fm.from_json("assets/runs/" + _path + args.run[0])

        
        for task in json:
            #print(task['train_config']['random_seed'])
            random_seed = int(task['train_config']['random_seed'])
            self.set_all_seed(random_seed)

            train_config = TrainConfig(int(task['train_config']['max_patience']),
                                        float(task['train_config']['learning_rate']), 
                                        float(task['train_config']['dropout']),
                                        int(task['train_config']['epochs']),
                                        int(task['train_config']['batch_size']), 
                                        random_seed, 
                                        float(task['train_config']['epsilon']))
            
            loaded_task = TaskLoader(Task( task['id'], task['split_folder'], 
                                            enums_by_list(TIDY_MODE, task['tidy_modes']), 
                                            enum_by_name(CORPUS, task['corpus']), 
                                            enum_by_name(NOTATION, task['notation']), 
                                            enum_by_name(MODEL, task['model']), 
                                            enum_by_name(ARCHITECTURE, task['architecture']), 
                                            enums_by_list(ANNOTATION_TYPE, task['goal']), 
                                            enum_by_name(TRAIN_MODE, task['train_mode']), 
                                            train_config ))
            
            #if task["notation"] == "BINARY":
            #    TrainerBinary(loaded_task.task)
            #else:
            Trainer(loaded_task.task) 

            final_model = fm.from_pickle(loc.abs_path([loc.TMP, f"{task['id']}.pickle"]))
            comparator = Comparator([final_model], f"METRICS_{task['id']}.pickle", just_last=True)
            best = comparator.get_best()

#            if self.args.grid_search is not None:
            if args.gs:
                grid_search.add_row(task, best)
            if args.ft:
                full_test.add_row(task, best)

        if args.gs:
            grid_search.get_best_run()
        if args.ft:
            full_test.get_mean(task['split_folder'], task['model'])

        with open("tests_log.txt","a") as fp:
            fp.write(f"✓ {'full test' if args.ft else 'grid search'} completed for {task['model']} ({task['split_folder']} - {task['architecture'].lower()})[{datetime.today().strftime('%d-%m-%Y')}]\n")

if __name__ == '__main__':
    #LOG.info(f'Subprocess started {sys.argv}')
    sys.stdout.flush()
    from ade_detection.cli import Parser
    args = Parser().parse()    
    CliHandler(args)
