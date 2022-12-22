#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Simone Scaboro'

#https://discuss.huggingface.co/t/t5-finetuning-tips/684

import os 
import sys
import shutil
from os import path
import json
import numpy as np
import torch
import random
from utils.grid_search import GridSearch
from utils.full_test import FullTest
from utils.dataset import load_dataset
from utils.cli import Parser

from trainers.trainer_t5 import TrainerT5
from trainers.trainer_scifive import TrainerSciFive
from trainers.trainer_gpt2 import TrainerGPT2
from trainers.trainer_bart import TrainerBART
from trainers.trainer_pegasus import TrainerPegasus
from utils.run import Runs

import utils.results_manager as rs

def set_all_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    # if you are using GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(args):
    

    if args.smm4h19:
        CORPUS = "SMM4H19"
        SPLIT_FOLDER = "smm4h_2019_ner"
    elif args.smm4h20:
        CORPUS = "SMM4H20"
        SPLIT_FOLDER = "smm4h_2020_ner"
    elif args.cadec:
        CORPUS = "CADEC"
        SPLIT_FOLDER = "cadec_ner"
    else:
        raise Exception("You must specify the dataset")

    if args.m not in ["pegasus","bart", "gpt2", "t5", "scifive"]:
        raise Exception("Model not selected")
    else:
        SPLIT_FOLDER += f"_{args.m}"
        if args.run == None:
            args.run = f"{args.m.upper()}.json"


    if args.gs:
        exec_manager = GridSearch(SPLIT_FOLDER, CORPUS, args.m.upper(), args.m.upper())
    if args.ft:
        exec_manager = FullTest(SPLIT_FOLDER, CORPUS, args.express, args.m.upper(), args.m.upper())

    exec_manager.generate_run() 
    _path = exec_manager.get_run_path()
    runs = Runs("assets/runs/" + _path + args.run) 

    # start training and testing for each run
    for run in runs:
        print(run)

        # setting the seed for determinism            
        set_all_seed(run.random_seed)

        # choose trainer
        trainers = {
                'bart': TrainerBART,
                'pegasus': TrainerPegasus,
                't5': TrainerT5,
                'scifive': TrainerSciFive,
                'gpt2': TrainerGPT2
                }
        trainer = trainers[args.m](run, args.gpu)

        trainer.train_and_test()

        # get the results in a daraframe (epoch, text, gold, pred, loss_train, loss_test)
        test_results = trainer.get_test_results()

        metrics = rs.compute_metrics(test_results)
        best_metrics = rs.compute_best_metrics(metrics, run)

        exec_manager.add_row(run, best_metrics)
        
    
    if args.gs:
        exec_manager.get_best_run()
    if args.ft:
        exec_manager.get_mean(run.split_folder, run.model)


if __name__ == '__main__':
    sys.stdout.flush()
    args = Parser().parse()    
    main(args)
