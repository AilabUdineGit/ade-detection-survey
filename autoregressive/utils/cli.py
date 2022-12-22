#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

class Parser(object):
    '''Cli entry point of the script, based on the library argparse
    see also: https://docs.python.org/3.9/library/argparse.html'''

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter, 
            description = "Autoregressive ADE detection pipeline",
            epilog = '' )

        self.parser.add_argument(
                '--run', dest='run',
                type=str, default=None,
                help='Run an array of tasks (remember to specify the name of your run .json)')
        
        self.parser.add_argument(
                '-express', dest='express',
                type=int, default=None,
                help="Perform full test for expressability")

        self.parser.add_argument(
                '-p', dest='p',
                action='store_const', const=True, default=False,
                help="Print predictions")

        self.parser.add_argument(
                '-m', dest='m',
                type=str, default=None,
                help="Select model")

        self.parser.add_argument(
                '-gpu', 
                type=int, default=1,
                help="Select gpu")

        # MODEL
        self.parser.add_argument(
                '-gpt2', dest='gpt2',
                action='store_const', const=True, default=False,
                help="Use a GPT2 model")
        
        self.parser.add_argument(
                '-t5', dest='t5',
                action='store_const', const=True, default=False,
                help="Use T5")

        self.parser.add_argument(
                '-bart', dest='bart',
                action='store_const', const=True, default=False,
                help="Use BART")

        # TRAINING TYPE
        self.parser.add_argument(
                '-gs', dest='gs',
                action='store_const', const=True, default=False,
                help="Perform grid search (save validation metrics)")
        
        self.parser.add_argument(
                '-ft', dest='ft',
                action='store_const', const=True, default=False,
                help="Perform full test (save test metrics)")

        # DATASET
        self.parser.add_argument(
                '-smm4h19', dest='smm4h19',
                action='store_const', const=True, default=False,
                help="Use SMM4H 2019")

        self.parser.add_argument(
                '-smm4h20', dest='smm4h20',
                action='store_const', const=True, default=False,
                help="Use SMM4H 2020")

        self.parser.add_argument(
                '-cadec', dest='cadec',
                action='store_const', const=True, default=False,
                help="Use CADEC")

    def parse(self):    
        return self.parser.parse_args()


    def parse_args(self, command):    
        return self.parser.parse_args(command)
