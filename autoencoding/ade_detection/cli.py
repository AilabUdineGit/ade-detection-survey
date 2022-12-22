#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Edoardo Lenzi, Beatrice Portelli, Simone Scaboro'
__version__ = '1.0'
__license__ = '???'
__copyright__ = '???'


import subprocess
import argparse
import sys
import os 

from ade_detection.cli_handler import CliHandler


class Parser(object):
    '''Cli entry point of the script, based on the library argparse
    see also: https://docs.python.org/3.9/library/argparse.html'''

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter, 
            description = '''                              Welcome to ADE Detection Script :)  

+---------------------------------------------------------------------------------------------+
|   /  _  \ \______ \ \_   _____/ \______ \   _____/  |_  ____   _____/  |_|__| ____   ____   |
|  /  /_\  \ |    |  \ |    __)_   |    |  \_/ __ \   __\/ __ \_/ ___\   __\  |/  _ \ /    \  |
| /    |    \|    `   \|        \  |    `   \  ___/|  | \  ___/\  \___|  | |  (  <_> )   |  \ | 
| \____|__  /_______  /_______  / /_______  /\___  >__|  \___  >\___  >__| |__|\____/|___|  / |
|         \/        \/        \/          \/     \/          \/     \/                    \/  |
+---------------------------------------------------------------------------------------------+''',
            epilog = 'Source: https://github.com/beatrice-portelli/ADE_Detection' )


        self.parser.add_argument('-i', '--import', dest='import_ds', action='store_const',
                                const=True, default=False,
                                help='Drop database and import all datasets')


        self.parser.add_argument('-c', '--clean', dest='clean', action='store_const',
                                const=True, default=False,
                                help='Clean temporary/useless files')


        self.parser.add_argument('--run', dest='run', metavar='N', type=str, nargs=1,
                                help='Run an array of tasks (remember to specify the name of your run .json)')


        self.parser.add_argument('--telegram', dest='telegram', action='store_const',
                                const=True, default=False,
                                help='Send notifications in broadcast on the Telegram Bot @ADE_Detection_Bot')

        # --------------
        # GRID AND FULL TEST
        self.parser.add_argument('-gs', dest='gs', action='store_const',
                                const=True, default=False,
                                help="Perform grid search (save validation metrics)")

        self.parser.add_argument('-ft', dest='ft', action='store_const',
                        const=True, default=False,
                        help="Perform grid search (save validation metrics)")
        
        self.parser.add_argument('-express', dest='express', type=int, default=None,
                        help="Perform full test for expressability")

        self.parser.add_argument('-autoreg', dest='autoreg', action='store_const',
                                const=True, default=False,
                                help="Perform grid search (save validation metrics)")

        # --------------
        # DATASET
        self.parser.add_argument(
            '-cadec', dest='cadec',
            action='store_const', const=True, default=False,
            help="Use CADEC")      

        self.parser.add_argument(
            '-smm4h', dest='smm4h',
            action='store_const', const=True, default=False,
            help="Use SMM4H")     

        # --------------
        # ARCHITECTURE
        self.parser.add_argument(
            '-lstm', dest='lstm',
            action='store_const', const=True, default=False,
            help="Use LSTM architecture")      

        self.parser.add_argument(
            '-crf', dest='crf',
            action='store_const', const=True, default=False,
            help="Use CRF architecture")      

        self.parser.add_argument(
            '-wrapper', dest='wrapper',
            action='store_const', const=True, default=False,
            help="Use wrapper architecture")      

    def parse(self):    
        return self.parser.parse_args()


    def parse_args(self, command):    
        return self.parser.parse_args(command)
