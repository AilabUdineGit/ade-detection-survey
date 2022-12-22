#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Edoardo Lenzi, Beatrice Portelli, Simone Scaboro'
__version__ = '1.0'
__license__ = '???'
__copyright__ = '???'


# load .env configs
from ade_detection.utils.env import Env
Env.load()

# ade_detection/cli.py wrapper
import subprocess 
import os 
import sys 

from ade_detection.cli import Parser
from ade_detection.cli_handler import CliHandler

args = Parser().parse()    
#if args.telegram:
#    if len(sys.argv) == 2:
#        TelegramService(None).start_polling()
#    else:
#        command = [f'{os.getcwd()}/ade_detection/cli_handler.py']
#        command.extend(sys.argv[1:])
#        process = subprocess.Popen(' '.join(command), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
#        TelegramService(process).monitor_process()
#else:
CliHandler(args)
