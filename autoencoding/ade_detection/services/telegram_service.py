#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Edoardo Lenzi, Beatrice Portelli, Simone Scaboro'
__version__ = '1.0'
__license__ = '???'
__copyright__ = '???'

from telegram.ext import MessageHandler
from telegram.ext import CommandHandler
from telegram.ext import Updater
from telegram.ext import Filters
from telegram import ParseMode
import psutil
import pickle
import signal
import time
import sys
import os 

import ade_detection.utils.localizations as loc
import ade_detection.utils.file_manager as fm


class TelegramService(object):


    def __init__(self, process):
        self.process = process
        self.bot = None
        with open(loc.abs_path([loc.CREDENTIALS, loc.BOT]), 'rb') as f:
            self.bot = pickle.load(f)
        self.admins = fm.from_json(loc.abs_path([loc.CREDENTIALS, loc.ADMINS])) 
        self.chats = fm.from_json(loc.abs_path([loc.CREDENTIALS, loc.CHATS]))
        self.updater = Updater(token=self.bot.token, use_context=True)
        self.dispatcher = self.updater.dispatcher

    
    def asyc_send(self, message):
            for chat in self.chats:
                self.bot.send_message(chat_id=chat, text=message, parse_mode=ParseMode.MARKDOWN)


    def start_polling(self):
        self.dispatcher.add_handler(CommandHandler('start', self.start))
        self.dispatcher.add_handler(CommandHandler('info', self.info))
        self.dispatcher.add_handler(CommandHandler('kill', self.kill))
        self.dispatcher.add_handler(MessageHandler(Filters.text & (~Filters.command), self.echo))
        self.dispatcher.add_handler(MessageHandler(Filters.command, self.unknown))
        self.updater.start_polling()


    def monitor_process(self):
        self.start_polling()
        self.asyc_send('👀👀👀👀👀👀👀\n\t\t\t*Start Monitoring*\n👀👀👀👀👀👀👀')
        while self.process.poll() is None:
            time.sleep(1)
            message = ''
            i = 0
            for line in iter(self.process.stdout.readline, b''):
                line = str(line, 'utf-8')
                if len(line) > 200:
                    line = line[-200:]
                if len(line.strip()) > 0:
                    message += '\n' + line.strip()
                    i += 1
                    sys.stdout.write(message)
                    sys.stdout.flush()
                if i > 10:
                    try:
                        self.asyc_send('```' + message + '```')                    
                        i = 0 
                        message = ''
                    except:
                        i = 0 
                        message = ''
            try:
                self.asyc_send('```' + message + '```')                    
            except:
                pass
            if psutil.virtual_memory().available < 1000000000:
                self.asyc_send('☠️☠️☠️☠️☠️☠️\n\t\t\t*Less than 1GB RAM*\n☠️☠️☠️☠️☠️☠️')
            #if psutil.virtual_memory().available < 500000000:
            #    self.asyc_send('☠️☠️☠️☠️☠️☠️\n\t\t\t*Less than 500MB RAM, process must be killed*\n☠️☠️☠️☠️☠️☠️')
            #    os.kill(self.process.pid, signal.SIGKILL)
            #    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            #    self.asyc_send('☠️ killed! ☠')
        self.asyc_send('👌👌👌👌👌👌👌\n\t\t\t*End Monitoring*\n👌👌👌👌👌👌👌')
        self.stop_polling()
        exit(0)


    def stop_polling(self):
        self.updater.stop()


    def start(self, update, context):
        chat_id = update.effective_chat.id
        if chat_id not in self.chats:
            self.chats.append(chat_id)
        fm.to_json(self.chats, loc.abs_path([loc.CREDENTIALS, loc.CHATS]))
        context.bot.send_message(chat_id=update.effective_chat.id, text='👍 Subscribed successfully! 👍')


    def echo(self, update, context):
        context.bot.send_message(chat_id=update.effective_chat.id, text=update.message.text)


    def unknown(self, update, context):
        context.bot.send_message(chat_id=update.effective_chat.id, text='🥴 Sorry, I didn\'t understand that command. 🥴')


    def kill(self, update, context): 
        if update.effective_user.id in self.admins:
            os.kill(self.process.pid, signal.SIGKILL)
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)  # Send the signal to all the process groups
            context.bot.send_message(chat_id=update.effective_chat.id, text='☠️ killed! ☠')
        else:
            context.bot.send_message(chat_id=update.effective_chat.id, text='🤡 unauthorized 🤡')


    def info(self, update, context):
        context.bot.send_message(chat_id=update.effective_chat.id, text=f'''Available memory: {psutil.virtual_memory().available}
Total memory: {psutil.virtual_memory().total}
CPUs load: {psutil.cpu_percent(interval=None, percpu=True)}''')