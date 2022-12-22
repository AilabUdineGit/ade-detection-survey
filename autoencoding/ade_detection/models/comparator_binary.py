#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Edoardo Lenzi, Beatrice Portelli'
__version__ = '1.0'
__license__ = '???'
__copyright__ = '???'


import os
import pandas as pd
from copy import deepcopy

from ade_detection.utils.logger import Logger
LOG = Logger.getLogger(__name__)
from ade_detection.services.model_service import ModelService
from ade_detection.models.evaluator import Evaluator
from ade_detection.utils.metrics import Metrics
import ade_detection.utils.localizations as loc 
import ade_detection.utils.file_manager as fm
from ade_detection.utils.graphics import *
from ade_detection.domain.enums import *
from ade_detection.domain.task import Task
from ade_detection.domain.best import Best

from sklearn.metrics import f1_score, precision_score, recall_score

POS_LABEL = 1
AVG_MODE = "binary"

class ComparatorBinary(object):

    def __init__(self, tasks:list, result_filename:str, model_filename = None, just_last=False):
        self.tasks = tasks        

        for task in tasks:
            model_service = ModelService(task)
            self.tokenizer = model_service.get_tokenizer()
            task = self.getAllMetrics(task)
            fm.to_pickle(task, loc.abs_path([loc.TMP, task.id+".pickle"]))
        
        if just_last:
            best = self.find_last_configuration(tasks)
        else:
            best = self.find_best_configuration(tasks)
        
        print(f"\n\nBEST:\n\n{self.best_to_str(best)}")
        best = Best(best, best)
        fm.to_pickle(best, loc.abs_path([loc.TMP, result_filename]))


    def getAllMetrics(self, task):
        #LOG.info('\n\n\n' + task.id + '\n\n\n')

        df = task.val_df
        loss_df = task.df

        tr_loss = loss_df['Training_Loss']
        vl_loss = loss_df['Valid_Loss']
        
        list_of_names = [f'preds_{e+1}' for e in range(task.train_config.epochs)]
        
        all_precision, all_recall, all_f1 = [], [], []
        
        for name in list_of_names:
            call_args = {
                "y_true": df.gold_labels.tolist(),
                "y_pred": df[name].tolist(),
                "average": AVG_MODE,
                "pos_label": POS_LABEL,
            }
            f1 = f1_score(**call_args)
            precision = precision_score(**call_args)
            recall = recall_score(**call_args)
                        
            all_f1.append(f1)
            all_precision.append(precision)
            all_recall.append(recall)

        metrics_df = pd.DataFrame(
            {'epoch': list_of_names,
            'precision': all_precision,
            'recall': all_recall,
            'f1score': all_f1,
            'training_loss': tr_loss,
            'validation_loss' : vl_loss
            })


        task.metrics_df = metrics_df
        
        return task
    
    
    def find_last_configuration(self, tasks:list):
        best = None

        if len(tasks) > 1:
            pass #LOG.info("This shouldn't happen in testing mode")
        
        task = tasks[0]
        
        
        df = task.metrics_df
        #curr_val_loss = 10000
        #curr_epoch = -1
        #curr_f1 = -1
        #curr_precision = -1
        #curr_recall = -1
        #curr_task = task
        
        line = df.iloc[-1]

        #for _, line in df.iterrows():

        tr_loss = line.training_loss
        vl_loss = line.validation_loss
        f1 = line.f1score
        precision = line.precision
        recall = line.recall 
        epoch = int(line.epoch.split("_")[-1])

        #if Metrics.overfit(tr_loss, vl_loss):
        #    break

        #if (f1 > curr_f1 and not Metrics.overfit(tr_loss, vl_loss)) or epoch == 1:
        curr_f1 = f1
        curr_precision = precision
        curr_recall = recall
        curr_epoch = epoch
        curr_val_loss = vl_loss
        curr_task = task
            
        #if best is None or best.best_f1 < curr_f1:
                
        best = deepcopy(curr_task)
        best.best_f1 = curr_f1
        best.precision = curr_precision
        best.recall = curr_recall
        best.best_val_loss = curr_val_loss 
        best.epochs = curr_epoch
                
        return best
        


    def find_best_configuration(self, tasks:list):
        best = None

        for task in tasks:
            df = task.metrics_df
            curr_val_loss = 10000
            curr_epoch = -1
            curr_f1 = -1
            curr_precision = -1
            curr_recall = -1
            curr_task = task
                
            for _, line in df.iterrows():
                    
                tr_loss = line.training_loss
                vl_loss = line.validation_loss
                f1 = line.f1score
                precision = line.precision
                recall = line.recall 
                epoch = int(line.epoch.split("_")[-1])
                    
                if Metrics.overfit(tr_loss, vl_loss):
                    break

                if (f1 > curr_f1 and not Metrics.overfit(tr_loss, vl_loss)) or epoch == 1:
                    curr_f1 = f1
                    curr_precision = precision
                    curr_recall = recall
                    curr_epoch = epoch
                    curr_val_loss = vl_loss
                    curr_task = task
            
            if best is None or best.best_f1 < curr_f1:
                
                best = deepcopy(curr_task)
                best.best_f1 = curr_f1
                best.precision = curr_precision
                best.recall = curr_recall
                best.best_val_loss = curr_val_loss 
                best.epochs = curr_epoch
                
        return best
                

    def best_to_str(self, best):
        
        values = [
            best.id,
            best.train_config.learning_rate,
            best.train_config.dropout,
            best.epochs,
            best.model.name + "+CRF" if "CRF" in best.architecture.name else best.model.name,
            best.best_f1,
            best.precision,
            best.recall,
        ]
        
        headers = [
            "config_id    |",
            "lr           |",
            "dropout      |",
            "epoch        |",
            "architecture |",
            "f1           |",
            "precision    |",
            "recall       |",
        ]
        
        best_str = "\n".join([h+" "+str(v) for h,v in zip(headers,values)])
        
        return best_str


    @staticmethod
    def compare_dirkson():
        df = pd.read_pickle(loc.abs_path([loc.ASSETS, loc.DIRKSON, loc.DIRKSON_TEST_RESULTS_PICKLE]))
        all_true_array = df.labels.values
        all_pred_array = df.pred_labels.values

        all_true = []
        for t in all_true_array:
            all_true.append(t)
        all_pred = []
        for t in all_pred_array:
            all_pred.append(t)

        evaluator = Evaluator(all_true, all_pred, [""])
        results, results_agg = evaluator.evaluate()

        print("\nresults - strict")
        print(results["strict"])
        print("\nresults - partial")
        print(results["partial"])

        print("strict")
        pre = results["strict"]["precision"]
        rec = results["strict"]["recall"]
        f1 = 0 if (pre==0 or rec==0) else (2*pre*rec)/(pre+rec)
        print("pre", round(pre, 2))
        print("rec", round(rec, 2))
        print("f1 ", round(f1, 2))

        print("partial")
        pre = results["partial"]["precision"]
        rec = results["partial"]["recall"]
        f1 = 0 if (pre==0 or rec==0) else (2*pre*rec)/(pre+rec)
        print("pre", round(pre, 2))
        print("rec", round(rec, 2))
        print("f1 ", round(f1, 2))