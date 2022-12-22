import os
import json
import pandas as pd
from utils.utils import *
import numpy as np
from utils.make_latex import make_latex
import random

class FullTest:
    def __init__(self, split_folder, corpus, explainability_test,  models=None, architecture=None):

        self.explainability_test = explainability_test
        self.corpus = corpus
        self.full_test_df = pd.DataFrame({})   
        
        self.corpus_split_folder = split_folder.lower()
        self.grid_params_path = f"assets/grid_params/"

        path = self.grid_params_path + f"{self.corpus_split_folder}.json"
        if not os.path.isfile(path):
            path = self.grid_params_path + f"general_params_{corpus.lower()}.json"

        with open(path) as fp:
            params = json.load(fp)
        
        if params['architecture'] == "GENERAL":
            params['architecture'] = architecture 
            params['models'] = models if type(models) == list else [models]

        full_test_folder = "full_test" if self.explainability_test is None else f"full_test_express{self.explainability_test}"
        self.architecture = params['architecture']
        self.task = params['task']
        self.models = params['models']
        self.source_len = params["source_len"] 
        self.target_len = params["target_len"] 
        self.best_run_path = f"assets/best_models/{self.task.lower()}/{self.architecture.lower()}/gs_{self.corpus_split_folder}/"
        self.run_path = f"assets/runs/{full_test_folder}/{self.task.lower()}/{self.architecture.lower()}/{self.corpus_split_folder}/"
        self.results_path = f"assets/runs/{full_test_folder}/{self.task.lower()}/{self.architecture.lower()}/{self.corpus_split_folder}/results/"
        recursive_check_path(self.run_path)
        recursive_check_path(self.results_path)


    def _check_or_create_dir(self, path):
            if not os.path.isdir(path):
                    os.mkdir(path)

    def _best_run_exists(self, path):
            return os.path.isfile(path)

    def get_run_path(self):
            return self.run_path.replace("assets/runs/","")


    def _create_run(self, model):
            
        path = self.best_run_path + f"{model}.json"
        if not self._best_run_exists(path):
            print(f"Best parameters for {model} (folder: {self.corpus_split_folder}) not found.")
            print(f"Skipping full test evaluation!")
            return

        with open(path,"r") as fp:
            best_run = json.load(fp)

        goal = ["ADR"]
        corpus = self.corpus
        split_folder = self.corpus_split_folder.lower()
        train_modes =  "TESTING"
        tidy_modes = ['MERGE_OVERLAPS', "SOLVE_DISCONTINUOUS"]
        lr = best_run['learning_rate']
        dropout = best_run['dropout']
        batch_size = best_run['batch_size']
        epochs = best_run['epochs']
        architecture = self.architecture
        runs = []
        if self.explainability_test is not None:
            if self.explainability_test == 0:
                seeds = ['65', '223', '240', '144', '84']
            if self.explainability_test == 1:
                seeds = ['88', '290', '173', '1', '200']
            if self.explainability_test == 2:
                seeds = ['99', '300', '155', '14', '11']
            if self.explainability_test == 4:
                seeds = []
                while len(seeds) < 30:
                    l = random.randint(1,500)
                    if str(l) not in seeds:
                        seeds.append(str(l))
            if self.explainability_test == 6:
                seeds = []
                while len(seeds) < 30:
                    l = random.randint(1,500)
                    if str(l) not in seeds:
                        seeds.append(str(l))
        else:
            seeds = ['42', '5', '12', '19', '33']
        for seed in seeds:
            single_run = {}
            single_run['id'] = f"FT-{corpus}-{model}[{seed}]"
            single_run['model'] = model
            single_run['architecture'] = architecture
            single_run['corpus'] = corpus
            single_run['split_folder'] = split_folder
            single_run['train_mode'] = train_modes
            single_run['notation'] = "IOB"
            single_run['train_config'] = {
                    'batch_size': batch_size,
                    'max_patience': "5",
                    'learning_rate': lr,
                    'dropout': dropout,
                    'epochs': epochs,
                    'epsilon': 1e-8,
                    'source_len': self.source_len,
                    'target_len': self.target_len,
                    'random_seed': seed
            }
            single_run['goal'] = goal
            single_run['tidy_modes'] = tidy_modes
            runs.append(single_run)
        with open(self.run_path + f"{model}.json", "w") as fp:
            json.dump(runs, fp, ensure_ascii=False, indent=4)

    def generate_run(self):
             
        for model in self.models:
            self._create_run(model)

        print("Full test runs completed")


    def add_row(self, run, best):

        r = best['partial']
        s = best['strict']
        
        idx = run.random_seed
        self.full_test_df.loc[idx, 'model'] = run.model
        self.full_test_df.loc[idx, 'f1(r)'] = r['f1']
        self.full_test_df.loc[idx, 'p(r)']  = r['precision']
        self.full_test_df.loc[idx, 'r(r)']  = r['recall']
        self.full_test_df.loc[idx, 'f1(s)'] = s['f1']
        self.full_test_df.loc[idx, 'p(s)']  = s['precision']
        self.full_test_df.loc[idx, 'r(s)']  = s['recall']
        
        self.full_test_df.to_csv(self.results_path + f"{run.model}.csv")
        self.full_test_df.to_pickle(self.results_path + f"{run.model}.pkl")


    def get_mean_sd(self, df, metric, str_rel):
        values = [float(s) for s in df[f"{metric}({str_rel})"].values]
        arr = np.array(values)
        return np.mean(arr, axis=0), np.std(arr, axis=0)


    def get_mean(self, corpus, model):
            
        df = pd.read_pickle(self.results_path + f"{model}.pkl")

        final_path = self.results_path + "final.pkl"
        if not os.path.isfile(final_path):
            final = pd.DataFrame({})
            final.to_pickle(final_path)

        df_final = pd.read_pickle(final_path)

        idx = len(df_final)
        df_final.loc[idx, 'model'] = model
        df_final.loc[idx, 'split_folder'] = corpus
        df_final.loc[idx, 'f1_avg(r)'] = round(self.get_mean_sd(df,"f1","r")[0],4)
        df_final.loc[idx, 'f1_std(r)'] = round(self.get_mean_sd(df,"f1","r")[1],4)
        df_final.loc[idx, 'p_avg(r)'] = round(self.get_mean_sd(df,"p","r")[0],4)
        df_final.loc[idx, 'p_std(r)'] = round(self.get_mean_sd(df,"p","r")[1],4)
        df_final.loc[idx, 'r_avg(r)'] = round(self.get_mean_sd(df,"r","r")[0],4)
        df_final.loc[idx, 'r_std(r)'] = round(self.get_mean_sd(df,"r","r")[1],4)
        df_final.loc[idx, 'f1_avg(s)'] = round(self.get_mean_sd(df,"f1","s")[0],4)
        df_final.loc[idx, 'f1_std(s)'] = round(self.get_mean_sd(df,"f1","s")[1],4)
        df_final.loc[idx, 'p_avg(s)'] = round(self.get_mean_sd(df,"p","s")[0],4)
        df_final.loc[idx, 'p_std(s)'] = round(self.get_mean_sd(df,"p","s")[1],4)
        df_final.loc[idx, 'r_avg(s)'] = round(self.get_mean_sd(df,"r","s")[0],4)
        df_final.loc[idx, 'r_std(s)'] = round(self.get_mean_sd(df,"r","s")[1],4)

        df_final.to_csv(final_path.replace("pkl","csv"))
        df_final.to_pickle(final_path)
        make_latex(self.results_path)
