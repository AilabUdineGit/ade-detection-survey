import os
import json
import pandas as pd
from ade_detection.utils.utils import *
import numpy as np
from ade_detection.utils.make_latex import make_latex

class FullTest:
    def __init__(self,
        corpus_split_folder,
        corpus,
        explainability_test=None,
        architecture_type=None
        ):
        """ Class to execute the full tests based on the best params found during the grid search
        """
        
        if architecture_type is None:
            raise Exception("You must specify the architecture type: wrapper, crf, lstm")

        self.explainability_test = explainability_test
        self.corpus = corpus
        self.full_test_df = pd.DataFrame({})   
        
        self.corpus_split_folder = corpus_split_folder.lower()
        self.grid_params_path = f"assets/grid_params/"

        with open(self.grid_params_path + f"{self.corpus_split_folder}.json") as fp:
            params = json.load(fp)
                
        self.architecture = architecture_type
        self.models = params['models']
        full_test_folder = f"full_test_express{self.explainability_test}" if self.explainability_test is not None else "full_test"
        self.best_run_path = f"assets/best_models/{self.architecture.lower()}/gs_{self.corpus_split_folder}/"
        self.run_path = f"assets/runs/{full_test_folder}/{self.architecture.lower()}/{self.corpus_split_folder}/"
        self.results_path = f"assets/results/{full_test_folder}/{self.architecture.lower()}/{self.corpus_split_folder}/"
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
        epochs = int(best_run['epochs'].replace(".0",""))
        architecture = self.architecture
        runs = []
        import random
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
            if self.explainability_test == 5:
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
            #seeds = ['42', '5', '28', '19', '33']
            #seeds = []
            #while len(seeds) < 30:
            #    l = random.randint(1,500)
            #    if str(l) not in seeds:
            #        seeds.append(str(l))
            #seeds = ['5', '4', '11', "34",'88', '290', '173', '1', '200','65', '223', '240', '144', '84']
        for seed in seeds:
            single_run = {}
            single_run['id'] = f"FT-{model}-{split_folder}-{architecture}[{seed}]"
            single_run['model'] = model
            single_run['architecture'] = architecture
            single_run['corpus'] = corpus
            single_run['split_folder'] = split_folder
            single_run['train_mode'] = train_modes
            single_run['notation'] = "IOB"
            single_run['train_config'] = {
                'batch_size': str(batch_size),
                'max_patience': "5",
                'learning_rate': lr,
                'dropout': str(dropout),
                'epochs': str(epochs).replace(".0",""),
                'epsilon': "1e-8",
                'random_seed': str(seed)
            }
            single_run['goal'] = goal
            single_run['tidy_modes'] = tidy_modes
            runs.append(single_run)
            with open(self.run_path + f"{model}.json", "w") as fp:
                json.dump(runs, fp, ensure_ascii=False, indent=4)

    def create_best_runs(self):
         
        for model in self.models:
            self._create_run(model)
        
        #create_best_params_latex(self.best_run_path, models)

        print("Full test runs completed")

#    with open(path + "latex_params_final.tex", "w") as fp:
#        fp.writelines(to_print)

    def add_row(self, task, best):

        model = task['model']
        
        r = best.partial
        s = best.strict
        idx = task['train_config']["random_seed"]
        self.full_test_df.loc[idx, 'model'] = task['model']
        self.full_test_df.loc[idx, 'f1(r)'] = str(r.best_f1)
        self.full_test_df.loc[idx, 'p(r)']  = str(r.precision)
        self.full_test_df.loc[idx, 'r(r)']  = str(r.recall)
        self.full_test_df.loc[idx, 'f1(s)'] = str(s.best_f1)
        self.full_test_df.loc[idx, 'p(s)']  = str(s.precision)
        self.full_test_df.loc[idx, 'r(s)']  = str(s.recall)
        
        self.full_test_df.to_csv(self.results_path + f"{model}.csv")
        self.full_test_df.to_pickle(self.results_path + f"{model}.pkl")

    def get_mean_sd(self, df, metric, str_rel):
        values = [float(s) for s in df[f"{metric}({str_rel})"].values]
        arr = np.array(values)
        return np.mean(arr, axis=0), np.std(arr, axis=0)

    def get_mean(self, corpus, model):
        
        df = pd.read_pickle(self.results_path + f"{model}.pkl")

        final_path = self.results_path + "final.pkl"
        complete_path = self.results_path + "complete.pkl"
        
        if not os.path.isfile(complete_path):
            final = pd.DataFrame({})
            final.to_pickle(complete_path)

        if not os.path.isfile(final_path):
            final = pd.DataFrame({})
            final.to_pickle(final_path)

        df_final = pd.read_pickle(final_path)
        df_complete = pd.read_pickle(complete_path)

        df_complete = pd.concat((df_complete,df))

        df_final = df_final.append(pd.Series({
            'model': model,
            'split_folder': corpus,
            'f1_avg(r)':    self.get_mean_sd(df,"f1","r")[0],
            'f1_std(r)':    self.get_mean_sd(df,"f1","r")[1],
            'p_avg(r)':     self.get_mean_sd(df,"p","r")[0],
            'p_std(r)':     self.get_mean_sd(df,"p","r")[1],
            'r_avg(r)':     self.get_mean_sd(df,"r","r")[0],
            'r_std(r)':     self.get_mean_sd(df,"r","r")[1],
            'f1_avg(s)':    self.get_mean_sd(df,"f1","s")[0],
            'f1_std(s)':    self.get_mean_sd(df,"f1","s")[1],
            'p_avg(s)':     self.get_mean_sd(df,"p","s")[0],
            'p_std(s)':     self.get_mean_sd(df,"p","s")[1],
            'r_avg(s)':     self.get_mean_sd(df,"r","s")[0],
            'r_std(s)':     self.get_mean_sd(df,"r","s")[1]
        }), ignore_index=True)

        df_final.to_csv(final_path.replace("pkl","csv"))
        df_final.to_pickle(final_path)
        df_complete.to_csv(complete_path.replace("pkl","csv"))
        df_complete.to_pickle(complete_path)
        make_latex(self.results_path, model, corpus)
