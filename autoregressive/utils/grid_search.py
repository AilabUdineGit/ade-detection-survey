import pandas as pd
from itertools import product
import json
import os
from utils.utils import *

class GridSearch:
    def __init__(self, split_folder, corpus, models=None, architecture=None):
        self.corpus = corpus.upper()
        self.corpus_split_folder = split_folder.lower()
        self.grid_search_df = pd.DataFrame({})
        self.gs_model = None
        self.grid_params_path = f"assets/grid_params/"
        recursive_check_path(self.grid_params_path)
       
        path = self.grid_params_path + f"{self.corpus_split_folder}.json"
        if not os.path.isfile(path):
            path = self.grid_params_path + f"general_params_{corpus.lower()}.json"

        with open(path) as fp:
            params = json.load(fp)
        
        if params['architecture'] == "GENERAL":
            params['architecture'] = architecture 
            params['models'] = models if type(models) == list else [models]

        self.architecture = params['architecture']
        self.task = params['task']

        self.run_path = f"assets/runs/grid_search/{self.task.lower()}/{self.architecture.lower()}/{self.corpus_split_folder}/"
        recursive_check_path(self.run_path)
        self.results_path = f"assets/runs/grid_search/{self.task.lower()}/{self.architecture.lower()}/{self.corpus_split_folder}/results/"
        recursive_check_path(self.results_path)
        self.best_run_path = f"assets/best_models/{self.task.lower()}/{self.architecture.lower()}/gs_{self.corpus_split_folder}/"
        recursive_check_path(self.best_run_path)
        
        self.params = params
        
    def get_run_path(self):
        return self.run_path.replace("assets/runs/","")

    def set_model(self, model):
        self.gs_model = model

    def generate_run(self):
        
        params = self.params

        ### PARAMETERS
        
        parameters = params['parameters']

        parameters = [v for v in parameters.values()]

        ### MODELS

        models = params['models']
        
        ### SEED
        seed = "42"
        goal = ["ADR"]
        corpus = self.corpus
        split_folder = self.corpus_split_folder.lower()
        train_modes =  "VALIDATION"
        tidy_modes = ['MERGE_OVERLAPS', "SOLVE_DISCONTINUOUS"]

        ### RUN CREATION
        for model in models:
            runs = []
            for lr,dropout,batch_size,epochs in product(*parameters):
                single_run = {}
                single_run['id'] = f"GS-{model}-{split_folder}-{self.architecture}[{str(lr).replace('.','_')}-{str(dropout).replace('.','_')}-{batch_size}-{epochs}]"
                single_run['model'] = model
                single_run['architecture'] = self.architecture
                single_run['corpus'] = corpus
                single_run['split_folder'] = split_folder
                single_run['train_mode'] = train_modes
                single_run['notation'] = "IOB"
                single_run['train_config'] = {
                    'batch_size': str(batch_size),
                    'max_patience': "5",
                    'learning_rate': lr,
                    'dropout': str(dropout),
                    'epochs': str(epochs),
                    'epsilon': "1e-8",
                    'source_len': params["source_len"],
                    'target_len': params["target_len"],
                    'random_seed': str(seed)
                    }
                single_run['goal'] = goal
                single_run['tidy_modes'] = tidy_modes
                runs.append(single_run)
            
            with open(self.run_path + f"/{model}.json", "w") as fp:
                json.dump(runs, fp, ensure_ascii=False, indent=4)



    def get_best_run(self):
        
        if not os.path.isfile(self.results_path + f"result_{self.gs_model}.pkl"):
            print(f"Grid Search results for model {self.gs_model} not found. Re-execute GS on it.")
            return False

        df = pd.read_pickle(self.results_path + f"result_{self.gs_model}.pkl")
        ids = [i for i,_ in df.iterrows()]
        values = [v['f1(r)'] for i,v in df.iterrows()]
        best_run = df.loc[ids[values.index(max(values))]]

        best_run = {
            'batch_size': best_run.batch,
            'max_patience': "5",
            'learning_rate': best_run.lr,
            'dropout': best_run.dropout,
            'epochs': best_run['epochs(r)'],
            'epsilon': best_run.epsilon,
            'random_seed': "42"
           }
                
        with open(self.best_run_path + f"{self.gs_model}.json", "w") as fp:
            json.dump(best_run, fp, ensure_ascii=False, indent=4)

        return True
        

    def add_row(self, run, best, keep_if_exists=False):
        
        model = run.model

        if keep_if_exists and os.path.isfile(self.results_path + f"result_{model}.pkl"):
            self.grid_search_df = pd.read_pickle(self.results_path + f"result_{model}.pkl")

        r = best['partial']
        s = best['strict']
        idx = len(self.grid_search_df)
        self.grid_search_df.loc[idx, 'model'   ] = model 
        self.grid_search_df.loc[idx, 'lr'      ] = run.learning_rate
        self.grid_search_df.loc[idx, 'dropout' ] = run.dropout
        self.grid_search_df.loc[idx, 'batch'   ] = run.batch_size
        self.grid_search_df.loc[idx, 'epsilon' ] = run.epsilon
        self.grid_search_df.loc[idx, 'epochs(s)'] = int(s['epochs'])
        self.grid_search_df.loc[idx, 'epochs(r)'] = int(r['epochs'])
        self.grid_search_df.loc[idx, 'f1(r)'   ] = round(r['f1'],3)
        self.grid_search_df.loc[idx, 'p(r)'    ] = round(r['precision'],3)
        self.grid_search_df.loc[idx, 'r(r)'    ] = round(r['recall'],3)
        self.grid_search_df.loc[idx, 'f1(s)'   ] = round(s['f1'],3)
        self.grid_search_df.loc[idx, 'p(s)'    ] = round(s['precision'],3)
        self.grid_search_df.loc[idx, 'r(s)'    ] = round(s['recall'],3)
        
        self.gs_model = model
        self.grid_search_df.to_csv(self.results_path + f"result_{model}.csv")
        self.grid_search_df.to_pickle(self.results_path + f"result_{model}.pkl")

