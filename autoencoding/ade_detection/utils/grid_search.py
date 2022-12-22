import pandas as pd
from itertools import product
import json
import os
from ade_detection.utils.utils import *

class GridSearch:
    def __init__(self,
        corpus_split_folder,
        corpus,
        architecture_type=None
        ):

        if architecture_type is None:
            raise Exception("You must specify the architecture type: wrapper, crf, lstm")

        self.corpus = corpus.upper()
        self.grid_search_df = pd.DataFrame({})
        self.gs_model = None
        self.grid_params_path = f"assets/grid_params/"
        recursive_check_path(self.grid_params_path)
        self.corpus_split_folder = corpus_split_folder.lower()
        
        with open(self.grid_params_path + f"{self.corpus_split_folder}.json") as fp:
            params = json.load(fp)

        self.architecture = architecture_type #params['architecture']

        self.run_path = f"assets/runs/grid_search/{self.architecture.lower()}/{self.corpus_split_folder}/"
        recursive_check_path(self.run_path)
        self.results_path = f"assets/runs/grid_search/{self.architecture.lower()}/{self.corpus_split_folder}/results/"
        recursive_check_path(self.results_path)
        self.best_run_path = f"assets/best_models/{self.architecture.lower()}/gs_{self.corpus_split_folder}/"
        recursive_check_path(self.best_run_path)

    def get_run_path(self):
        return self.run_path.replace("assets/runs/","")

    def set_model(self, model):
        self.gs_model = model

    def generate_run(self):
        
        with open(self.grid_params_path + f"{self.corpus_split_folder}.json") as fp:
            params = json.load(fp)

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
            'batch_size': str(best_run.batch),
            'max_patience': "5",
            'learning_rate': str(best_run.lr),
            'dropout': str(best_run.dropout),
            'epochs': str(best_run['epochs(r)']),
            'epsilon': str(best_run.epsilon),
            'random_seed': "42"
           }
                
        with open(self.best_run_path + f"{self.gs_model}.json", "w") as fp:
            json.dump(best_run, fp, ensure_ascii=False, indent=4)

        return True
        

    def add_row(self, task, best, keep_if_exists=True):
        
        model = task['model']

        if keep_if_exists and os.path.isfile(self.results_path + f"result_{model}.pkl"):
            self.grid_search_df = pd.read_pickle(self.results_path + f"result_{model}.pkl")

        r = best.partial
        s = best.strict

        self.grid_search_df = self.grid_search_df.append(pd.Series({
            'model'   : task['model'],
            'lr'      : task['train_config']['learning_rate'], 
            'dropout' : task['train_config']['dropout'],
            'epochs(s)'  : str(int(s.epochs)), 
            'epochs(r)'  : str(int(r.epochs)), 
            'batch'   : task['train_config']['batch_size'],
            'epsilon' : task['train_config']['epsilon'],
            'f1(r)'   : str(round(r.best_f1,3)),
            'p(r)'    : str(round(r.precision,3)),
            'r(r)'    : str(round(r.recall,3)),
            'f1(s)'   : str(round(s.best_f1,3)),
            'p(s)'    : str(round(s.precision,3)),
            'r(s)'    : str(round(s.recall,3))
            }),ignore_index=True)
        
        self.gs_model = model
        self.grid_search_df.to_csv(self.results_path + f"result_{model}.csv")
        self.grid_search_df.to_pickle(self.results_path + f"result_{model}.pkl")

