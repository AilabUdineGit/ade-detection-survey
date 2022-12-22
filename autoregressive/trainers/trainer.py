from typing import Dict
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from utils.interval_merger import IntervalMerger

class Trainer:
    def __init__(self):
        """
            Base initializer for the trainers
        """ 
        self.df_results = pd.DataFrame({})
        self.train_df = pd.DataFrame({})
        self.test_df = pd.DataFrame({})
        # self.device = None
        # self.tokenizer = None
        # self.model = None 
        # self.task = None
        # self.optimizer = None

    def train_and_test(self):
        """
            Perform training and testing and save the results in self.df_results
        """

        # build dataset and dataloader from dataframe
        train_dataloader = DataLoader(
            self.train_dataset,
            sampler=RandomSampler(self.train_dataset),
            batch_size=self.run.batch_size,
            num_workers=8
        )

        test_dataloader = DataLoader(
            self.test_dataset,
            sampler=SequentialSampler(self.test_dataset),
            batch_size=self.run.test_batch_size,
            num_workers=8
        )

#        self.run.epochs = 1
        for epoch in tqdm(range(self.run.epochs), desc="Training.."):

            train_loss = self._train(train_dataloader)
            with torch.no_grad():
                test_loss, test_results = self._test(test_dataloader)

            # test and train results are dataframes with index the sample index
            test_results['epoch'] = epoch+1
            test_results['train_loss'] = train_loss
            test_results['test_loss'] = test_loss

            self.df_results = pd.concat((self.df_results, test_results))

        self.df_results.to_csv(f"assets/results/RES-{self.run.id}.csv")


    def _build_target_text(self,df: pd.DataFrame):
        """
            Starting from the source text and the span, build the target label (word; word)
        """
        df['target'] = df.apply(lambda row : ";" if len(row.spans)==0 else "; ".join([row.text[s:e] for (s,e) in row['spans']]), axis=1)
        return df['target']


    
    def get_test_results(self) -> pd.DataFrame:
        """
            Returns the dataframe with the results from the training and testing phase

            Returns
            -------
            df_results : pd.Dataframe
                Dataframe with columns: (epoch, text, gold, pred, loss_train, loss_test) build during the training
        """
        return self.df_results
    

    def _load_splitted_dataset(self, corpus, train_mode):
        df_path = f"assets/datasets/{corpus}"
        train_df = pd.read_csv(f"{df_path}/train.csv")
        if train_mode == "VALIDATION":
            test_df = pd.read_csv(f"{df_path}/eval.csv")
        else:
            test_df = pd.read_csv(f"{df_path}/test.csv")
            val_df = pd.read_csv(f"{df_path}/eval.csv")
            train_df = pd.concat((train_df,val_df))
        return train_df, test_df

    def print_pred_gold(self, preds, gold):
        red = lambda t : f"\033[96m {t} \033[0m"
        green = lambda t : f"\033[92m {t} \033[0m"

        for p,g in zip(preds[:20], gold[:20]):
            print("-"*50)
            print(red(p), " ----> ", green(g))

    def _preprocess(self, train_df, test_df):
        raise Exception("You must definethis function for each model") 
        
    def _train(self, loader):
        raise Exception("You must definethis function for each model") 

    def _test(self, loader):
        raise Exception("You must define this function for each model")
