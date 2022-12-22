from typing import Dict
from dataclasses import field, fields
from transformers.optimization import Adafactor, get_constant_schedule_with_warmup
from tqdm import tqdm
import pandas as pd
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from data.dataset import T5Dataset
from trainers.trainer import Trainer

class TrainerSciFive(Trainer):
    def __init__(self, run, gpu):
        """
            Initialize trainer defining model, tokenizer and data

            Parameters
            ----------
            train_ds : pd.DataFrame
                training dataframe
            test_ds : pd.DataFrame
                testing dataframe
            task : dict
                dictionary with the run informations     
        """ 
        self._gpu = gpu
        super().__init__()
        
        model_name = "razent/SciFive-base-Pubmed"

        print("Training SciFive")
        config = T5Config.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name, config=config)
        self.tokenizer =  T5Tokenizer.from_pretrained(model_name, truncate=True)

        # build dataset
        self.train_df, self.test_df = self._load_splitted_dataset(run.corpus, run.train_mode)

        self.train_dataset = T5Dataset(self.train_df, run.source_len, run.target_len, self.tokenizer)
        self.test_dataset = T5Dataset(self.test_df, run.source_len, run.target_len, self.tokenizer)

        # init model a tas taskknd tokenizer

        self.device = torch.device(f"cuda:{self._gpu}" if torch.cuda.is_available() else "cpu")

        self.model = self.model.to(self.device)

        self.run = run

        self.optimizer = Adafactor(
            self.model.parameters(),
            lr=self.run.learning_rate,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0, 
            decay_rate=-0.8,
            beta1=None,
            weight_decay=0.0,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False
        ) 

        self.scheduler = get_constant_schedule_with_warmup(self.optimizer, num_warmup_steps=10)

    def _train(self, loader):
        total_loss = 0
        self.model.train() 
        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            labels[labels == self.tokenizer.pad_token_id] = -100

            outputs = self.model(
                    input_ids = input_ids,
                    attention_mask = attention_mask,
                    labels = labels
                    )

            loss = outputs[0]
            
            total_loss += loss.item()

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.model.zero_grad()

        total_loss /= len(loader)
        return total_loss


    @torch.inference_mode()
    def _test(self, loader):
        df = {'text':[],'gold':[],'pred':[]}
        total_loss = 0
        self.model.eval()
        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            labels[labels == self.tokenizer.pad_token_id] = -100
            
            with torch.no_grad():
                outputs = self.model(
                        input_ids = input_ids,
                        attention_mask = attention_mask,
                        labels = labels
                        )

            generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=20,
                    num_beams=1, length_penalty=2.0, repetition_penalty=1.0, do_sample=False,
                    early_stopping=True, top_k=None, top_p=None, num_return_sequences=1
                    )

            loss = outputs[0]
            df['text'].extend(input_ids.cpu().numpy())
            df['pred'].extend(generated_ids.cpu().numpy())
            df['gold'].extend(batch["labels"])
            
            total_loss += loss.item()

        decode = lambda encoded_text : self.tokenizer.decode(encoded_text, skip_special_tokens=True, clean_up_tokenization_spaces=True)
 
        df["pred"] = [decode(output) for output in df["pred"]]
        df["gold"] = [decode(output) for output in df["gold"]]
        df["text"] = [decode(output)[len("ner ade: "):] for output in df["text"]]

        #self.print_pred_gold(df['pred'], df['gold'])
        
        total_loss /= len(loader)
        return total_loss, pd.DataFrame(df)

