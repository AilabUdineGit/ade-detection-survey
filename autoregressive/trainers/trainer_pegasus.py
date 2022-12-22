from typing import Dict
from dataclasses import field, fields
from transformers.optimization import Adafactor, get_constant_schedule_with_warmup
from tqdm import tqdm
import pandas as pd
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from data.dataset import BartDataset
from trainers.trainer import Trainer

class TrainerPegasus(Trainer):
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
        
        model_name = "google/pegasus-xsum"


        print("Training Pegasus")
        # config = BartConfig.from_pretrained(model_name)
        self.model = PegasusForConditionalGeneration.from_pretrained(model_name)#, config=config)
        self.tokenizer =  PegasusTokenizer.from_pretrained(model_name)

        # build dataset
        self.train_df, self.test_df = self._load_splitted_dataset(run.corpus, run.train_mode)
        

        self.train_dataset = BartDataset(self.train_df, run.source_len, run.target_len, self.tokenizer)
        self.test_dataset = BartDataset(self.test_df, run.source_len, run.target_len, self.tokenizer)

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

        # self.scheduler = get_constant_schedule_with_warmup(self.optimizer, num_warmup_steps=10)

    # def train_and_test(self):
        # model_name = "facebook/bart-base"


        # # config = BartConfig.from_pretrained(model_name)
        # # self.model = BartForConditionalGeneration.from_pretrained(model_name)#, config=config)
        # # self.tokenizer =  BartTokenizer.from_pretrained(model_name)
        # # self.model = self.model.cuda() 
        # # self.optimizer = Adafactor(
            # # self.model.parameters(),
            # # lr=5e-5,#self.run.learning_rate,
            # # eps=(1e-30, 1e-3),
            # # clip_threshold=1.0, 
            # # decay_rate=-0.8,
            # # beta1=None,
            # # weight_decay=0.0,
            # # scale_parameter=False,
            # # relative_step=False,
            # # warmup_init=False
        # # ) 
        # df_train = pd.read_csv("assets/datasets/SMM4H20/train.csv")
        # df_test = pd.read_csv("assets/datasets/SMM4H20/eval.csv")
        # print("MY THING")
        # inputs_train = self.tokenizer(df_train.input_text.values.tolist(), max_length=64, padding=True, truncation=True, return_tensors="pt")
        # inputs_test = self.tokenizer(df_test.input_text.values.tolist(), max_length=64, padding=True, truncation=True, return_tensors="pt")

        # labels_train = self.tokenizer(df_train.target_text.values.tolist(), max_length=20, padding=True, truncation=True, return_tensors="pt")
        # labels_test = self.tokenizer(df_test.target_text.values.tolist(), max_length=20, padding=True, truncation=True, return_tensors="pt")

        # class MyDataset(Dataset):
            # def __init__(self, data, label):
                # self.data = data
                # self.label = label
            
            # def __len__(self):
                # return len(self.data.input_ids)

            # def __getitem__(self,idx):
                # return {
                    # "input_ids": self.data.input_ids[idx],
                    # "attention_mask": self.data.attention_mask[idx],
                    # "label": self.label.input_ids[idx],
                # }
        
        # train_dataloader = DataLoader(MyDataset(inputs_train, labels_train), batch_size=32)
        # test_dataloader = DataLoader(MyDataset(inputs_test, labels_test), batch_size=32)

        # for epoch in tqdm(range(10)):
            # self.model.train()
            # for batch in train_dataloader:
                # input_ids = batch["input_ids"].cuda()
                # attention_mask = batch["attention_mask"].cuda()
                # label = batch["label"].cuda()
                # label[label == self.tokenizer.pad_token_id] = -100
                
                # outputs = self.model(
                            # input_ids = input_ids,
                            # attention_mask = attention_mask,
                            # labels = label
                            # )

                # loss = outputs[0]
                
                # loss.backward()
                # self.optimizer.step()
                # self.model.zero_grad()

            # with torch.no_grad():
                # df = {'text':[],'gold':[],'pred':[]}
                
                # self.model.eval()
                # for batch in tqdm(test_dataloader, desc="test"):
                    # input_ids = batch["input_ids"].cuda()
                    # attention_mask = batch["attention_mask"].cuda()
                    # label = batch["label"].cuda()
                    # label[label == self.tokenizer.pad_token_id] = -100

                    # with torch.no_grad():
                        # outputs = self.model(
                            # input_ids = input_ids,
                            # attention_mask = attention_mask,
                            # labels = label
                            # )
                            
                    # generated_ids = self.model.generate(
                        # input_ids=input_ids,
                        # attention_mask=attention_mask,
                        # max_length=20,
                        # num_beams=1, length_penalty=2.0, repetition_penalty=1.0, do_sample=False,
                        # early_stopping=True, top_k=None, top_p=None, num_return_sequences=1
                        # )
                    
                    # df['text'].extend(input_ids.cpu().numpy())
                    # df['pred'].extend(generated_ids.cpu().numpy())
                    # df['gold'].extend(batch["label"])

                # decode = lambda encoded_text : self.tokenizer.decode(encoded_text, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                # df["pred"] = [decode(output) for output in df["pred"]]
                # df["gold"] = [decode(output) for output in df["gold"]]
                # df["text"] = [decode(output) for output in df["text"]]

                # self.print_pred_gold(df["pred"], df["gold"])



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
            # self.scheduler.step()
            self.model.zero_grad()

        total_loss /= len(loader)
        return total_loss


    def _test(self, loader):
        with torch.no_grad():
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
            df["text"] = [decode(output) for output in df["text"]]
            
            # self.print_pred_gold(df['pred'], df['gold'])
            
            total_loss /= len(loader)
        return total_loss, pd.DataFrame(df)

