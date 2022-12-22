
from typing import Dict
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoModel,  BeamSearchScorer, Trainer, TrainingArguments
import transformers
from dataclasses import field, fields
from transformers.optimization import Adafactor, get_constant_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from data.dataset import GPT2Dataset
from trainers.trainer import Trainer

class TrainerGPT2(Trainer):
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
        
        model_name = "gpt2" 
        
        print("Training GPT2")
        self.prefix = "<|endoftext|>" 
        self.divider = "\n Label: "
        self.tokenizer = GPT2Tokenizer.from_pretrained(
                model_name,
                bos_token=self.prefix,
                eos_token=self.prefix,
                pad_token=self.prefix
                )
        self.tokenizer.pad_token = self.prefix
        self.tokenizer.padding_side = "left"
        #self.tokenizer.add_special_token("<new1>")
        self.model =  GPT2LMHeadModel.from_pretrained(model_name)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.pad_token_id = 50256
        self.model.config.max_length = run.source_len

        # build dataset
        self.train_df, self.test_df = self._load_splitted_dataset(run.corpus, run.train_mode)

        self.train_dataset = GPT2Dataset(self.train_df, run.source_len, run.target_len, self.tokenizer, self.divider)
        self.test_dataset = GPT2Dataset(self.test_df, run.source_len, run.target_len, self.tokenizer, self.divider)

        self.target_len = run.target_len
        # init model a tas taskknd tokenizer

        self.device = torch.device(f"cuda:{self._gpu}" if torch.cuda.is_available() else "cpu")

        self.model = self.model.to(self.device)

        self.run = run

        self.optimizer = AdamW(#Adafactor(
            self.model.parameters(),
            lr=self.run.learning_rate,
            # eps=(1e-30, 1e-3),
            # clip_threshold=1.0, 
            # decay_rate=-0.8,
            # beta1=None,
            # weight_decay=0.0,
            # scale_parameter=False,
            # relative_step=False,
            # warmup_init=False
        ) 

        # self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=10)

    # def train_and_test(self):
        # training_args = TrainingArguments(
                # fp16=True,
                # fp16_opt_level="03",

                # num_train_epochs=3,
                # per_device_train_batch_size=4,
                # gradient_accumulation_steps=4,

                # warmup_steps=100,
                # weight_decay=0.01,

                # output_dir="assets/",
                # overwrite_output_dir=True,
                # do_eval=False,
                # logging_strategy="no",
                # save_strategy="no",
                # save_total_limit=None
                # )

        # trainer = transformers.Trainer(
                # model=self.model,
                # args=training_args,
                # train_dataset=self.train_dataset,
                # data_collator= lambda data: {
                    # 'input_ids': torch.stack([f[0] for f in data]), 
                    # 'attention_mask': torch.stack([f[1] for f in data]), 
                    # 'labels': torch.stack([f[0] for f in data]),
                # }
                # )
        # trainer.train()
        # print("FINE")

    def _train(self, loader):
        total_loss = 0
        self.model.train() 
        for batch_idx, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            outputs = self.model(
                    input_ids = input_ids,
                    attention_mask = attention_mask,
                    labels = input_ids
                    )

            loss = outputs[0]
            
            total_loss += loss.item()

            loss.backward()
            self.optimizer.step()
            self.model.zero_grad()

        total_loss /= len(loader)
        # input_ids = input_ids.detach().cpu()
        # attention_mask = attention_mask.detach().cpu()
        return total_loss


    @torch.inference_mode()
    def _test(self, loader):
        df = {'text':[],'gold':[],'pred':[]}
        total_loss = 0
        self.model.eval()
        for batch in loader:
             
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = self.model(
                        input_ids = input_ids,
                        attention_mask = attention_mask,
                        labels = input_ids 
                        )
            
            gen_input_ids = batch["only_text"].to(self.device)
            gen_attention_mask = batch["only_text_mask"].to(self.device)
            
            generated_ids = self.model.generate(
                   input_ids=gen_input_ids,
                   attention_mask=gen_attention_mask,
                   bos_token_id=self.tokenizer.bos_token_id,
                   eos_token_id=self.tokenizer.eos_token_id,
                   pad_token_id=self.tokenizer.pad_token_id,
                   num_return_sequences=1,
                   no_repeat_ngram_size=2,
                   max_length=gen_input_ids.shape[1]+self.target_len,
                   repetition_penalty=1.5,
                   num_beams=2
                   # temperature=.85,
                   # do_sample=True,
                   #top_k=125,
                   #early_stopping=True
                   )
            #print(generated_ids[0])
            #assert False
            loss = outputs[0]
            df['text'].extend(input_ids.cpu().numpy())
            df['pred'].extend(generated_ids.cpu().numpy())
            df['gold'].extend(batch["only_labels"].numpy().tolist())
            total_loss += loss.item()

        decode = lambda encoded_text : self.tokenizer.decode(encoded_text, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        prefix_len = len("Input: ")
        df["pred"] = [decode(output) for output in df["pred"]]
        df["text"] = [decode(output) for output in df["text"]]
        #print(df["pred"][1])
        #print(df["text"][1])
        #print("-"*50)
        special_token_pos_pred = [t.find(self.divider) for t in df["pred"]]
        special_token_pos_text = [t.find(self.divider) for t in df["text"]]
        df["pred"] = [output[pos+len(self.divider):] for output,pos in zip(df["pred"], special_token_pos_pred)]
        df["gold"] = [decode(output)[prefix_len:] for output in df["gold"]]
        df["text"] = [output[prefix_len:pos] for output,pos in zip(df["text"], special_token_pos_text)]
        #print(df["pred"][1])
        #print(df["text"][1])
        #self.print_pred_gold(df['pred'], df['gold'])
        
        total_loss /= len(loader)
        return total_loss, pd.DataFrame(df)

