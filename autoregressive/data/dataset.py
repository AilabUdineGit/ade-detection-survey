from torch.utils.data import Dataset
import pandas as pd
import torch

## T5DatasetClass
class T5Dataset(torch.utils.data.Dataset):
    def __init__(self, df, source_len, target_len, tokenizer):
        df = df.fillna("")
        df['target_text'] = df['target_text'].apply(lambda r: ";" if r == "none" else r)
        df['input_text'] = "ner ade: " + df['input_text']
        self.texts = tokenizer(df['input_text'].values.tolist(), max_length=source_len, padding=True, truncation=True, return_tensors="pt")
        self.labels = tokenizer(df['target_text'].values.tolist(), max_length=target_len, padding=True, truncation=True, return_tensors="pt")

    def __len__(self):
        return len(self.texts["input_ids"])

    def __getitem__(self,idx):
        return {
                'input_ids': self.texts['input_ids'][idx],
                'attention_mask': self.texts['attention_mask'][idx],
                'labels': self.labels['input_ids'][idx]
                }


## GPT2DatasetClass
class GPT2Dataset(torch.utils.data.Dataset):
    def __init__(self, df, source_len, target_len, tokenizer, divider):
        df = df.fillna("")
        before_text = "Input: "
        df['target_text'] = df['target_text'].apply(lambda r: ";" if r == "none" else r)
        df['only_text'] = before_text + df['input_text'] + divider
        df['input_text'] = before_text + df['input_text'] + divider + df['target_text']
        df['only_target'] = df['target_text']
        self.texts = tokenizer(df['input_text'].values.tolist(), max_length=source_len+target_len, padding="max_length", truncation=True, return_tensors="pt")
        self.only_texts = tokenizer(df['only_text'].values.tolist(), max_length=source_len, padding="max_length", truncation=True, return_tensors="pt")
        self.only_labels = tokenizer(df['only_target'].values.tolist(), max_length=target_len, padding="max_length", truncation=True, return_tensors="pt")
    
    def __len__(self):
        return len(self.texts["input_ids"])

    def __getitem__(self,idx):
        return {
                'input_ids': self.texts['input_ids'][idx],
                'attention_mask': self.texts['attention_mask'][idx],
                'only_labels': self.only_labels['input_ids'][idx],
                'only_text': self.only_texts['input_ids'][idx],
                'only_text_mask': self.only_texts['attention_mask'][idx]
                }

# ## GPT2DatasetClass
# class GPT2Dataset(torch.utils.data.Dataset):
    # def __init__(self, df, source_len, target_len, tokenizer, divider):
        # df = df.fillna("")
        # df['target_text'] = df['target_text'].apply(lambda r: ";" if r == "none" else r)
        # df['only_text'] = df['input_text'] + divider
        # df['input_text'] = df['input_text'] + divider + df['target_text']
        # df['only_target'] = df['target_text']
        # self.texts = tokenizer(df['input_text'].values.tolist(), max_length=source_len+target_len, padding="max_length", truncation=True, return_tensors="pt")
        # self.only_texts = tokenizer(df['only_text'].values.tolist(), max_length=source_len, padding="max_length", truncation=True, return_tensors="pt")
        # self.only_labels = tokenizer(df['only_target'].values.tolist(), max_length=target_len, padding="max_length", truncation=True, return_tensors="pt")
    
    # def __len__(self):
        # return len(self.texts["input_ids"])

    # def __getitem__(self,idx):
        # return self.texts['input_ids'][idx], self.texts['attention_mask'][idx], self.only_labels['input_ids'][idx]


class BartDataset(torch.utils.data.Dataset):
    def __init__(self, df, source_len, target_len, tokenizer):
        df = df.fillna("")
        df['target_text'] = df['target_text'].apply(lambda r: ";" if r == "none" else r)
        df['input_text'] = df['input_text']
        
        self.texts = tokenizer(df['input_text'].values.tolist(), max_length=source_len, padding=True, truncation=True, return_tensors="pt")
        self.labels = tokenizer(df['target_text'].values.tolist(), max_length=target_len, padding=True, truncation=True, return_tensors="pt")

    def __len__(self):
        return len(self.texts["input_ids"])

    def __getitem__(self,idx):
        return {
                'input_ids': self.texts['input_ids'][idx],
                'attention_mask': self.texts['attention_mask'][idx],
                'labels': self.labels['input_ids'][idx]
                }
