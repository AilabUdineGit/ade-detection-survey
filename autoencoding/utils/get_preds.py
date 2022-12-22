#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Edoardo Lenzi, Beatrice Portelli, Simone Scaboro'
__version__ = '1.0'
__license__ = '???'
__copyright__ = '???'

from tqdm import tqdm
import argparse
import os
import pandas as pd

import sys
sys.path.append("..\\ADE_core")
sys.path.append("../ADE_core")

# load .env configs
from ade_detection.utils.env import Env
Env.load()

Env.set_value(Env.DB, Env.TEST_DB)


parser = argparse.ArgumentParser()
parser.add_argument('-P', '--path', type=str, help="Path to the pickled Task")

args = parser.parse_args()

data = pd.read_pickle(args.path)

df = data.val_df
list_of_names = [f'preds_{e+1}' for e in range(data.train_config.epochs)]
list_of_names = list(filter(lambda x: x in df.columns, list_of_names))


BASE = f"out/{data.id}/"
if not os.path.exists(BASE):
    os.makedirs(BASE)

for name in tqdm(list_of_names):
    
    result_df = []
    
    for doc in data.split.test:
        int_id = doc.id
        ext_id = doc.doc.external_id
        full_text = doc.text
        row = df.loc[int(int_id)]

        result_df.append({
            "doc_id": ext_id,
            "text": full_text,
            "gold_class": row.gold_labels,
            "pred_class": row[name],
        })
    
    result_df = pd.DataFrame(result_df)
    result_df.to_csv(f"{BASE}{name}.tsv", sep="\t", index=None)