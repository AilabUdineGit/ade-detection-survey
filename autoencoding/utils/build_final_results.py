#!/usr/bin/env python
# coding: utf-8

# In[ ]:




# In[1]:


import pickle5
from tqdm import tqdm
import sys
from wasabi import msg

if len(sys.argv) == 1:
    raise Exception("You must specify the file")

NAME = sys.argv[1]

all_doc_dict = dict()

import pickle5
with open(f"tmp/{NAME}.pickle", "rb") as f:
    data = pickle5.load(f)
data

#-----------------------------------------------------------------------

outpath = f"{NAME}.txt"

import unicodedata

def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    only_ascii = nfkd_form.encode('ASCII', 'ignore').decode()
    return only_ascii

data.split.test[0].subtokens[1].token.text

#-----------------------------------------------------------------------

def find_subdoc(idx):
    for item in data.split.test:
        if item.id == idx:
            return item

int_2_iob = {
    0: "O",
    1: "I",
    2: "B",
}

#-----------------------------------------------------------------------

def get_max_pred(p1, p2):
    if "B" in [p1,p2]:
        return "B"
    elif "I" in [p1,p2]:
        return "I"
    else:
        return "O"

# In[2]:


def overwrite(s, new, start, end):
    s = s[:start] + new + s[end:]
    return s

all_docs = []
final_preds = []

PREDS_IDX = data.val_df.columns.tolist().index("detok_gold_labels")-1

for idx, line in data.val_df.iterrows():
    PRINT = False
    if PRINT: print(idx)
    sent = line.input_mask 
    #if "[CLS]" not in sent:
        # print(sent)
        #sent = ["[CLS]", sent[0][5:]]+sent[1:]
        # print(sent)
    #start = sent.index("[CLS]")+1 
    #end = sent.index("[SEP]")
    start = 1
    end = sent.index(0)-1 if 0 in sent else -1
    sent = sent[start:end] if end != -1 else sent[start:]
    preds = line.iat[PREDS_IDX]
    preds = preds[start:end] if end != -1 else preds[start:]
    preds = [int_2_iob[p] for p in preds]
    if PRINT: print(sent)
    sdoc = find_subdoc(idx)
    tweet_id = sdoc.doc.external_id
    doc_id = sdoc.doc.id
    
    all_doc_dict[tweet_id] = sdoc.doc.text
        
    if PRINT: print(sdoc.doc.id, tweet_id)
    
    sdoc_tokens = []
    
    subtoks = [x for x in sdoc.subtokens if x.token is not None]
    a = [x.token for x in subtoks]
    
    text = " " * subtoks[-1].token.end
    
    new_sent = []
    new_preds = []
    
    last_tok = None
    last_pred = None

    for tok, pred in zip(a, preds):
        text = overwrite(text, tok.original_text, tok.begin, tok.end)
        if last_tok is not None and tok.id == last_tok.id:
            pred = get_max_pred(pred, last_pred)
            new_preds[-1] = pred
        else:
            new_sent.append(tok)
            new_preds.append(pred)
        last_tok = tok
        last_pred = pred
    
    #if len(sdoc.text) == 0:
        #print(text)
        #print(sdoc.text)
        #print("----")
        #break
    #if text.strip() != sdoc.text.strip():
        #print("|"+text+"|")
        #print("|"+sdoc.text+"|")
        #print()
    #assert text == sdoc.text
    
    if PRINT: print([x.original_text for x in new_sent])
    if PRINT: print(new_preds)
    if PRINT: print(text)
        
    assert len(new_sent) == len(new_preds)
    
    preds = new_preds
    sdoc_tokens = new_sent
    
    pred_ents_token_lvl = []
    pred_ents_char_lvl = []

    tmp_ent = False
    i=0
    while i<len(preds):
        if preds[i] == "B": #or preds[i]=="I":
            start_ent = i
            i += 1
            while i<len(preds) and preds[i] == "I":
                i += 1
            end_ent = i
            pred_ents_token_lvl.append((start_ent,end_ent))
        else:
            i += 1

    for s,e in pred_ents_token_lvl:
        toks = sdoc_tokens[s:e]
        s_c = toks[0].begin
        e_c = toks[-1].end
        pred_ents_char_lvl.append((s_c, e_c))

        final_preds.append((sdoc.doc.external_id, s_c, e_c, sdoc.doc.text[s_c:e_c]))

        #if doc_id not in all_docs:
        #    print("1)",sdoc.doc.text[s_c:e_c])
        #    print("2)"," ".join(sent[s:e]))
        #    print("3)"," ".join([t.text for t in sdoc_tokens][s:e]))
        #    print()

    all_docs.append(doc_id)

# In[7]:


import pandas as pd
df = pd.DataFrame({"tweet_id": list(all_doc_dict.keys()), "text": list(all_doc_dict.values())})
#df

# In[6]:


final_preds_2 = []

for prediction in final_preds:
    text_id, start, end, pred_text = prediction
    full_text = df[df.tweet_id == text_id].text.iloc[0]
    doc_text = full_text[start:end]
    assert doc_text == pred_text
    final_preds_2.append((text_id, start, end, pred_text, full_text))

all_pred_ids = set([p[0] for p in final_preds_2])

#-----------------------------------------------------------------------

df = pd.DataFrame(final_preds_2, dtype="str", columns=["doc_id", "start", "end", "extraction", "text"])

#df.to_csv(f"_RESULTS/{NAME}.csv")
df.to_pickle(f"_RESULTS/neg_spec/{NAME}.pickle")
df.to_csv(f"_RESULTS/neg_spec/{NAME}.csv")

keys = df.doc_id.unique()
new_df = pd.DataFrame({})
for key in keys:
    tmp = df[df.doc_id == key]
    new_df = new_df.append(pd.Series({
        "doc_id": key, 
        "text": tmp.text.values[0],
        "extraction": tmp.extraction.values,
        "predicted_intervals": list(zip([int(s) for s in tmp.start.values], [int(e) for e in tmp.end.values]))}),ignore_index=True)

new_df.to_pickle(f"_RESULTS/neg_spec/cl_{NAME}.pickle")
new_df.to_csv(f"_RESULTS/neg_spec/cl_{NAME}.csv")
