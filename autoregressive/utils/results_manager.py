import pandas as pd
from typing import Dict
from copy import copy
from tabulate import tabulate
from utils.evaluator import EvaluatorNER
from utils.interval_merger import IntervalMerger, merge_overlaps
from utils.shivam_metrics import *

def compute_best_metrics(results: pd.DataFrame, run=None) -> Dict[str,Dict[str,float]]:
    """
        Given a DataFrame, returns the best epoch results

        Parameters
        ----------
        results : pd.DataFrame
        run:
            Specify only if you want to save the complete metrics

        Returns
        -------
        metrics : Dict[str,Dict[str,float]]
            Dictionary with partial and strict metrics for the best epoch
    """
    # compute overfitting
    results['overfitting'] = results.apply(lambda row: is_overfitting(row.train_loss, row.test_loss), axis=1)
    
    if run:
        results.to_csv(f"assets/results/MET-NEW-{run.id}.csv")

    # drop the rows where the overfitting is == True
    results.drop(results[results.overfitting].index, inplace=True)
    
    if run.train_mode == "VALIDATION" or run.train_mode == "GRID_SEARCH":
        # take the best iteration based on F1 partial
        relevant_attribute = "f1_par"
    else:
        relevant_attribute = "epoch"
    
    max_row = results[relevant_attribute].idxmax()
    best = results.loc[max_row]

    rows = [
            ["p", best['precision_par'], best['(sh)precision_par']],
            ["r", best['recall_par'], best['(sh)recall_par']],
            ["f", best['f1_par'], best['(sh)f1_par']]]
    print()
    print(tabulate(rows, headers=["", "ours", "shs"]))
    print()

    best = get_dict_from_series(best)
    return best


def get_dict_from_series(s: pd.Series) -> Dict[str,Dict[str,float]]:
    """ Transform a pd.Series in a dictionary """

    metrics_names = ['precision','recall','f1']
    metrics = {m:0 for m in metrics_names}
    new_dics = {'partial': copy(metrics), 'strict': copy(metrics)}
    
    for m_kind,m in zip(['str','par'],['strict','partial']):
        for m_name in metrics_names:
            new_dics[m][m_name] = s[f"{m_name}_{m_kind}"]
        new_dics[m]['epochs'] = s['epoch']
    return new_dics


def is_overfitting(tr_loss, vl_loss):
    """ Check if there is overfitting """
    return vl_loss/tr_loss >= 10


def compute_metrics(results: pd.DataFrame) -> pd.DataFrame:
    """
        Given the results dataframe, returns a dataframe where per each epoch there is the metrics
    """
    df = pd.DataFrame({})
    for epoch,df_epoch in results.groupby("epoch"):
        results = pd.Series() 
        df_epoch.reset_index(inplace=True)
        metrics,fuzzy_metrics,_ = get_metrics(df_epoch[['text','gold','pred']])
        
        results['epoch'] = epoch
        results['train_loss'] = df_epoch.loc[0]['train_loss']
        results['test_loss'] = df_epoch.loc[0]['test_loss']
        
        results['f1_str'] = metrics['strict']['f1']
        results['precision_str'] = metrics['strict']['precision']
        results['recall_str'] = metrics['strict']['recall']

        results['f1_par'] = metrics['partial']['f1']
        results['precision_par'] = metrics['partial']['precision']
        results['recall_par'] = metrics['partial']['recall']

        results['(sh)f1_str'] = fuzzy_metrics['strict']['f1']
        results['(sh)precision_str'] = fuzzy_metrics['strict']['precision']
        results['(sh)recall_str'] = fuzzy_metrics['strict']['recall']

        results['(sh)f1_par'] = fuzzy_metrics['partial']['f1']
        results['(sh)precision_par'] = fuzzy_metrics['partial']['precision']
        results['(sh)recall_par'] = fuzzy_metrics['partial']['recall']
         
        df = df.append(results, ignore_index=True)
    return df


def get_word_pos(preds,text):
    spans = []
    for word in preds.split(";"):
        word = word.strip()
        s = text.find(word)
        # prediction founded in the text
        if s != -1:
            e = s + len(word)
            spans.append((s,e))
        # otherwise...
        else:
            # split each prediction in single words
            for subword in word.split(" "):
                subword = subword.strip()
                s = text.find(subword)
                if s != -1:
                    e = s + len(subword)+1
                # skip if the word is not in the text
                else:
                    continue
                spans.append((s,e))
    return spans


def create_metrics_series(metrics, fuzzy_metrics) -> pd.Series:
    results = pd.Series() 
        
    results['f1_str'] = metrics['strict']['f1']
    results['precision_str'] = metrics['strict']['precision']
    results['recall_str'] = metrics['strict']['recall']

    results['f1_par'] = metrics['partial']['f1']
    results['precision_par'] = metrics['partial']['precision']
    results['recall_par'] = metrics['partial']['recall']

    results['(sh)f1_str'] = fuzzy_metrics['strict']['f1']
    results['(sh)precision_str'] = fuzzy_metrics['strict']['precision']
    results['(sh)recall_str'] = fuzzy_metrics['strict']['recall']

    results['(sh)f1_par'] = fuzzy_metrics['partial']['f1']
    results['(sh)precision_par'] = fuzzy_metrics['partial']['precision']
    results['(sh)recall_par'] = fuzzy_metrics['partial']['recall']
    
    return results

def get_metrics(df: pd.DataFrame) -> Dict[str,Dict[str,float]]:
    """ compute f1, precision and recall diven a dataframe """
    def add_spans(row):
        merger = IntervalMerger()
        row["gold"] = row["gold"] if not pd.isna(row["gold"]) else ";"
        row["pred"] = row["pred"] if not pd.isna(row["pred"]) else ";"
        # row["pred_spans"] = get_word_pos(row.pred,row.text)
        row["pred_spans"] = merger.merge(get_word_pos(row.pred,row.text))
        row["gold_spans"] = merge_overlaps(get_word_pos(row.gold,row.text))
        spans = row["gold_spans"]
        if any([any([len(set(spans[i]).intersection(set(spans[j]))) != 0 for j in range(i+1,len(spans))]) for i in range(len(spans)-1)]):
            # print(spans)
            pass
        return row

    df = df.apply(lambda row: add_spans(row), axis=1)    
    # from a text and a set of words, get the character indexes
    #get_word_pos = lambda p,t : [(t.find(word.strip()),t.find(word.strip())+len(word.strip())) for word in p.split(";") if t.find(word.strip()) != -1]
    
    # get spans to compute our metrics
    #df['pred_spans'] = df.apply(lambda row: IntervalMerger().merge(get_word_pos(row.pred,row.text)), axis=1).copy()
    #df['gold_spans'] = df.apply(lambda row: IntervalMerger().merge(get_word_pos(row.gold,row.text)), axis=1).copy()

    # get column with a specific name to compute shivam's metrics
    df['pred_str'] = df['pred']
    df['gold_str'] = df['gold']

    # compute our metrics
    standard_metrics= EvaluatorNER(df, "pred_spans", "gold_spans").get() 
    # compute shivam's metrics
    fuzzy_metrics = get_fuzzy_metrics(df)
    return standard_metrics, fuzzy_metrics,df
