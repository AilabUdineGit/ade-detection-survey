import pandas as pd
import os
import numpy as np
from make_latex import make_latex

def get_mean_sd(df, metric, str_rel):
    values = [float(s) for s in df[f"{metric}({str_rel})"].values]
    arr = np.array(values)
    return np.mean(arr, axis=0), np.std(arr, axis=0)

def get_row(model,dataset,architecture):
    results_path = f"/media/HDD/sscaboro/ADE_detection/assets/results/full_test/{architecture}/{dataset}/"
    df = pd.read_csv(results_path + f"{model}.csv")
    df.to_pickle(results_path + f"{model}.pkl")
    df = pd.read_pickle(
        results_path + f"{model}.pkl"
        )

    final_path = results_path + "final.pkl"
    if not os.path.isfile(final_path):
        final = pd.DataFrame({})
        final.to_pickle(final_path)

    df_final = pd.read_pickle(final_path)

    df_final = df_final.append(pd.Series({
        'model': model,
        'split_folder': dataset,
        'f1_avg(r)':    get_mean_sd(df,"f1","r")[0],
        'f1_std(r)':    get_mean_sd(df,"f1","r")[1],
        'p_avg(r)':     get_mean_sd(df,"p","r")[0],
        'p_std(r)':     get_mean_sd(df,"p","r")[1],
        'r_avg(r)':     get_mean_sd(df,"r","r")[0],
        'r_std(r)':     get_mean_sd(df,"r","r")[1],
        'f1_avg(s)':    get_mean_sd(df,"f1","s")[0],
        'f1_std(s)':    get_mean_sd(df,"f1","s")[1],
        'p_avg(s)':     get_mean_sd(df,"p","s")[0],
        'p_std(s)':     get_mean_sd(df,"p","s")[1],
        'r_avg(s)':     get_mean_sd(df,"r","s")[0],
        'r_std(s)':     get_mean_sd(df,"r","s")[1]
    }), ignore_index=True)

    df_final.to_csv(final_path.replace("pkl","csv"))
    df_final.to_pickle(final_path)
    make_latex(results_path,"","")


if __name__=="__main__":
    arch = "bert_lstm"
    for model in ["ENDRBERT"]:
        for dataset in ["cadec"]:
            get_row(model,dataset,arch)