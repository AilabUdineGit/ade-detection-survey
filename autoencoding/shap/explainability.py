"""
    Sources:
        https://towardsdatascience.com/explaining-scikit-learn-models-with-shap-61daff21b12a
        https://towardsdatascience.com/deep-learning-model-interpretation-using-shap-a21786e91d16
"""

import sklearn
import xgboost
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import shap
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
np.set_printoptions(formatter={'float':lambda x:"{:.4f}".format(x)})
import pandas as pd
pd.options.display.float_format = "{:.3f}".format
import seaborn as sns
import matplotlib.pyplot as plt
import os
sns.set(style='darkgrid', context='talk', palette='rainbow')



def get_models_tuned():
    # load the model with thei features
    models = pd.read_csv("models_property_2.csv")
    models.set_index("model",inplace=True)
    eval_df = pd.DataFrame({})
    # now we build the training dataset
    for dataset in ['cadec_article','smm4h_2019_ner']:
        for model, row in models.iterrows():
            # to be replaced try-catch to rapidly skip untested models
            model_path = f"../assets/runs/full_test_express5/ner/bert_wrapper/{dataset}/results/{model}.pkl"

            if not os.path.exists(model_path):
                print(f"Missing {model}")
                continue

            f1s = pd.read_pickle(model_path)['f1(r)'].values
            
            for f1 in f1s:
                new_row = row.copy()
                new_row['f1'] = f1
                new_row['dataset'] = dataset
                new_row['model'] = model
                eval_df = eval_df.append(new_row, ignore_index=True)
    return eval_df

def get_X_y(eval_df, attribute_type, dataset):
    #drop size
    cols_to_drop = [c for c in eval_df.columns if c.find("_size") != -1]
    if attribute_type == "all":
        cols_to_drop.append("dataset")#,"model"]
    else:
        cols_to_drop.extend([c for c in eval_df.columns if c.find(attribute_type) == -1 and c!="f1"])
        
    df = eval_df[ (eval_df.dataset == dataset)].drop(columns=set(cols_to_drop)-set(["model"]))
    X = df[list(set(df.columns)-set(["f1"]))]
    X = X.rename(columns=rename_columns(attribute_type))
    y = df.f1.values
    return X,y

def rename_columns(typ):
    if typ == "arc_":
        return {"arc_seq2seq": "Seq2Seq", "arc_autoreg": "Autoreg.", "arc_autoenc": "Autoenc."}
    if typ == "dom_":
        return {"dom_med": "Medical", "dom_gen": "General"}
    if typ == "obj_":
        return {"obj_mlm": "MLM", "obj_rtd": "RTD", "obj_nsp": "NSP", "obj_sop": "SOP", "obj_sbo": "SBO", "obj_plm": "PLM", "obj_sob2": "SOB2", "obj_shu": "SHU", "obj_ntp": "NTP", "obj_gso": "GSO"}
    if typ == "size_":
        return {"size_over_130": "Over 130", "size_100_110": "100 to 110", "size_120_130": "120 to 130", "size_less_100": "Less 100", "size_110_120": "110 to 120"}
    if typ == "train_":
        return {"train_from_scratch": "From scratch", "train_fine_tuned": "Fine-tuned"}
    if typ == "all":
        new_names = dict()
        for attribute_type in ["obj_","dom_","size_","arc_","train_"]:
            new_names.update(rename_columns(attribute_type))
        return new_names


<<<<<<< HEAD
def get_column_coefficients(X, model):
    print("Model coefficients:\n")
    for i in range(X.shape[1]):
        print(X.columns[i], "=", model.coef_[i].round(4))


def partition_data(X, y, divide_in_train_test=True):
    if divide_in_train_test:
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=X.model.values)
        X_train = X_train.drop(columns=["model"])
        X_test = X_test.drop(columns=["model"])
        return X_train, X_test, y_train, y_test 
    X = X.drop(columns=["model"])
    return X, None,y ,None

def beeswarm_plot(shap_values, save_folder):
    shap.plots.beeswarm(shap_values)
    plot_and_save(f'{save_folder}/beeswarm_plot.png')


def bar_plot(shap_values, save_folder):
    shap.plots.bar(shap_values)
    plot_and_save(f'{save_folder}/bar_plot.png')


def waterfall_plot(shap_values, save_folder):
    shap.plots.waterfall(shap_values[0])
    plot_and_save(f'{save_folder}/summary.png')


def summary_plot(shap_values, save_folder):
    shap.summary_plot(shap_values)
    plot_and_save(f'{save_folder}/summary.png')


def summary_violin_plot(shap_values, save_folder):
    shap.summary_plot(shap_values, plot_type='violin')
    plot_and_save(f'{save_folder}/summary_violin.png')


def heatmap_plot(shap_values, save_folder):
    shap.plots.heatmap(shap_values)
    plot_and_save(f'{save_folder}/heatmap.png')


def plot_and_save(path):
    w, _ = plt.gcf().get_size_inches()
    plt.gcf().set_size_inches(w, w*3/4)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.clf()


def get_S_model(model_type):
    if model_type == "linear":
        return sklearn.linear_model.LinearRegression()
    if model_type == "random_forest":
        return sklearn.ensemble.RandomForestRegressor(random_state=42)
    if model_type == "boost":
        return xgboost.XGBRegressor()

    raise Exception("No model selected")

if __name__=="__main__":
    eval_df = get_models_tuned() 
    model_type = "boost"
    use_test_set = True
    for attribute_type in ["obj_","dom_","arc_","train_", "all"]:
        for dataset in ['smm4h_2019_ner','cadec_article']:

            #if attribute_type == "all":
            #    eval_df = eval_df[[c for c in eval_df.columns if c.find("size_") == -1]]
            X,y = get_X_y(eval_df, attribute_type, dataset)
            df = X.copy()
            df["f1"] = y
            df["f1"] = df["f1"].astype("float64")
            df = df[df.f1 > 1e-6]
            df = df[df.model != "GPT2"]
            df.to_csv(f"complete_dataset_{dataset}.csv")
            df.to_pickle(f"complete_dataset_{dataset}.pkl")
            X = df[list(set(df.columns)-set(["f1"]))]
            y = df.f1.values
            # X.to_csv("complete_dataset_{model_type}_{dataset}_{attribute_type}.csv")
            X_train, X_test, y_train, y_test = partition_data(X,y,use_test_set)
            
            model = get_S_model(model_type) 
            model.fit(X_train, y_train)
            
            # compute the SHAP values for the linear model
            explainer = shap.Explainer(model)
            #explainer = shap.explainers.Linear(model.predict, X_test)
            #if use_test_set:
            shap_values = explainer(X_test)
            #else:
            #    shap_values = explainer(X_train)

            save_folder = f"plots/{model_type}/{dataset}/{attribute_type}"
            os.makedirs(save_folder, exist_ok=True)

            bar_plot(shap_values, save_folder)
            waterfall_plot(shap_values, save_folder)
            summary_plot(shap_values, save_folder)
            summary_violin_plot(shap_values, save_folder)
            heatmap_plot(shap_values, save_folder)
            beeswarm_plot(shap_values, save_folder)
