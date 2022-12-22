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
import seaborn as sns
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


def beeswarm_plot(shap_values, save_folder):
    shap.plots.beeswarm(shap_values, alpha=0.05)
    plot_and_save(f'{save_folder}/beeswarm.png')


def bar_plot(shap_values, save_folder):
    shap.plots.bar(shap_values)
    plot_and_save(f'{save_folder}/bar.png')


def waterfall_plot(shap_values, save_folder):
    shap.plots.waterfall(shap_values[0])
    plot_and_save(f'{save_folder}/waterfall.png')


def summary_plot(shap_values, save_folder):
    shap.summary_plot(shap_values)
    plot_and_save(f'{save_folder}/summary.png')


def summary_violin_plot(shap_values, save_folder):
    shap.summary_plot(shap_values, plot_type='violin')
    plot_and_save(f'{save_folder}/summary_violin.png')


def heatmap_plot(shap_values, save_folder):
    shap.plots.heatmap(shap_values)
    plot_and_save(f'{save_folder}/heatmap.png')


def dependence_plot(shap_values, feature1, feature2, X,save_folder):
    shap.dependence_plot(ind=feature1, shap_values=shap_values, features=X, interaction_index=feature2)
    plot_and_save(f'{save_folder}/dependence.png')


def plot_and_save(path):
    w, _ = plt.gcf().get_size_inches()
    plt.gcf().set_size_inches(w*2, w*3/4)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.clf()


def get_models_tuned():
    # load the model with thei features
    models = pd.read_csv("models_property_7.csv")
    models.set_index("model",inplace=True)
    eval_df = pd.DataFrame({})
    # now we build the training dataset

    datasets = ['smm4h20','cadec']
    #datasets = ['smm4h20']

    for dataset in datasets:
        for model, row in models.iterrows():
            # to be replaced try-catch to rapidly skip untested models
            model_path = f"../assets/results/full_test_express6/bert_wrapper/{dataset}/{model}.pkl"

            if not os.path.exists(model_path):
                print(f"Missing {model} for {dataset}")
                continue

            f1s = pd.read_pickle(model_path)['f1(r)'].values
            
            for f1 in f1s:
                new_row = row.copy()
                new_row['f1'] = f1
                new_row['dataset'] = dataset
                new_row['model'] = model
                eval_df = eval_df.append(new_row, ignore_index=True)
    
    return eval_df


def get_X_y(eval_df, dataset):
    #drop size
    #cols_to_drop = [c for c in eval_df.columns if c.find("_size") != -1]
    cols_to_drop = []
    cols_to_drop.append("dataset")

    df = eval_df.drop(columns=set(cols_to_drop)-set(["model"]))

    X = df[list(set(df.columns)-set(["f1"]))]
    #X = X.rename(columns=rename_columns(attribute_type))
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


def partition_data(X, y, use_test_set=True):
    if use_test_set:
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=X.model.values, test_size=0.50)
        X_train = X_train.drop(columns=["model"])
        X_test = X_test.drop(columns=["model"])
        res = X_train, X_test, y_train, y_test 
    else:
        X = X.drop(columns=["model"])
        res = X, X, y ,y 
        
    return res 



def get_S_model(model_type):
    if model_type == "linear":
        return sklearn.linear_model.LinearRegression()
    if model_type == "random_forest":
        return sklearn.ensemble.RandomForestRegressor(random_state=42)
    if model_type == "boost":
        return xgboost.XGBRegressor()

    raise Exception("No model selected")



if __name__ == "__main__":
    eval_df_original = get_models_tuned() 
    model_type = "boost"
    use_test_set = False

    datasets = ['smm4h20','cadec']
    #datasets = ['smm4h20']

    for dataset in datasets:
        eval_df = eval_df_original.query(f"dataset == \"{dataset}\"")

        X,y = get_X_y(eval_df, dataset)
        df = X.copy()

        df["f1"] = y
        df["f1"] = df["f1"].astype("float64")

        df = df[df.f1 > 1e-6]

        if dataset == "cadec":
            # to_exlude = ["GPT2", "ALBERT", 'PEGASUS', 'T5', 'BART', 'SCIFIVE']
            #to_exlude = ["GPT2", "ALBERT"]
            to_exlude = ["ALBERT"]
        if dataset == 'smm4h20':
            to_exlude = ["ALBERT"]
            #to_exlude = ["GPT2", "ALBERT"]

        df = df[~df.model.isin(to_exlude)]

        print(f"Models: {len(df.model.unique())}")
        print(df.groupby("model").f1.count())
    
        df.to_csv(f"complete_dataset_{dataset}.csv")
        df.to_pickle(f"complete_dataset_{dataset}.pkl")

        X = df[list(set(df.columns)-set(["f1"]))]
        y = df.f1.values

        X_train, X_test, y_train, y_test = partition_data(X,y,use_test_set)
        
        model = get_S_model(model_type) 
        model.fit(X_train, y_train)
        
        # compute the SHAP values for the linear model
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_test)

        save_folder = f"plots_9/{model_type}/{dataset}"
        os.makedirs(save_folder, exist_ok=True)

        bar_plot(shap_values, save_folder)
        #waterfall_plot(shap_values, save_folder)
        summary_plot(shap_values, save_folder)
        summary_violin_plot(shap_values, save_folder)
        heatmap_plot(shap_values, save_folder)
        beeswarm_plot(shap_values, save_folder)

        from itertools import repeat, chain
        revert_dict = lambda d: dict(chain(*[zip(val, repeat(key)) for key, val in d.items()]))
        
        features = list(set(df.reset_index().columns) - set(["model","f1","index"]))

        def grouped_shap(shap_vals, features, groups):
            groupmap = revert_dict(groups)
            print(shap_vals)
            shap_Tdf = pd.DataFrame(shap_vals, columns=pd.Index(features, name='features')).T
            shap_Tdf['group'] = shap_Tdf.reset_index().features.map(groupmap).values
            shap_grouped = shap_Tdf.groupby('group').sum().T
            return shap_grouped

        shap_vals = explainer.shap_values(X_test)
        shap_df = pd.DataFrame(shap_vals, columns=pd.Index(features, name='features'))

        print(shap_df.head())

        groups = dict(
            architecture = ["arc_dec","arc_enc","arc_s2s"],
            size = ["size_less_100","size_between_100_130","size_over_130"],
            domain = ["dom_general","dom_social","dom_medical"],
            train = ["train_scratch","train_fine_tuning"]
        )

        #shap_time = grouped_shap(shap_vals, features, groups)
        #shap.summary_plot(shap_time.values, features=shap_time.columns)
        #plot_and_save(f"{save_folder}/prova.png")

        feat_order = shap_df.abs().mean().sort_values().index.tolist()[::-1]
        sns.heatmap(shap_df.corr().abs().loc[feat_order, feat_order], cbar=False, cmap="crest")
        plot_and_save(f"{save_folder}/correlation.png")