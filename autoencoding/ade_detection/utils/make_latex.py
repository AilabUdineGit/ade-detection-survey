import pandas as pd
import json

def avg(row, label, modality):
    return round(row[f"{label}_avg({modality})"]*100,4)

def std(row, label, modality):
    return round(row[f"{label}_std({modality})"]*100,4)

def make_latex(dir_path, arch, dataset):
    #df = pd.read_csv(dir_path + "final.csv")
    #df.to_pickle(dir_path + "final.pkl")
    path = dir_path + "final.pkl"

    df = pd.read_pickle(path)
    to_print = []
    to_print.append("\\begin{table*}[!hbtp]\n")
    to_print.append("\\centering\n")
    to_print.append("\\resizebox{\\linewidth}{!}{%\n")
    to_print.append("\\begin{tabular}{r|ccc|ccc}\n")
    to_print.append("& \\multicolumn{3}{c}{\\textbf{Relaxed}} & \\multicolumn{3}{c}{\\textbf{Strict}}\\\\\\hline\n")
    to_print.append("& \\textbf{F1} & \\textbf{P} & \\textbf{R} & \\textbf{F1} & \\textbf{P} & \\textbf{R}\\\\\n")
    latex_model_name = {row.model:"\\xyend" if row.model=="ENDRBERT" else f"\\{row.model.lower()}" for idx,row in df.iterrows()}
    for idx, row in df.iterrows():
        model_row = ""
        model_row += latex_model_name[row.model]

        for modality in ['r','s']:
            for label in ['f1','p','r']:
                _avg = str(avg(row,label,modality))
                _std = str(std(row,label,modality))
                if len(_std) >= 4:
                    _std = str(round(float(_std),2))
                if len(_std) < 4:
                    _std = _std+("0"*(4-len(_std)))
                if len(_avg) >= 5:
                    _avg = str(round(float(_avg),2))
                if len(_avg) < 5:
                    _avg = _avg+("0"*(5-len(_avg)))
                model_row += f" & \\asd{{{_avg}}}{{{_std}}}"
        model_row += "\\\\\n"
        to_print.append(model_row)

    to_print.append("\\end{tabular}\n")
    to_print.append("}\n")
    to_print.append(f"\\caption{{TMP:{dataset+arch}}}\n")
    to_print.append(f"\\label{{tab:{dataset+arch}}}\n")
    to_print.append("\\end{table*}\n")

    with open(path.replace("final.pkl", "final_tex.tex"), "w") as fp:
        fp.writelines(to_print)



def create_best_params_latex(path):
    models = [  "BERT",
                "SPANBERT",
                "ROBERTA",
                "ELECTRA",
                "XLNET",
                "BERTWEET",
                "BIOBERT",
                "BIOCLINICALBERT",
                "BIOROBERTA",
                "BIOCLINICALROBERTA",
                "BIOELECTRA",
                "PUBMEDBERT",
                "SCIBERT"]
    to_print = []
    to_print.append("\\begin{tabular}{r|ccccc|ccccc}\n")
    to_print.append("& \\multicolumn{5}{c}{\\textbf{CADEC}} & \\multicolumn{5}{c}{\\textbf{SMM4H20}}\\\\\\hline\n")
    to_print.append("& \\textbf{lr} & \\textbf{dropout} & \\textbf{epoch} & \\textbf{batch\\_size} & \\textbf{max\\_len} & \\textbf{lr} & \\textbf{dropout} & \\textbf{epoch} & \\textbf{batch\\_size} & \\textbf{max\\_len}\\\\\n")
    latex_model_name = {model:"\\xyend" if model=="ENDRBERT" else f"\\{model.lower()}" for mode in models}
    for model in models:
        model_row = ""
        if model != "BERTWEET":
            with open(path + f"gs_cadec_article/{model}.json","r") as fp:
                best_run_cadec = json.load(fp)
            lr          = best_run_cadec['learning_rate']
            dropout     = best_run_cadec['dropout']
            batch_size  = best_run_cadec['batch_size']
            epochs      = int(best_run_cadec['epochs'].replace(".0",""))
        else:
            lr, dropout, batch_size, epochs = "", "", "", ""
        
        model_row += f"{latex_model_name[model]} & {lr} & {dropout} & {epochs} & {batch_size} & 64"

        with open(path + f"gs_smm4h_2019_ner/{model}.json","r") as fp:
            best_run_smm4h = json.load(fp)

        lr          = best_run_smm4h['learning_rate']
        dropout     = best_run_smm4h['dropout']
        batch_size  = best_run_smm4h['batch_size']
        epochs      = int(best_run_smm4h['epochs'].replace(".0",""))

        model_row += f" & {lr} & {dropout} & {epochs} & {batch_size} & 512\\\\\n"
        to_print.append(model_row)

    to_print.append("\\end{tabular}\n")
    with open(path+"best_params_latex.tex", "w") as fp:
        fp.writelines(to_print)

#path = "/mnt/HDD/sscaboro/repo/ADE_myversion/assets/best_models/ner/bert_lstm/"
#create_best_params_latex(path)


if __name__=="__main__":
    for model in ["bert_wrapper","bert_crf","bert_lstm"]:
        for dataset in ["cadec","smm4h20"]:
            main_path = f"assets/runs/old_full_test/{model}/{dataset}/results/"
            make_latex(main_path)