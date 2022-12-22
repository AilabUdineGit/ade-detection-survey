import pandas as pd

print("2020")
# SMM4H_2019_NER --> 2020
df = pd.read_csv("/mnt/HDD/sscaboro/repo/ADE_myversion/assets/datasets/task1_2020_2019/task1_2020_2019_complete_dataset2.csv")
df2 = pd.read_csv("/mnt/HDD/sscaboro/repo/ADE_myversion/assets/datasets/smm4h19_task1/smm4h_task1_tweets_all.txt",sep="\t")
df2.to_csv("/mnt/HDD/sscaboro/repo/ADE_myversion/assets/datasets/smm4h19_task1/smm4h_task1_tweets_all.csv")
df_ids = set([str(a) for a in df.tweet_id.values])
df_ids2 = set([str(a) for a in df2.tweet_id.values])

path_split = "/mnt/HDD/sscaboro/repo/ADE_myversion/assets/splits/nade/"

with open(path_split+"negade.id","r") as fp:
    negade = set([int(a.replace("\n","")) for a in fp.readlines()])

df = df[df.tweet_id.isin(negade)].drop(columns=["id"])
df2 = df2[df2.tweet_id.isin(negade)].rename(columns={'boh': "user_id","ade_or_not":"is_ade"})
print(df.columns)
print(df2.columns)

print(len(df))
print(len(df2))
df = pd.concat([df,df2])
print(len(df))
df.to_csv("utils/negade_nade_samples.csv")

print("rows: ", len(df_ids))
print("rows: ", len(df_ids2))
print("ids: ", len(negade))

print("ids in rows", len(negade.intersection(df_ids)))
print("ids2 in rows", len(negade.intersection(df_ids2)))
print("ids2 in ids", len(df_ids.intersection(df_ids2)))


