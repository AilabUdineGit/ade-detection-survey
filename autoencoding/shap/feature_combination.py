from itertools import combinations

features = ["mlm", "rtd", "nsp", "sbo", "sop", "plm", "sob2", "shu", "ntp", "gso"]
prefix = "obj"

new_features = []
new_features.extend(features)

for i in range(2,len(features)+1):
    new_features.extend(map(lambda comb : "_".join(comb) ,combinations(features, i)))



features = [prefix+"_"+feature for feature in new_features]
numerical_features = [(i-1) / (len(features)) for i in  range(1,len(features)+1) ]


for i,f in enumerate(zip(numerical_features,features)):
    print(i,f)
