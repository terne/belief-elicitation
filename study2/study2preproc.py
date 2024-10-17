import pandas as pd
#import numpy as np

# Reading Prolific files
D = pd.read_csv("D_2_September 20, 2022_07.00.csv",skiprows=[1,2])
R = pd.read_csv("R_2_September 21, 2022_02.10.csv",skiprows=[1,2])


print(D.columns)
assert D.columns.values.tolist() == R.columns.values.tolist()
#print(D.polAffil.value_counts())

D.drop(D[(D.polAffil=="Independent")|(D.polAffil=="Republican")|(D.polAffil=="None")|(D.polAffil=="Other")].index, inplace=True)

R.drop(R[(R.polAffil=="Independent")|(R.polAffil=="Democrat")|(R.polAffil=="None")|(R.polAffil=="Other")].index, inplace=True)
print(len(R))

data = pd.concat([D,R],ignore_index=True)
data = data[data.DistributionChannel!="preview"]


data.dropna(inplace=True)
data.reset_index(drop=True,inplace=True)
#data["ID"] = data.index


data["Q1_D_midpoint"] = (data["Q1_2_D_3"]+data["Q1_2_D_4"])/2
data["Q1_R_midpoint"] = (data["Q1_2_R_3"]+data["Q1_2_R_4"])/2
data["Q1_midpoint"] = (data["Q1_D_midpoint"]+data["Q1_R_midpoint"])/2

data["Q2_D_midpoint"] = (data["Q2_2_D_1"]+data["Q2_2_D_2"])/2
data["Q2_R_midpoint"] = (data["Q2_2_R_1"]+data["Q2_2_R_2"])/2
data["Q2_midpoint"] = (data["Q2_D_midpoint"]+data["Q2_R_midpoint"])/2

data["Q4_D_midpoint"] = (data["Q4_2_D_1"]+data["Q4_2_D_2"])/2
data["Q4_R_midpoint"] = (data["Q4_2_R_1"]+data["Q4_2_R_2"])/2
data["Q4_midpoint"] = (data["Q4_D_midpoint"]+data["Q4_R_midpoint"])/2

data["Q5_D_midpoint"] = (data["Q5_2_D_1"]+data["Q5_2_D_2"])/2
data["Q5_R_midpoint"] = (data["Q5_2_R_1"]+data["Q5_2_R_2"])/2
data["Q5_midpoint"] = (data["Q5_D_midpoint"]+data["Q5_R_midpoint"])/2

print(data.head())
print(data.polAffil.value_counts())
print(len(data))
print(len(data.PROLIFIC_PID.unique()))



data.to_csv("study2data.csv")