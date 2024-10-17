import pandas as pd

# reading prolific file
R_first = pd.read_csv("R_first20.csv",skiprows=[1,2])
R_first.drop(R_first[(R_first.polAffil=="Independent")|(R_first.polAffil=="Democrat")|(R_first.polAffil=="None")|(R_first.polAffil=="Other")].index, inplace=True)
R_first["survey"] = ["R_first"]*len(R_first)

# reading prolific file
D_first = pd.read_csv("D_first20.csv",skiprows=[1,2])
D_first.drop(D_first[(D_first.polAffil=="Independent")|(D_first.polAffil=="Republican")|(D_first.polAffil=="None")|(D_first.polAffil=="Other")].index, inplace=True)
D_first["survey"] = ["D_first"]*len(D_first)

# reading prolific file
D = pd.read_csv("D_June 10, 2022_01.55.csv",skiprows=[1,2])
D.drop(D[(D.polAffil=="Independent")|(D.polAffil=="Republican")|(D.polAffil=="None")|(D.polAffil=="Other")].index, inplace=True)
D["survey"] = ["D_main"]*len(D)

# reading prolific file
R = pd.read_csv('R_June 24, 2022_09.02.csv',skiprows=[1,2])
R.drop(R[(R.polAffil=="Independent")|(R.polAffil=="Democrat")|(R.polAffil=="None")|(R.polAffil=="Other")].index, inplace=True)
R["survey"] = ["R_main"]*len(R)

temp = pd.concat([R_first,R,D_first,D],ignore_index=True)

repeated = [key for key, value in temp.PROLIFIC_PID.value_counts().items() if value>1]

D = D[~D.PROLIFIC_PID.isin(repeated)]
R = R[~R.PROLIFIC_PID.isin(repeated)]

data = pd.concat([R_first,R,D_first,D],ignore_index=True)
data.dropna(inplace=True)
data = data[data.DistributionChannel!="preview"]
#data.drop(data[(data.polAffil=="Independent")|(data.polAffil=="None")|(data.polAffil=="Other")].index, inplace=True)
data.reset_index(drop=True,inplace=True)

data["ID"] = data.index

data.rename(columns = {'Q5_1':'Q3_1', 'Q3_1':'Q4_1','Q4_1':'Q5_1'}, inplace = True)
data.rename(columns = {'Q5_2_1':'Q3_2_1', 'Q3_2_1':'Q4_2_1','Q4_2_1':'Q5_2_1'}, inplace = True)
data.rename(columns = {'Q5_2_2':'Q3_2_2', 'Q3_2_2':'Q4_2_2','Q4_2_2':'Q5_2_2'}, inplace = True)

data["Q1_midpoint"] = (data["Q1_2_3"]+data["Q1_2_4"])/2
data["Q2_midpoint"] = (data["Q2_2_1"]+data["Q2_2_2"])/2
data["Q3_midpoint"] = (data["Q3_2_1"]+data["Q3_2_2"])/2
data["Q4_midpoint"] = (data["Q4_2_1"]+data["Q4_2_2"])/2
data["Q5_midpoint"] = (data["Q5_2_1"]+data["Q5_2_2"])/2
data["Q6_midpoint"] = (data["Q6_2_1"]+data["Q6_2_2"])/2
data["Q1_width"] = data["Q1_2_4"]-data["Q1_2_3"]
data["Q2_width"] = data["Q2_2_2"]-data["Q2_2_1"]
data["Q3_width"] = data["Q3_2_2"]-data["Q3_2_1"]
data["Q4_width"] = data["Q4_2_2"]-data["Q4_2_1"]
data["Q5_width"] = data["Q5_2_2"]-data["Q5_2_1"]
data["Q6_width"] = data["Q6_2_2"]-data["Q6_2_1"]
data["width_median"] = data[["Q1_width","Q2_width","Q3_width","Q4_width","Q5_width","Q6_width"]].median(axis=1)
data["width_mean"] = data[["Q1_width","Q2_width","Q3_width","Q4_width","Q5_width","Q6_width"]].mean(axis=1)

data.to_csv("study1data.csv")

# check nans
print(data[data.isna().any(axis=1)])

repeated = [key for key, value in data.PROLIFIC_PID.value_counts().items() if value>1]
print(len(repeated))

print(data[data.PROLIFIC_PID.isin(repeated)].sort_values(by="PROLIFIC_PID").survey.values)
