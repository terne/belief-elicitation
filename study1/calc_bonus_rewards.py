import pandas as pd
import numpy as np

data = pd.read_csv("study1data.csv")

def S(x,L,U):
    a = 0
    b = 1
    lamb = 0.5
    g = (1-lamb)/lamb
    W = U-L
    if L<=x<=U:
        return (1-(W/(b-a)))**g
    else:
        return 0

def S_simple(x,L,U):
    # The function can be simplified to this
    W = U-L
    if L<=x<=U:
        return 1-(W/1)
    else:
        return 0

lower_bound_columns =  ['Q1_2_3', 'Q2_2_1', 'Q3_2_1',
                        'Q4_2_1', 'Q5_2_1', 'Q6_2_1']

upper_bound_columns = ['Q1_2_4',  'Q2_2_2','Q3_2_2',
                       'Q4_2_2', 'Q5_2_2', 'Q6_2_2']


averages = [data.Q1_1.mean(), data.Q2_1.mean(), data.Q3_1.mean(),
           data.Q4_1.mean(), data.Q5_1.mean(), data.Q6_1.mean()]

Raverages = [data[data.polAffil=="Republican"].Q1_1.mean(), 
             data[data.polAffil=="Republican"].Q2_1.mean(), 
             data[data.polAffil=="Republican"].Q3_1.mean(),
             data[data.polAffil=="Republican"].Q4_1.mean(), 
             data[data.polAffil=="Republican"].Q5_1.mean(), 
             data[data.polAffil=="Republican"].Q6_1.mean()]

Daverages = [data[data.polAffil=="Democrat"].Q1_1.mean(), 
             data[data.polAffil=="Democrat"].Q2_1.mean(), 
             data[data.polAffil=="Democrat"].Q3_1.mean(),
             data[data.polAffil=="Democrat"].Q4_1.mean(), 
             data[data.polAffil=="Democrat"].Q5_1.mean(), 
             data[data.polAffil=="Democrat"].Q6_1.mean()]


print(averages) 

print(Raverages) 
print(Daverages)

len(data)
len(data[data.Condition==1])

np.random.seed(22)

choosing_type = "out of all"

if choosing_type == "out of all":
    total_rewards = 0
    rewards = []
    rewards_D = []
    rewards_R = []
    winners_D = []
    winners_R = []
    bonus_winners = np.random.choice(data[data.Condition==1].PROLIFIC_PID, 
                                     size=int(len(data[data.Condition==1].PROLIFIC_PID)/25),
                                     replace=False)
    
    #print(bonus_winners)
    
    for i in bonus_winners:
        rs = []
        for indel in range(6):
            L = data[data.PROLIFIC_PID==i][lower_bound_columns[indel]].values[0]
            U = data[data.PROLIFIC_PID==i][upper_bound_columns[indel]].values[0]
            s = S_simple(averages[indel],L,U)*10
            rs.append(s)
        reward = np.round(np.random.choice(rs),1)
        poltemp = data[data.PROLIFIC_PID==i].polAffil.values[0]
        if poltemp=="Democrat":
            rewards_D.append(reward)
            winners_D.append(i)
        else:
            rewards_R.append(reward)
            winners_R.append(i)
        #print(rs)
        rewards.append(reward)
        total_rewards+=reward
    #print(total_rewards)
#print(rewards)
#print(len(rewards))
#print(len(bonus_winners))