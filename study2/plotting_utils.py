import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
from matplotlib import pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices
from statsmodels.graphics.factorplots import interaction_plot
from statsmodels.stats.weightstats import ttost_ind
import matplotlib.pylab as pylab
from collections import defaultdict
from sklearn.decomposition import PCA
from itertools import chain, combinations
from matplotlib import rcParams

sns.set_palette("binary")

BOXPROPS = {
    'boxprops':{'facecolor':'white', 'edgecolor':'black'},
    'medianprops':{'color':'black'},
    'whiskerprops':{'color':'black'},
    'capprops':{'color':'black'},
    'color':'white',
    "linewidth":0.8
}

BOXPROPS2 = {
    'boxprops':{'edgecolor':'black'},
    'medianprops':{'color':'black'},
    'whiskerprops':{'color':'black'},
    'capprops':{'color':'black'},
    #'color':'white',
    "linewidth":0.8
}

meanlineprops = dict(linestyle='--', linewidth=1, color='black')

arguments = [
    "Abortion is morally unacceptable, and it goes against the qualities and ethics that make this country great.",
    "More guns equals less crime. Just because crimes were committed with guns it does not mean control would work.",
    "The minimum wage increasing will allow more people to have more money, stimulating the economy and helping citizens who are currently in poverty reach out of it, take a foothold, and stay in the middle class.",
    "A just society’s goal should be to protect and further the wellbeing of its people (and, indeed, of all people, since being just requires a lack of bias toward or against other societies). Killing people as a form of punishment does not, as a rule, serve the interest of such a society."
]

topics = [
    "abortion",
    "gun control",
    "minimum wage",
    "death penalty",
]

argument_type = [
    "Republican",
    "Republican",
    "Democrat",
    "Democrat",
]


firsttask_columns = ['Q1_1', 'Q2_1', 'Q4_1', 'Q5_1']

#lower_bound_columns =  ['Q1_2_3', 'Q2_2_1', 'Q3_2_1',
#                        'Q4_2_1', 'Q5_2_1', 'Q6_2_1']

#upper_bound_columns = ['Q1_2_4',  'Q2_2_2','Q3_2_2',
#                       'Q4_2_2', 'Q5_2_2', 'Q6_2_2']

midpoint_columns = ['Q1_midpoint','Q2_midpoint','Q4_midpoint','Q5_midpoint']

def mean_stats(data):

    Dargs_toinclude = ['Q4_1','Q5_1']
    Rargs_toinclude = ['Q1_1', 'Q2_1']

    Dargs_beliefs_toinclude = ['Q4_midpoint','Q5_midpoint']
    Rargs_beliefs_toinclude = ['Q1_midpoint','Q2_midpoint']

    # participants' mean opinion for D arguments
    demargs_D_M = data[data.polAffil=="Democrat"][Dargs_toinclude].mean(axis=1).values
    print(demargs_D_M.mean())
    demargs_R_M = data[data.polAffil=="Republican"][Dargs_toinclude].mean(axis=1).values
    print(demargs_R_M.mean())
    print(stats.mannwhitneyu(demargs_D_M, demargs_R_M, alternative="greater"))

    # participants' mean belief for D arguments
    demargs_belief_D_M = data[data.polAffil=="Democrat"][Dargs_beliefs_toinclude].mean(axis=1).values
    print(demargs_belief_D_M.mean())
    demargs_belief_R_M = data[data.polAffil=="Republican"][Dargs_beliefs_toinclude].mean(axis=1).values
    print(demargs_belief_R_M.mean())
    print(stats.mannwhitneyu(demargs_belief_D_M, demargs_belief_R_M, alternative="greater"))


    # participants' mean opinion for R arguments
    repargs_D_M = data[data.polAffil=="Democrat"][Rargs_toinclude].mean(axis=1).values
    print(repargs_D_M.mean())
    repargs_R_M = data[data.polAffil=="Republican"][Rargs_toinclude].mean(axis=1).values
    print(repargs_R_M.mean())
    print(stats.mannwhitneyu(repargs_D_M, repargs_R_M, alternative="less"))
    
    # participants' mean belief for R arguments
    repargs_belief_D_M = data[data.polAffil=="Democrat"][Rargs_beliefs_toinclude].mean(axis=1).values
    print(repargs_belief_D_M.mean())
    repargs_belief_R_M = data[data.polAffil=="Republican"][Rargs_beliefs_toinclude].mean(axis=1).values
    print(repargs_belief_R_M.mean())
    print(stats.mannwhitneyu(repargs_belief_D_M, repargs_belief_R_M, alternative="less"))


    demargs_wilcox_D = stats.wilcoxon(demargs_D_M,demargs_belief_D_M)[1]
    demargs_wilcox_R = stats.wilcoxon(demargs_R_M,demargs_belief_R_M)[1]

    repargs_wilcox_D = stats.wilcoxon(repargs_D_M,repargs_belief_D_M)[1]
    repargs_wilcox_R = stats.wilcoxon(repargs_R_M,repargs_belief_R_M)[1]

    
    d1 = {"Statement pol": "Democrat",
        "Opinion D M (SD)": str(np.round(demargs_D_M.mean(),4))+" ("+str(np.round(demargs_D_M.std(),4))+")",
        "Opinion R M (SD)": str(np.round(demargs_R_M.mean(),4))+" ("+str(np.round(demargs_R_M.std(),4))+")",
        "Opinion D Mdn": str(np.round(np.median(demargs_D_M),3))+" "+str(np.quantile(demargs_D_M ,[0.25,0.75])),
        "Opinion R Mdn": str(np.round(np.median(demargs_R_M),3))+" "+str(np.quantile(demargs_R_M ,[0.25,0.75])),
        "Opinion Mann-Whitney": stats.mannwhitneyu(demargs_D_M, demargs_R_M, alternative="greater")[1],
        "Belief D M (SD)": str(np.round(demargs_belief_D_M.mean(),4))+" ("+str(np.round(demargs_belief_D_M.std(),4))+")",
        "Belief R M (SD)": str(np.round(demargs_belief_R_M.mean(),4))+" ("+str(np.round(demargs_belief_R_M.std(),4))+")",
        "Belief D Mdn": str(np.round(np.median(demargs_belief_D_M),3))+" "+str(np.quantile(demargs_belief_D_M ,[0.25,0.75])),
        "Belief R Mdn": str(np.round(np.median(demargs_belief_R_M),3))+" "+str(np.quantile(demargs_belief_R_M ,[0.25,0.75])),
        "Belief Mann-Whitney": stats.mannwhitneyu(demargs_belief_D_M, demargs_belief_R_M, alternative="greater")[1],
        "Wilcoxon D":  demargs_wilcox_D,
        "Wilcoxon R": demargs_wilcox_R
     }

    d2 = {"Statement pol": "Republican",
        "Opinion D M (SD)": str(np.round(repargs_D_M.mean(),4))+" ("+str(np.round(repargs_D_M.std(),4))+")",
        "Opinion R M (SD)": str(np.round(repargs_R_M.mean(),4))+" ("+str(np.round(repargs_R_M.std(),4))+")",
        "Opinion D Mdn": str(np.round(np.median(repargs_D_M),3))+" "+str(np.quantile(repargs_D_M,[0.25,0.75])),
        "Opinion R Mdn": str(np.round(np.median(repargs_R_M),3))+" "+str(np.quantile(repargs_R_M ,[0.25,0.75])),
        "Opinion Mann-Whitney": stats.mannwhitneyu(repargs_D_M, repargs_R_M, alternative="less")[1],
        "Belief D M (SD)": str(np.round(repargs_belief_D_M.mean(),4))+" ("+str(np.round(repargs_belief_D_M.std(),4))+")",
        "Belief R M (SD)": str(np.round(repargs_belief_R_M.mean(),4))+" ("+str(np.round(repargs_belief_R_M.std(),4))+")",
        "Belief D Mdn": str(np.round(np.median(repargs_belief_D_M),3))+" "+str(np.quantile(repargs_belief_D_M,[0.25,0.75])),
        "Belief R Mdn": str(np.round(np.median(repargs_belief_R_M),3))+" "+str(np.quantile(repargs_belief_R_M ,[0.25,0.75])),
        "Belief Mann-Whitney": stats.mannwhitneyu(repargs_belief_D_M, repargs_belief_R_M, alternative="less")[1],
        "Wilcoxon D":  repargs_wilcox_D,
        "Wilcoxon R": repargs_wilcox_R
     }

    
    return pd.DataFrame([d1,d2])


def per_statement_boxplot(data,firsttask_columns,midpoint_columns):
    temp = data.melt(id_vars=["polAffil"],value_vars=firsttask_columns+midpoint_columns)

    temp["Task"] = ["Belief" if "midpoint" in i else "Opinion" for i in temp.variable]
    temp["Statement"] = ["R"+s[1] if int(s[1])<4 else "D"+str(int(s[1])-3) for s in temp.variable]

    fig, axes = plt.subplots(ncols=4, sharex=True, sharey=True, figsize=(17,5))
    for ax, (n,grp) in zip(axes, temp.groupby("Statement")):
        sns.boxplot(x="Task", y="value", hue="polAffil", 
                    data=grp, 
                    ax=ax,
                    showfliers=False,
                    showmeans=True,
                    hue_order=["Democrat","Republican"],
                    meanline=True,
                    meanprops=meanlineprops,
                    **BOXPROPS2)
        sns.stripplot(x="Task", y="value", hue="polAffil",
                    dodge=True,
                    data=grp, 
                    ax=ax, 
                    color=".25",
                    size=1.2,
                    hue_order=["Democrat","Republican"])
        ax.set_title(n)
        #for i,box in enumerate(ax.artists):
        #    box.set_edgecolor('black')
        #    box.set_facecolor('white')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set(ylabel = "Opinion response and belief interval midpoint")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        handles, labels = ax.get_legend_handles_labels()
        
    axes[1].spines['left'].set_visible(False)
    axes[2].spines['left'].set_visible(False)
    axes[3].spines['left'].set_visible(False)
    #axes[4].spines['left'].set_visible(False)
    #axes[5].spines['left'].set_visible(False)
    axes[0].get_legend().remove()
    axes[1].get_legend().remove()
    axes[2].get_legend().remove()
    axes[3].get_legend().remove()
    #axes[4].get_legend().remove()
    #axes[5].get_legend().remove()
    axes[1].get_yaxis().set_visible(False)
    axes[2].get_yaxis().set_visible(False)
    axes[3].get_yaxis().set_visible(False)
    #axes[4].get_yaxis().set_visible(False)
    #axes[5].get_yaxis().set_visible(False)
    axes[0].set_xlabel("")
    axes[1].set_xlabel("")
    axes[2].set_xlabel("")
    axes[3].set_xlabel("")
    #axes[4].set_xlabel("")
    #axes[5].set_xlabel("")

    plt.legend(handles[0:2], labels[0:2],bbox_to_anchor=(-2.9, 1.2), loc="upper center", borderaxespad=0.,frameon=True,ncol=2,fontsize="large");
    #return fig


def collected_boxplot(data,firsttask_columns,midpoint_columns):
    plt.rc('axes', labelsize=16)
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    plt.rc('axes', titlesize=14)
    plt.rc('legend', fontsize=12) 
    plt.rc('figure', titlesize=18)
    demargs = pd.melt(data, id_vars=["ID","polAffil","educ","gender","age","polAlign_1"],value_vars=firsttask_columns[2:])
    repargs = pd.melt(data, id_vars=["ID","polAffil","educ","gender","age","polAlign_1"],value_vars=firsttask_columns[:2])

    demargs_midpoint = pd.melt(data, id_vars=["ID","polAffil","educ","gender","age","polAlign_1"],value_vars=midpoint_columns[2:])
    repargs_midpoint = pd.melt(data, id_vars=["ID","polAffil","educ","gender","age","polAlign_1"],value_vars=midpoint_columns[:2])

    fig, ax = plt.subplots(2,2, figsize=(10,10), sharey=True)

    sns.boxplot(x="polAffil", y="value", 
                data=demargs, 
                showfliers = False,
                ax=ax[0,0],
                showmeans=True,
                meanline=True,
                hue_order=["Democrat","Republican"],
                meanprops=meanlineprops,
                **BOXPROPS)
    sns.stripplot(x="polAffil", y="value", 
                data=demargs, 
                color=".25",
                size=1.8,
                hue_order=["Democrat","Republican"],
                ax=ax[0,0])
    ax[0,0].set_title("Opinions")
    ax[0,0].set(ylabel = "Response for Democrat\nstatements (D1 and D2)")

    sns.boxplot(x="polAffil", y="value", 
                data=repargs, 
                showfliers = False,
                ax=ax[1,0],
                showmeans=True,
                meanline=True,
                hue_order=["Democrat","Republican"],
                meanprops=meanlineprops,
                **BOXPROPS)
    sns.stripplot(x="polAffil", y="value", 
                data=repargs, 
                color=".25",
                size=1.8,
                hue_order=["Democrat","Republican"],
                ax=ax[1,0])
    ax[1,0].set_title("Opinions")
    ax[1,0].set(ylabel = "Response for Republican\nstatements (R1 and R2)")

    sns.boxplot(x="polAffil", y="value", 
                data=demargs_midpoint, 
                showfliers = False,
                ax=ax[0,1],
                showmeans=True,
                meanline=True,
                hue_order=["Democrat","Republican"],
                meanprops=meanlineprops,
                **BOXPROPS)
    sns.stripplot(x="polAffil", y="value", 
                data=demargs_midpoint, 
                color=".25",
                size=1.8,
                hue_order=["Democrat","Republican"],
                ax=ax[0,1])
    ax[0,1].set_title("Beliefs")
    
    sns.boxplot(x="polAffil", y="value", 
                data=repargs_midpoint, 
                showfliers = False,
                ax=ax[1,1],
                showmeans=True,
                meanline=True,
                hue_order=["Democrat","Republican"],
                meanprops=meanlineprops,
                **BOXPROPS)
    sns.stripplot(x="polAffil", y="value", 
                data=repargs_midpoint, 
                color=".25",
                size=1.8,
                hue_order=["Democrat","Republican"],
                ax=ax[1,1])
    ax[1,1].set_title("Beliefs")
    
    ax[0,1].set(ylabel = "")
    ax[0,1].spines['left'].set_visible(False)
    ax[0,1].get_yaxis().set_visible(False)
    ax[1,1].set(ylabel = "")
    ax[1,1].spines['left'].set_visible(False)
    ax[1,1].get_yaxis().set_visible(False)

    ax[0,0].set(xlabel = "")
    ax[0,1].set(xlabel = "")
    ax[1,0].set(xlabel = "")
    ax[1,1].set(xlabel = "")

    for ax_curr in ax.flatten():
        ax_curr.spines['right'].set_visible(False)
        ax_curr.spines['top'].set_visible(False)
    

    plt.plot([], [], '-', linewidth=0.8, color="black", label='mean')
    plt.plot([], [], '--', linewidth=0.8, color="black", label='median')
    plt.legend(bbox_to_anchor=(-1, 2.65), loc="upper center", borderaxespad=0.,frameon=True,ncol=2,fontsize="large")
    plt.subplots_adjust(hspace=0.4)



def LMM_results_and_plots(data,control_gender=False, control_age=False, control_educ=False):
    demargs = pd.melt(data, id_vars=["ID","polAffil","educ","gender","age","polAlign_1"],
                            value_vars=['Q4_1','Q5_1'])

    repargs = pd.melt(data, id_vars=["ID","polAffil","educ","gender","age","polAlign_1"],
                            value_vars=['Q1_1', 'Q2_1'])

    demargs_midpoint = pd.melt(data, id_vars=["ID","polAffil","educ","gender","age"],
                                value_vars=['Q4_midpoint','Q5_midpoint'])

    repargs_midpoint = pd.melt(data, id_vars=["ID","polAffil","educ","gender","age"],
                                value_vars=['Q1_midpoint', 'Q2_midpoint'])

    Dstatements = {}
    Rstatements = {}
    

    for d,stance in list(zip([Dstatements,Rstatements],[[demargs,demargs_midpoint],[repargs,repargs_midpoint]])):

        for i,j in list(zip(["Opinion","Belief"], stance)):
            model = smf.mixedlm("value ~ C(polAffil)",
                            j,groups=j["ID"])
            res = model.fit()
            
            print(i,res.summary())
            
            d["{} ~ Republican".format(i)] = res.pvalues["C(polAffil)[T.Republican]"]

            if control_gender:
                model = smf.mixedlm("value~ C(polAffil)+C(gender)",
                                j,groups=j["ID"])
                res = model.fit()
                #print(res.summary())
                d["{} ~ Republican + gender".format(i)] = res.pvalues["C(polAffil)[T.Republican]"]
                print("gender control",res.summary())
            
            if control_age:
                model = smf.mixedlm("value~ C(polAffil)+age",
                            j,groups=j["ID"])
                res = model.fit()
                #print(res.summary())
                d["{} ~ Republican + age".format(i)] = res.pvalues["C(polAffil)[T.Republican]"]
                print("control age",res.summary())

            if control_educ:
                model = smf.mixedlm("value~ C(polAffil)+C(educ)",
                            j,groups=j["ID"])
                res = model.fit()
                #print(res.summary())
                d["{} ~ Republican + educ".format(i)] = res.pvalues["C(polAffil)[T.Republican]"]
                print("gender educ",res.summary())
    
    LMM_results = []
    LMM_results.append(Dstatements)
    LMM_results.append(Rstatements)
    temp = pd.DataFrame(LMM_results)
    temp = temp.transpose(copy=True)
    temp.columns = ["D statements", "R statements"]
    return temp
    

    

def plot_polalign(data):
    temp = data.melt(id_vars=["polAffil","age"],value_vars=["polAlign_1"])

    fig, ax = plt.subplots(figsize=(10,10), sharey=True)

    sns.boxplot(x="polAffil", y="value", 
                data=temp, 
                showfliers = False,
                ax=ax,
                showmeans=True,
                meanline=True,
                hue_order=["Democrat","Republican"],
                meanprops=meanlineprops,
                **BOXPROPS)
    sns.stripplot(x="polAffil", y="value", 
                data=temp, 
                color=".25",
                size=1.8,
                hue_order=["Democrat","Republican"],
                ax=ax)



def interval_distributions(data):
    lower_bound_columns_R =  ['Q1_2_R_3', 'Q2_2_R_1',
                            'Q4_2_R_1', 'Q5_2_R_1']

    upper_bound_columns_R = ['Q1_2_R_4',  'Q2_2_R_2',
                        'Q4_2_R_2', 'Q5_2_R_2']

    lower_bound_columns_D =  ['Q1_2_D_3', 'Q2_2_D_1',
                            'Q4_2_D_1', 'Q5_2_D_1']

    upper_bound_columns_D = ['Q1_2_D_4',  'Q2_2_D_2',
                        'Q4_2_D_2', 'Q5_2_D_2']
    
    firsttask_columns = ['Q1_1', 'Q2_1', 'Q4_1', 'Q5_1']

    
    R = data[data.polAffil=="Republican"]
    D = data[data.polAffil=="Democrat"]


    fig, ax = plt.subplots(2,4,sharex=True, sharey=True, figsize=(20,10))


    for indel,s in enumerate([2,3,0,1]):

        if s==0:
            statement = "R1"
        elif s==1:
            statement = "R2"
        elif s==2:
            statement = "D1"
        else:
            statement = "D2"


        
        Rintervals_Rbeliefs = []
        Rintervals_Dbeliefs = []

        Dintervals_Rbeliefs = []
        Dintervals_Dbeliefs = []
        
        for i in range(len(R)):
            #print(i)
            interval_Republicans_R = np.arange(R.iloc[i][lower_bound_columns_R[s]],
                                            R.iloc[i][upper_bound_columns_R[s]]+0.01,
                                            0.01)
            Rintervals_Rbeliefs.extend(interval_Republicans_R)

            interval_Republicans_D = np.arange(R.iloc[i][lower_bound_columns_D[s]],
                                            R.iloc[i][upper_bound_columns_D[s]]+0.01,
                                            0.01)
            Rintervals_Dbeliefs.extend(interval_Republicans_D)

        for i in range(len(D)):
            #print(i)
            interval_Democrats_R = np.arange(D.iloc[i][lower_bound_columns_R[s]],
                                            D.iloc[i][upper_bound_columns_R[s]]+0.01,
                                            0.01)
            Dintervals_Rbeliefs.extend(interval_Democrats_R)

            interval_Democrats_D = np.arange(D.iloc[i][lower_bound_columns_D[s]],
                                            D.iloc[i][upper_bound_columns_D[s]]+0.01,
                                            0.01)
            Dintervals_Dbeliefs.extend(interval_Democrats_D)
        
        
        sns.kdeplot(Dintervals_Dbeliefs, shade=True, label="Democrats", color="b",ax=ax[0,indel])
        sns.kdeplot(Rintervals_Dbeliefs, shade=True, label="Republicans", color="r",ax=ax[0,indel])

        ax[0,indel].set_title(statement)
        ax[0,0].set_ylabel("Density\n\nBelief of Democrats' response")
        ax[1,0].set_ylabel("Density\n\nBelief of Republicans' response")
        
        
        sns.kdeplot(Dintervals_Rbeliefs, shade=True, label="Democrats", color="b", ax=ax[1,indel])
        sns.kdeplot(Rintervals_Rbeliefs, shade=True, label="Republicans", color="r", ax=ax[1,indel])
        
        # only one line may be specified; full height
        curr_Dmean = D[firsttask_columns[s]].mean()
        ax[0,indel].axvline(x=curr_Dmean, label='Actual D mean', linestyle="--",c="black", linewidth=0.8)
        
        ax[1,indel].plot([], [], '--', linewidth=0.8, color="black", label='Actual D mean')
        curr_Rmean = R[firsttask_columns[s]].mean()
        ax[1,indel].axvline(x=curr_Rmean, label='Actual R mean',linestyle="-",c="black",linewidth=0.8)
        
    
    plt.legend(bbox_to_anchor=(-3.2, 2.5), loc="upper center", 
    borderaxespad=0.,frameon=True,ncol=2, fontsize="x-large")
    




def new_per_statement_boxplot(data,firsttask_columns,midpoint_columns):
    temp = data.melt(id_vars=["polAffil"],value_vars=firsttask_columns+midpoint_columns)

    task_var = []
    for i in temp.variable:
        if "R_midpoint" in i:
            task_var.append("Belief of Republicans'\njudgement")
        elif "D_midpoint" in i:
            task_var.append("Belief of Democrats'\njudgement")
        else:
            task_var.append("Judgement")

    temp["Task"] = task_var
    
    temp["Statement"] = ["R"+s[1] if int(s[1])<4 else "D"+str(int(s[1])-3) for s in temp.variable]

    fig, axes = plt.subplots(ncols=4, sharex=True, sharey=True, figsize=(17,5))
    for ax, (n,grp) in zip(axes, temp.groupby("Statement")):
        sns.boxplot(x="Task", y="value", hue="polAffil", 
                    data=grp, 
                    ax=ax,
                    showfliers=False,
                    showmeans=True,
                    hue_order=["Democrat","Republican"],
                    meanline=True,
                    meanprops=meanlineprops,
                    **BOXPROPS2)
        sns.stripplot(x="Task", y="value", hue="polAffil",
                    dodge=True,
                    data=grp, 
                    ax=ax, 
                    color=".25",
                    size=1.2,
                    hue_order=["Democrat","Republican"])
        ax.set_title(n)
        #for i,box in enumerate(ax.artists):
        #    box.set_edgecolor('black')
        #    box.set_facecolor('white')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set(ylabel = "Judgement and belief interval midpoint")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        handles, labels = ax.get_legend_handles_labels()
        
    axes[1].spines['left'].set_visible(False)
    axes[2].spines['left'].set_visible(False)
    axes[3].spines['left'].set_visible(False)
    #axes[4].spines['left'].set_visible(False)
    #axes[5].spines['left'].set_visible(False)
    axes[0].get_legend().remove()
    axes[1].get_legend().remove()
    axes[2].get_legend().remove()
    axes[3].get_legend().remove()
    #axes[4].get_legend().remove()
    #axes[5].get_legend().remove()
    axes[1].get_yaxis().set_visible(False)
    axes[2].get_yaxis().set_visible(False)
    axes[3].get_yaxis().set_visible(False)
    #axes[4].get_yaxis().set_visible(False)
    #axes[5].get_yaxis().set_visible(False)
    axes[0].set_xlabel("")
    axes[1].set_xlabel("")
    axes[2].set_xlabel("")
    axes[3].set_xlabel("")
    #axes[4].set_xlabel("")
    #axes[5].set_xlabel("")

    plt.legend(handles[0:2], labels[0:2],bbox_to_anchor=(-2.9, 1.2), loc="upper center", borderaxespad=0.,frameon=True,ncol=2,fontsize="large");
    #return fig



def new_collected_boxplots(data,data1, firsttask_columns, midpoint_columns):
    
    plt.rc('axes', labelsize=14)
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    plt.rc('axes', titlesize=14)
    plt.rc('legend', fontsize=14) 
    plt.rc('figure', titlesize=16)
    
    temp = data.melt(id_vars=["polAffil"],value_vars=firsttask_columns+midpoint_columns)
    temp1 = data1.melt(id_vars=["polAffil"],value_vars=firsttask_columns+midpoint_columns)


    temp["Task"] = ["Belief" if "midpoint" in i else "Judgment" for i in temp.variable]
    temp1["Task"] = ["Belief" if "midpoint" in i else "Judgment" for i in temp1.variable]

    
    temp["Statement"] = ["R arguments" if int(s[1])<4 else "D arguments" for s in temp.variable]
    temp1["Statement"] = ["R arguments" if int(s[1])<4 else "D arguments" for s in temp1.variable]


    
    fig = plt.figure(figsize=(17,5))
    subfigs = fig.subfigures(1, 2,wspace=5)

    axesLeft = subfigs[0].subplots(1, 2, sharex=True,sharey=True)
    axesRight = subfigs[1].subplots(1, 2, sharex=True,sharey=True)

    subfigs[0].suptitle('Experiment 1',pad=70)
    subfigs[1].suptitle('Experiment 2',pad=70)

    for ax, (n,grp) in zip(axesLeft, temp.groupby("Statement")):
        sns.boxplot(x="Task", y="value", hue="polAffil", 
                    data=grp, 
                    ax=ax,
                    showfliers=False,
                    showmeans=True,
                    hue_order=["Democrat","Republican"],
                    meanline=True,
                    meanprops=meanlineprops,
                    **BOXPROPS2)
        sns.stripplot(x="Task", y="value", hue="polAffil",
                    dodge=True,
                    data=grp, 
                    ax=ax, 
                    color=".25",
                    size=1.2,
                    hue_order=["Democrat","Republican"])
        ax.set_title(n)
        
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set(ylabel = "Judgement and belief interval midpoint")
        ax.set_xticklabels(ax.get_xticklabels())
        handles, labels = ax.get_legend_handles_labels()
    
    for ax, (n,grp) in zip(axesRight, temp1.groupby("Statement")):
        sns.boxplot(x="Task", y="value", hue="polAffil", 
                    data=grp, 
                    ax=ax,
                    showfliers=False,
                    showmeans=True,
                    hue_order=["Democrat","Republican"],
                    meanline=True,
                    meanprops=meanlineprops,
                    **BOXPROPS2)
        sns.stripplot(x="Task", y="value", hue="polAffil",
                    dodge=True,
                    data=grp, 
                    ax=ax, 
                    color=".25",
                    size=1.2,
                    hue_order=["Democrat","Republican"])
        ax.set_title(n)
        
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set(ylabel = "Judgement and belief interval midpoint")
        ax.set_xticklabels(ax.get_xticklabels())
        handles, labels = ax.get_legend_handles_labels()

        
    axesLeft[1].spines['left'].set_visible(False)
    
    axesRight[1].spines['left'].set_visible(False)
    
    axesLeft[0].get_legend().remove()
    axesLeft[1].get_legend().remove()
    axesRight[0].get_legend().remove()
    axesRight[1].get_legend().remove()
    
    axesLeft[1].get_yaxis().set_visible(False)
    
    axesRight[1].get_yaxis().set_visible(False)
    
    axesLeft[0].set_xlabel("")
    axesLeft[1].set_xlabel("")
    axesRight[0].set_xlabel("")
    axesRight[1].set_xlabel("")
    
    return fig, handles, labels
    
   