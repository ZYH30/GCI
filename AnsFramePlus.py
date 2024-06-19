# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 10:51:46 2023

@author: yhzha
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score,roc_auc_score,average_precision_score

from itertools import combinations
from utilPlus import gcm_test,anm_test
import os
import seaborn as sns

import datetime
import pytz

def causalDirect(p_value_foward,p_value_backward,v1,v2,alpha = 0.05):
    result = False
    if p_value_foward >= p_value_backward:
        if p_value_foward >= alpha:
            if p_value_backward < alpha:
                print("Identifiable: \n \
                      {} is cause and P-value is {}, \n \
                      {} is effect and P-value is {}.".format(v1,round(p_value_foward,2),v2,round(p_value_backward,2)))
                result = True
            else:
                print("Unidentifiable: \n \
                      ALL direction's P-value large than {}. \n \
                      The direction mabey that {} cause {}.".format(alpha, v1, v2))
        else:
            print("Unidentifiable: \n \
                  ALL direction's P-value small than {}. \n \
                  The direction mabey that {} cause {}.".format(alpha, v1, v2))
    else:
        if p_value_backward >= alpha:
            if p_value_foward < alpha:
                print("Identifiable: \n \
                      {} is cause and P-value is {}, \n \
                      {} is effect and P-value is {}.".format(v2,round(p_value_backward,2),v1,round(p_value_foward,2)))
                result = True
            else:
                print("Unidentifiable: \n \
                      ALL direction's P-value large than {}. \n \
                      The direction mabey that {} cause {}.".format(alpha,v2,v1))
        else:
            print("Unidentifiable: \n \
                  ALL direction's P-value small than {}. \n \
                  The direction mabey that {} cause {}.".format(alpha, v2,v1))
    return result
        
def P_Identify(dataFrame,searchV,aimV,searchVType,corMat,corTheta = 0.05,CITheta = 0.05,directTheta = 0.1,is_str = True,exclSet = []):
    """
    Parameters
    ----------
    dataFrame : dataFrame
        DESCRIPTION.
    searchV : list
        DESCRIPTION.
    aimV : TYPE
        DESCRIPTION.
    aimVType : TYPE
        DESCRIPTION.
    corMat : DataFrame
        DESCRIPTION.
    corTheta : TYPE, optional
        DESCRIPTION. The default is 0.2.
    CITheta : TYPE, optional
        DESCRIPTION. The default is 0.05.
    directTheta : TYPE, optional
        DESCRIPTION. The default is 0.05.
    is_str : TYPE, Bool
        DESCRIPTION. Whether strict judgement. 
        False means judging only according to the size of the P value in different directions; 
        True means judging the parent-child relationship strictly according to the threshold;

    Returns
    -------
    None.

    """
    aimVType = searchVType[aimV]

    if len(searchV) == len(corMat):
        print("Start Identify!")
        print("Aim Variable:",aimV)
    else:
        print("corMat is not consistency with searchV")
        return
    
    # AdjIndex = corMat[aimV] > corTheta
    AdjIndex = (corMat[aimV] >= corTheta) | (corMat[aimV] <= -corTheta)

    Adj = set([searchV[i] for i in range(len(searchV)) if AdjIndex[i]])
    Adj = Adj - set([aimV])
    Adj = Adj - set(exclSet)
    print('exclSet:',exclSet)
    print('Init Adj:',Adj)
    
    AdjDict = {}
    for x_a in Adj:
        AdjDict[x_a] = corMat.loc[x_a,aimV]
        # print(x_a,AdjDict[x_a])
    
    sortedAdjDict = sorted(AdjDict.items(), key=lambda x: x[1])
    # print(sortedAdjDict)

    NoAdj = [] 
    # for x_ in Adj:
    for x_c in sortedAdjDict:
        
        x_ = x_c[0]
        print(x_)
        size = 1
        CAdj = Adj - set([x_])
        if len(CAdj) >= size:
            test = True
        else:
            test = False
            print("No much variable to CI test!")
        while test:
            for CAdj_ in combinations(CAdj,size):
                print(CAdj_)
                
                test_statistic,P_ = gcm_test(X = dataFrame[[aimV]].values.ravel(), 
                                             Y = dataFrame[[x_]].values.ravel(), 
                                             Z = dataFrame[list(CAdj_)], 
                                             X_type = aimVType, 
                                             Y_type = searchVType[x_],
                                             is_hypyopt = True)
                
                if P_ > CITheta:
                    NoAdj.append(x_)
                    Adj = CAdj
                    test = False
                    
                    break
            size = size + 1
            if size > 3:
                test = False
            if len(CAdj) < size:
                test = False
    print("Train Adj:",Adj)

    parSet = []
    chiSet = []
    for pc_ in Adj:
        ## ANM
        P_foward_, P_back_ = anm_test(X = dataFrame[[aimV]].values.ravel(), 
                                      Y = dataFrame[[pc_]].values.ravel(), 
                                      X_type = aimVType, 
                                      Y_type = searchVType[pc_],
                                      is_hypyopt = True)

        causalDirect(P_foward_, P_back_,aimV, pc_)
        if is_str:
            if (P_foward_ > directTheta) & (P_back_ <= directTheta):
                print('find')
                print(P_foward_,P_back_)
                chiSet.append(pc_)
            elif (P_foward_ <= directTheta) & (P_back_ > directTheta):
                print('find')
                print(P_foward_,P_back_)
                parSet.append(pc_)
            elif P_foward_ <= P_back_:
                parSet.append(pc_)
            elif abs(P_foward_ - P_back_) <= round(directTheta / 5, 2):
                parSet.append(pc_)
            else:
                chiSet.append(pc_)
        else:
            if P_back_ >= directTheta:
                parSet.append(pc_)
            elif P_foward_ >= directTheta:
                chiSet.append(pc_)
            else:
                parSet.append(pc_)

    print("parSet:",parSet)
    print("chiSet:",chiSet)
                
    return parSet,chiSet,NoAdj,Adj


def A_Identify(dataFrame,searchV,aimV,searchVType,corMat,corTheta = 0.05,CITheta = 0.05,directTheta = 0.1,is_str = True):
    search = True
    toIdeSet = [aimV]
    Ancestors = {}
    exclSet = []
    while search:
        toIdeSet_ = []
        exclSet_ = []
        for IdeV in toIdeSet:
            P_IdeV,C_IdeV,NA_IdeV,A_IdeV = P_Identify(dataFrame,searchV,IdeV,searchVType,corMat,corTheta,CITheta,directTheta,is_str,exclSet)
            toIdeSet_.extend(P_IdeV)
            exclSet_.extend(C_IdeV)
            Ancestors[IdeV] = P_IdeV
        if len(toIdeSet_) == 0:
            search = False
        else:
            exclSet.extend(toIdeSet)
            exclSet.extend(exclSet_)
            toIdeSet = [item for item in toIdeSet_ if item not in exclSet]
            toIdeSet = list(set(toIdeSet))
            
    
    return Ancestors


dataFile = './dataset/data-G-Case1.csv' # Options: data-G-Case1, data-G-Case2, data-G-Case3, data-G-Case4

dataFrame = pd.read_csv(dataFile)
dataFrame = dataFrame.drop(columns = 'y_delta')
dataFrame.columns

useCol = list(dataFrame.columns)

# 'discrete' 'continuous'
useColType = {}
for col_ in useCol:
    useColType[col_] = 'continuous'

corMat = dataFrame.corr(method='spearman')

plt.subplots(figsize = (20,10))
hosts1=sns.heatmap(corMat,annot=True,linewidths = .5,cmap ="YlGnBu")
plt.show()

Start_time = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
print("Start_time:", Start_time)
 

# Gene Data
Ancestors = A_Identify(dataFrame = dataFrame,searchV = useCol,aimV = 'y',
                       searchVType = useColType, corMat = corMat, 
                       corTheta = 0.2,CITheta = 0.05,directTheta = 0.05)

print('Ancestors:',Ancestors)

End_time = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
print("End_time:", End_time)
 
# date diff
time_difference = End_time - Start_time
print("time_difference:", time_difference)
