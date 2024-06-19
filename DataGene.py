# -*- coding: utf-8 -*-
"""
Created on Mon May 27 18:09:23 2024

@author: yhzha
"""


import os 

import numpy as np 
import pandas as pd
from numpy import mat 
import matplotlib.pyplot as plt
import seaborn as sns
import time
from DataUtil import GeneVa,GeneCo,sigmoid,GeneRe # GeneReP as GeneRe
from DataSetup import Case1Setup,Case2Setup,Case3Setup,Case4Setup

def toDataFrame(arrayX,VName,dim):           
    DataFrameX = pd.DataFrame(arrayX)
    if dim == 1:
        Xname = [VName]
        DataFrameX.columns = Xname
    else:
        Xname = [VName + '_{}'.format(i) for i in range(dim)]
        DataFrameX.columns = Xname
    return Xname, DataFrameX

dataLen = 1000
is_save = True
dataSetup = Case1Setup # Options: Case1Setup, Case2Setup, Case3Setup, Case4Setup
saveFile = 'Case1' # Options: Case1, Case2, Case3, Case4

V_dict = dataSetup['V_dict']
coff_dict = dataSetup['coff_dict']
resu_dict = dataSetup['resu_dict']
# gene variable
Va = {}

rootseed = 2023
leafcoffseed = 2033
leafrandseed = 2053
coff_value_dict = {}

for r_v in V_dict.keys():
    # print(r_v)
    if V_dict[r_v][0] == 'root':
        corSet = None
        varSet = None
        if V_dict[r_v][1] > 1:
            corSet = V_dict[r_v][2]
        else:
            varSet = V_dict[r_v][2]
        
        Va[r_v] = GeneVa(sizeSet = dataLen, dimSet = V_dict[r_v][1],seedSet = rootseed,corSet = corSet,varSet = varSet)
        rootseed += 1
    elif r_v == 't_delta':
        Va[r_v] = Va['t'] - 1
    elif r_v  == 'y_delta' and saveFile in ('Case1','Case3','Case4'):
        continue
    elif r_v  in ('y_delta','X6_delta') and saveFile == 'Case2':
        continue
    elif V_dict[r_v][0] in ('leaf','other'):
        if r_v == 'y' and saveFile in ('Case1','Case4','Case3'):
            gene_value = 0
            for c_ in coff_dict[r_v].keys():
                if coff_dict[r_v][c_][0] == 'F':
                    coff_value_dict[r_v + c_] = GeneCo(dimSet = V_dict[c_][1],seedSet = leafcoffseed,coffSet = coff_dict[r_v][c_][1],coffMin = None,coffMax = None)
                    leafcoffseed += 1
                if coff_dict[r_v][c_][0] == 'R':
                    coff_value_dict[r_v + c_] = GeneCo(dimSet = V_dict[c_][1],seedSet = coff_dict[r_v][c_][2],coffSet = None,coffMin = coff_dict[r_v][c_][1][0],coffMax = coff_dict[r_v][c_][1][1])
                if list(resu_dict[r_v].keys())[0] == 'Y':
                    gene_value += resu_dict[r_v]['Y'][c_] * GeneRe(Va[c_], coff_value_dict[r_v + c_], dimSet = V_dict[c_][1],lineSet = False)
                else:
                    gene_value += GeneRe(Va[c_], coff_value_dict[r_v + c_], dimSet = V_dict[c_][1],lineSet = False)
            
            np.random.seed(leafrandseed)
            v_random = np.random.normal(0, 1, (dataLen,1))
            if list(resu_dict[r_v].keys())[0] == 'Y':
                gene_value += resu_dict[r_v]['Y']['Rm'] * sigmoid(v_random)
                gene_value[gene_value>1] = 1
                gene_value[gene_value<0] = 0
                
                gene_value = np.array([np.random.binomial(1,i,1)[0] for i in gene_value])
                gene_value = gene_value[:, np.newaxis]
            else:
                gene_value += resu_dict[r_v]['N'] * v_random

            Va[r_v] = gene_value
            r_v = 'y_delta'
            gene_value = 0
            for c_ in coff_dict[r_v].keys():
                if coff_dict[r_v][c_][0] == 'F':
                    coff_value_dict[r_v + c_] = GeneCo(dimSet = V_dict[c_][1],seedSet = leafcoffseed,coffSet = coff_dict[r_v][c_][1],coffMin = None,coffMax = None)
                    leafcoffseed += 1
                if coff_dict[r_v][c_][0] == 'R':
                    coff_value_dict[r_v + c_] = GeneCo(dimSet = V_dict[c_][1],seedSet = coff_dict[r_v][c_][2],coffSet = None,coffMin = coff_dict[r_v][c_][1][0],coffMax = coff_dict[r_v][c_][1][1])
                if list(resu_dict[r_v].keys())[0] == 'Y':
                    gene_value += resu_dict[r_v]['Y'][c_] * GeneRe(Va[c_], coff_value_dict[r_v + c_], dimSet = V_dict[c_][1],lineSet = False)
                else:
                    gene_value += GeneRe(Va[c_], coff_value_dict[r_v + c_], dimSet = V_dict[c_][1],lineSet = False)
            
            if list(resu_dict[r_v].keys())[0] == 'Y':
                gene_value += resu_dict[r_v]['Y']['Rm'] * sigmoid(v_random)
                gene_value[gene_value>1] = 1
                gene_value[gene_value<0] = 0
                
                gene_value = np.array([np.random.binomial(1,i,1)[0] for i in gene_value])
                gene_value = gene_value[:, np.newaxis]
            else:
                gene_value += resu_dict[r_v]['N'] * v_random

            Va[r_v] = gene_value
            leafrandseed += 1
        elif r_v in ('X6','y') and saveFile == 'Case2':
            gene_value = 0
            for c_ in coff_dict[r_v].keys():
                if coff_dict[r_v][c_][0] == 'F':
                    coff_value_dict[r_v + c_] = GeneCo(dimSet = V_dict[c_][1],seedSet = leafcoffseed,coffSet = coff_dict[r_v][c_][1],coffMin = None,coffMax = None)
                    leafcoffseed += 1
                if coff_dict[r_v][c_][0] == 'R':
                    coff_value_dict[r_v + c_] = GeneCo(dimSet = V_dict[c_][1],seedSet = coff_dict[r_v][c_][2],coffSet = None,coffMin = coff_dict[r_v][c_][1][0],coffMax = coff_dict[r_v][c_][1][1])
                if list(resu_dict[r_v].keys())[0] == 'Y':
                    gene_value += resu_dict[r_v]['Y'][c_] * GeneRe(Va[c_], coff_value_dict[r_v + c_], dimSet = V_dict[c_][1],lineSet = False)
                else:
                    gene_value += GeneRe(Va[c_], coff_value_dict[r_v + c_], dimSet = V_dict[c_][1],lineSet = False)
            
            np.random.seed(leafrandseed)
            v_random = np.random.normal(0, 1, (dataLen,1))
            if list(resu_dict[r_v].keys())[0] == 'Y':
                gene_value += resu_dict[r_v]['Y']['Rm'] * sigmoid(v_random)
                gene_value[gene_value>1] = 1
                gene_value[gene_value<0] = 0
                gene_value = np.array([np.random.binomial(1,i,1)[0] for i in gene_value])
                gene_value = gene_value[:, np.newaxis]
            else:
                gene_value += resu_dict[r_v]['N'] * v_random

            Va[r_v] = gene_value
            
            if r_v == 'y':
                r_v = 'y_delta'
            else:
                r_v = 'X6_delta'
            
            # r_v = 'y_delta'
            gene_value = 0
            for c_ in coff_dict[r_v].keys():
                if coff_dict[r_v][c_][0] == 'F':
                    coff_value_dict[r_v + c_] = GeneCo(dimSet = V_dict[c_][1],seedSet = leafcoffseed,coffSet = coff_dict[r_v][c_][1],coffMin = None,coffMax = None)
                    leafcoffseed += 1
                if coff_dict[r_v][c_][0] == 'R':
                    coff_value_dict[r_v + c_] = GeneCo(dimSet = V_dict[c_][1],seedSet = coff_dict[r_v][c_][2],coffSet = None,coffMin = coff_dict[r_v][c_][1][0],coffMax = coff_dict[r_v][c_][1][1])
                    # leafcoffseed += 1
                if list(resu_dict[r_v].keys())[0] == 'Y':
                    gene_value += resu_dict[r_v]['Y'][c_] * GeneRe(Va[c_], coff_value_dict[r_v + c_], dimSet = V_dict[c_][1],lineSet = False)
                else:
                    gene_value += GeneRe(Va[c_], coff_value_dict[r_v + c_], dimSet = V_dict[c_][1],lineSet = False)
            
            if list(resu_dict[r_v].keys())[0] == 'Y':
                gene_value += resu_dict[r_v]['Y']['Rm'] * sigmoid(v_random)
                gene_value[gene_value>1] = 1
                gene_value[gene_value<0] = 0
                gene_value = np.array([np.random.binomial(1,i,1)[0] for i in gene_value])
                gene_value = gene_value[:, np.newaxis]
            else:
                gene_value += resu_dict[r_v]['N'] * v_random

            Va[r_v] = gene_value
            leafrandseed += 1
        else:
            gene_value = 0
            for c_ in coff_dict[r_v].keys():
                if coff_dict[r_v][c_][0] == 'F':
                    coff_value_dict[r_v + c_] = GeneCo(dimSet = V_dict[c_][1],seedSet = leafcoffseed,coffSet = coff_dict[r_v][c_][1],coffMin = None,coffMax = None)
                    leafcoffseed += 1
                if coff_dict[r_v][c_][0] == 'R':
                    coff_value_dict[r_v + c_] = GeneCo(dimSet = V_dict[c_][1],seedSet = coff_dict[r_v][c_][2],coffSet = None,coffMin = coff_dict[r_v][c_][1][0],coffMax = coff_dict[r_v][c_][1][1])
                if list(resu_dict[r_v].keys())[0] == 'Y':
                    gene_value += resu_dict[r_v]['Y'][c_] * GeneRe(Va[c_], coff_value_dict[r_v + c_], dimSet = V_dict[c_][1],lineSet = False)
                else:
                    gene_value += GeneRe(Va[c_], coff_value_dict[r_v + c_], dimSet = V_dict[c_][1],lineSet = False)
            
            np.random.seed(leafrandseed)
            v_random = np.random.normal(0, 1, (dataLen,1))
            if list(resu_dict[r_v].keys())[0] == 'Y':
                gene_value += resu_dict[r_v]['Y']['Rm'] * sigmoid(v_random)
                gene_value[gene_value>1] = 1
                gene_value[gene_value<0] = 0
                
                gene_value = np.array([np.random.binomial(1,i,1)[0] for i in gene_value])
                gene_value = gene_value[:, np.newaxis]
            else:
                gene_value += resu_dict[r_v]['N'] * v_random

            leafrandseed += 1
            Va[r_v] = gene_value

GeneDataFrame = pd.DataFrame()
XName = []
for r_v in V_dict.keys():
    if V_dict[r_v][0] == 'other':
        continue
    else:
        XName_, DataF_ = toDataFrame(Va[r_v],r_v,V_dict[r_v][1])
        GeneDataFrame = pd.concat([GeneDataFrame,DataF_],axis=1)
        XName += XName_

if is_save:
    GeneDataFrame.to_csv('./dataset/data-G-{}.csv'.format(saveFile),index = False)

corrspear = GeneDataFrame.corr(method='spearman')
plt.subplots(figsize = (20,10))
hosts1=sns.heatmap(corrspear,annot=True,linewidths = .5,cmap ="YlGnBu")
# plt.savefig("Fig.png",dpi=500,bbox_inches = 'tight')
plt.show()