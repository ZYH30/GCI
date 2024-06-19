# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 20:27:04 2023

@author: yhzha
"""

import os 
import numpy as np 
import pandas as pd
from numpy import mat 
import matplotlib.pyplot as plt
import seaborn as sns
import time


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoidBased(x):
    return ((1 / (1 + np.exp(-x)) - 0.5) * 2) ** 2

def GeneVa(sizeSet,dimSet,seedSet,corSet = None,varSet = None):
    """
    Parameters
    ----------
    sizeSet : TYPE
        sample size.
    dimSet : TYPE
        variable dim.
    seedSet : TYPE
        random seed.
    corSet : TYPE, optional
        DESCRIPTION. The default is None.
    """
    if dimSet == 1:
        np.random.seed(seedSet)
        v = np.random.normal(0, varSet, (sizeSet,1))
    else:
        np.random.seed(seedSet)
        v_mean = np.random.normal(0, 1, dimSet)
        # v_cor = np.random.uniform(low = -1, high=1.0, size=(dimSet,dimSet))
        # v_cor = (v_cor + v_cor.T) * 0.5
        v_cor = np.ones((dimSet,dimSet)) 
        # v_cor = np.random.uniform(low = 1, high = 2, size = dimSet)
        for i in range(dimSet):
            for j in range(dimSet):
                if corSet == 'R':
                    if i > j:
                        seed = (seedSet + i) * (j + 1)
                        
                        np.random.seed(seed)
                        v_cor[i,j] = np.random.uniform(low = -1, high=1.0, size=(1,))[0]
                        v_cor[j,i] = v_cor[i,j]
                else:
                    if i != j:
                        v_cor[i,j] = corSet
        # print(v_cor)
        np.random.seed(seedSet)
        v = np.random.multivariate_normal(mean=v_mean, cov=v_cor, size=sizeSet)
    
    return v



def GeneCo(dimSet,seedSet,coffSet = None,coffMin = None,coffMax = None):
    """
    Parameters
    ----------
    dimSet : TYPE
        variable dim.
    seedSet : int
        random seed.
    dimUse : TYPE, int
        DESCRIPTION. The default is 1.
    coffSet : TYPE, float or array
        float: 1.0; array: np.array([[3.68467327],[1.52134877]],ndmin = 2).
    coffMin : TYPE, optional
        DESCRIPTION. The default is None.
    coffMax : TYPE, optional
        DESCRIPTION. The default is None.
    """
    if dimSet == 1:
        if coffSet is not None:
            v_coff = coffSet
        else:
            np.random.seed(seedSet)
            v_coff = np.random.uniform(low = coffMin, high = coffMax, size=(1,))
    else:
        if coffSet is not None:
            v_coff = coffSet
        else:
            np.random.seed(seedSet)
            v_coff = np.random.uniform(low = coffMin, high = coffMax, size=(dimSet, 1))
            # print(v_coff)
    return v_coff

'''
def GeneRe(v, v_coff, dimSet,lineSet = False):
    """
    Parameters
    ----------
    v : TYPE
        DESCRIPTION.
    v_coff : TYPE
        DESCRIPTION.
    dimSet : TYPE
        DESCRIPTION.
    dimUse : TYPE, optional
        DESCRIPTION. The default is 1.
    lineSet : TYPE, optional
        DESCRIPTION. The default is False.
    """
    
    if dimSet == 1:
        if lineSet:
            v_o = v * v_coff
        else:
            v_o = v_coff * sigmoid(v)
    else:
        if lineSet:
            v_o = mat(v[:,0:dimSet]) * mat(v_coff)
        else:
            v_o = mat(sigmoid(v)[:,0:dimSet]) * mat(v_coff)
        
    return v_o
'''
def GeneRe(v, v_coff, dimSet,lineSet = False):
    """
    Parameters
    ----------
    v : TYPE
        DESCRIPTION.
    v_coff : TYPE
        DESCRIPTION.
    dimSet : TYPE
        DESCRIPTION.
    dimUse : TYPE, optional
        DESCRIPTION. The default is 1.
    lineSet : TYPE, optional
        DESCRIPTION. The default is False.
    """
    if dimSet == 1:
        if lineSet:
            v_o = v * v_coff
        else:
            v_o = np.sin(v * v_coff) 
    else:
        if lineSet:
            v_o = mat(v[:,0:dimSet]) * mat(v_coff)
        else:
            v_o = np.sin(mat(v[:,0:dimSet]) * mat(v_coff))
        
    return v_o

def GeneReP(v, v_coff, dimSet,lineSet = False):
    """
    Parameters
    ----------
    v : TYPE
        DESCRIPTION.
    v_coff : TYPE
        DESCRIPTION.
    dimSet : TYPE
        DESCRIPTION.
    dimUse : TYPE, optional
        DESCRIPTION. The default is 1.
    lineSet : TYPE, optional
        DESCRIPTION. The default is False.
    """
    if dimSet == 1:
        if lineSet:
            v_o = v * v_coff
        else:
            v_o = v_coff * sigmoidBased(v)
    else:
        if lineSet:
            v_o = mat(v[:,0:dimSet]) * mat(v_coff)
        else:
            v_o = mat(sigmoidBased(v)[:,0:dimSet]) * mat(v_coff)
        
    return v_o

def toDataFrame(arrayX,VName,dim):           
    DataFrameX = pd.DataFrame(arrayX)
    if dim == 1:
        Xname = [VName]
        DataFrameX.columns = Xname
    else:
        Xname = [VName + '_{}'.format(i) for i in range(dim)]
        DataFrameX.columns = Xname
    return Xname, DataFrameX