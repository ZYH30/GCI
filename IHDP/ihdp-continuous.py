# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 09:07:11 2024

@author: l
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,LabelEncoder,StandardScaler

import os 

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoidBased(x):
    # sigmoid [0, 1]
    return (1 / (1 + np.exp(-x)) - 0.5) * 2

def generate_new_variable(X, variable_type='continuous', seed = 0, is_y = [False,'t_name','yf','discrete']):
    X_copy = X.copy()
    np.random.seed(seed)  
    
    num_vars = np.random.randint(2, 4)
    
    chosen_vars = np.random.choice(X_copy.columns, num_vars, replace=False)
    
    g_mean = np.round(np.random.uniform(0.4, 1.6, size=1)[0],2)
    
    coefficients = np.round(np.random.normal(g_mean, 0.05, size=num_vars), 4)
   
    if is_y[0]:
        if is_y[2] == 'yf':
            new_variable_data = np.dot(X_copy[chosen_vars], coefficients)
            
            new_variable_data = np.sin(new_variable_data)
        else:
            if is_y[3] == 'discrete':
                X_copy[is_y[1]] = np.array([np.abs(val - 1) for val in X_copy[is_y[1]]])
            else:
                X_copy[is_y[1]] = X_copy[is_y[1]] - 0.1

           
            new_variable_data = np.dot(X_copy[chosen_vars], coefficients)
            new_variable_data = np.sin(new_variable_data)
        
        parent_variables = {'parent_variables': chosen_vars}
        parent_variables_coff = {'parent_coff': {var: coef for var, coef in zip(chosen_vars, coefficients)}}
        
        return new_variable_data,parent_variables,parent_variables_coff

        
    new_variable_data = np.dot(X_copy[chosen_vars], coefficients)
    
    if variable_type == 'continuous':
        new_variable_data = np.sin(new_variable_data)
        new_variable_data_F = new_variable_data + 0.2 * np.random.normal(0, 1, X.shape[0])
    elif variable_type == 'discrete':
        new_variable_data = sigmoid(new_variable_data)
        noise = np.random.normal(0, 1, X.shape[0])
        noise = sigmoid(noise)  
        new_variable_data_pro = 0.9 * new_variable_data + 0.1 * noise
        
        new_variable_data_F = np.array([np.random.binomial(1,i,1)[0] for i in new_variable_data_pro])
        
    
    parent_variables = {'parent_variables': chosen_vars}
    parent_variables_coff = {'parent_coff': {var: coef for var, coef in zip(chosen_vars, coefficients)}}
    
    return new_variable_data_F,parent_variables,parent_variables_coff


X_SourceFile = './ihdp.csv'
X_Source = pd.read_csv(X_SourceFile)

v_type_dict = {}
for i in range(X_Source.shape[1]):
    if i < 6:
        v_type_dict[X_Source.columns[i]] = 'continuous'
    else:
        v_type_dict[X_Source.columns[i]] = 'discrete'

scaler = StandardScaler()

continuousV = [key for key, value in v_type_dict.items() if value == 'continuous']
stasDict = {'min': X_Source[continuousV].min(),
            'max': X_Source[continuousV].max(),
            'mean': X_Source[continuousV].mean(),
            'std': X_Source[continuousV].std()}

X_Source.loc[:,continuousV] = scaler.fit_transform(X_Source.loc[:,continuousV])


# dict
parent_dict = {}
parent_coff_dict = {}
seed_dict = {}
seed_init = 10

# first level 
first_level_num = 15
first_level_D = pd.DataFrame()

for i in range(first_level_num):
    '''
    if i == 6:
        v_type = 'discrete' 
    else:
        v_type = 'continuous'
    '''
    v_type = 'continuous'
    result,parent_variables,parent_variables_coff = generate_new_variable(X_Source, v_type, seed_init)
    first_level_D[f'X_{i}'] = result
    parent_dict[f'X_{i}'] = list(parent_variables['parent_variables'])
    parent_coff_dict[f'X_{i}'] = parent_variables_coff['parent_coff']
    v_type_dict[f'X_{i}'] = v_type
    seed_dict[f'X_{i}'] = seed_init
    seed_init = round(seed_init * 1.3 + 6)

# second level 
second_level_num = 15
second_level_D = pd.DataFrame()

for i in range(first_level_num,first_level_num + second_level_num):  
    v_type = 'continuous'
    result,parent_variables,parent_variables_coff = generate_new_variable(first_level_D, v_type, seed_init)
    second_level_D[f'X_{i}'] = result
    parent_dict[f'X_{i}'] = list(parent_variables['parent_variables'])
    parent_coff_dict[f'X_{i}'] = parent_variables_coff['parent_coff']
    v_type_dict[f'X_{i}'] = v_type
    seed_dict[f'X_{i}'] = seed_init
    seed_init = round(seed_init * 1.8 + 3)

# third level 
third_level_num = 15
third_level_D = pd.DataFrame()

for i in range(first_level_num + second_level_num, first_level_num + second_level_num + third_level_num):  
    v_type = 'continuous'
    result,parent_variables,parent_variables_coff = generate_new_variable(second_level_D, v_type, seed_init)
    third_level_D[f'X_{i}'] = result
    parent_dict[f'X_{i}'] = list(parent_variables['parent_variables'])
    parent_coff_dict[f'X_{i}'] = parent_variables_coff['parent_coff']
    v_type_dict[f'X_{i}'] = v_type
    seed_dict[f'X_{i}'] = seed_init
    seed_init = round(seed_init * 1.4 + 8)

y = 'X_23'
t = 'X_6' 

second_level_D['y_f'],parent_variables,parent_variables_coff = generate_new_variable(first_level_D, v_type_dict[y], seed_dict[y], is_y = [True,t,'yf','continuous'])
second_level_D['y_cf'],parent_variables,parent_variables_coff = generate_new_variable(first_level_D, v_type_dict[y], seed_dict[y], is_y = [True,t,'ycf','continuous'])

data_ = pd.concat([X_Source,first_level_D,second_level_D,third_level_D],axis=1)
data_.rename(columns={t: 't'}, inplace=True)
data_.rename(columns={y: 'y'}, inplace=True)

is_save = True
if is_save:        
    data_.to_csv('./data-ihdp-continuous.csv',index = False)

