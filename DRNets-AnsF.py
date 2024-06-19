# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 18:58:34 2024

@author: yhzha
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.layers import Lambda, Input, Dense,BatchNormalization,Concatenate,Add,Subtract
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.losses import mse, binary_crossentropy,CosineSimilarity,cosine_similarity,sparse_categorical_crossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers
from tensorflow.keras import regularizers

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import random
import tensorflow as tf
from numpy import mat 

import warnings
warnings.filterwarnings('ignore') 


def Drnet(t_dim = 1,c_dim = 25,head_num = 5):
    
    intermediate_x_dim = c_dim + 2
    intermediate_dim = round(c_dim/4 + 2)
    
    inputs_t = Input(shape=(t_dim,), name='input_t_balance')
    inputs_c = Input(shape=(c_dim,), name='input_c_balance')
    inputs_t_head = Input(shape=(t_dim,), name='inputs_t_head')
    

    x_rep_o_ = Dense(intermediate_x_dim, activation='elu')(inputs_c)
    x_rep_o_ = Dense(round(intermediate_x_dim/2), activation='elu')(x_rep_o_)
    x_rep_o = Dense(round(intermediate_x_dim/2), activation='elu',name = 'x_rep_o'
                  )(x_rep_o_)

    
    x_y = Concatenate()([x_rep_o, inputs_t])
    
    # split head
    head_ind = []
    head_x_predtic_y = []
    for i in range(head_num):
        ind_ = tf.cast(tf.where(inputs_t_head == i)[:,0], tf.int32)
        x_y_ = tf.gather(x_y, ind_)
        x_predtic_y_ = Dense(intermediate_dim, activation='elu',kernel_regularizer = 'l1')(x_y_)
        x_predtic_y_ = Dense(intermediate_dim, activation='elu',kernel_regularizer = 'l1')(x_predtic_y_)
        x_predtic_y = Dense(1)(x_predtic_y_)
        
        head_ind.append(ind_)
        head_x_predtic_y.append(x_predtic_y)
    
    y = tf.dynamic_stitch(head_ind, head_x_predtic_y)
    
    DrnetM = Model([inputs_t, inputs_c,inputs_t_head], 
                    y, name='Drnet')
    

    optimizer = Adam(learning_rate=0.005)
    # DrnetM.compile(optimizer = 'adam',loss = 'mse')
    DrnetM.compile(optimizer = optimizer,loss = 'mse')
    
    return DrnetM

def splitData(data, x_name, x_dim = 5):
    dataLinearGaussian = data
    
    t_column = 't'
    c_column = x_name
    
    y_column = 'y'
    
    y_delta_column = 'y_delta'
    
    t_head_column = 't_head'
    
    
    random.seed(20180808)
    train_val_rate = 0.8
    train_val_samples = round(len(dataLinearGaussian) * train_val_rate)

    train_val_sample_select = random.sample(range(len(dataLinearGaussian)), train_val_samples)
    test_sample_select = list(set(range(len(dataLinearGaussian))) - set(train_val_sample_select))
    
    print("----split dataSet----")
    input_t = dataLinearGaussian.loc[train_val_sample_select,t_column].reset_index(drop=True)
    input_c = dataLinearGaussian.loc[train_val_sample_select,c_column].reset_index(drop=True)
    input_y = dataLinearGaussian.loc[train_val_sample_select,y_column].reset_index(drop=True)
    input_t_head = dataLinearGaussian.loc[train_val_sample_select,t_head_column].reset_index(drop=True)

    y_true = dataLinearGaussian.loc[train_val_sample_select,y_column].reset_index(drop=True)
    y_delta = dataLinearGaussian.loc[train_val_sample_select,y_delta_column].reset_index(drop=True)

    input_t_test = dataLinearGaussian.loc[test_sample_select,t_column].reset_index(drop=True)
    input_c_test = dataLinearGaussian.loc[test_sample_select,c_column].reset_index(drop=True)
    input_y_test = dataLinearGaussian.loc[test_sample_select,y_column].reset_index(drop=True)
    input_t_head_test = dataLinearGaussian.loc[test_sample_select,t_head_column].reset_index(drop=True)

    y_true_test = dataLinearGaussian.loc[test_sample_select,y_column].reset_index(drop=True)
    y_delta_test = dataLinearGaussian.loc[test_sample_select,y_delta_column].reset_index(drop=True)
    
    return input_t,input_c,input_y,y_true,y_delta,input_t_test,input_c_test,input_y_test,y_true_test,y_delta_test ,input_t_head,input_t_head_test



'''
############################
# Case1 
dataFile = './dataset/data-G-Case1.csv'
# Total
# x_dim = 8
# x_name = ['X{}'.format(i) for i in range(1,x_dim + 1)]

# AnsF
x_name = ['X4','X5']
x_dim = 2
############################
'''

'''
############################
# Case2
dataFile = './dataset/data-G-Case2.csv'
# Total
# x_dim = 9
# x_name = ['X{}'.format(i) for i in range(1,x_dim + 1)]

# AnsF
x_name = ['X6','X7','X8']
x_dim = 3
############################
'''

'''
############################
# Case3 
dataFile = './dataset/data-G-Case3.csv'
# Total
# x_dim = 7
# x_name = ['X{}'.format(i) for i in range(1,x_dim + 1)]

# AnsF
x_name = ['X4','X5']
x_dim = 2
############################
'''


############################
# Case4 
dataFile = './dataset/data-G-Case4.csv'
# Total
# x_dim = 7
# x_name = ['X{}'.format(i) for i in range(1,x_dim + 1)]

# AnsF
x_name = ['X3','X4','X5']
x_dim = 3

############################


epochs = 400
batch_size = 800
head_num = 2

moniData = pd.read_csv(dataFile)
moniData.columns

# resultSavePath = x_type[1] + '-' + y_type[1] + '-gan' + '.csv'
moniData['t_head'] = 0

quant_t = np.quantile(moniData['t'],[0,0.5,1])
for i in range(head_num):
    moniData.loc[(moniData['t'] >= quant_t[i]) & (moniData['t'] < quant_t[i+1]),'t_head'] = i


input_t,input_c,input_y,y_true,y_delta,input_t_test,input_c_test,input_y_test,y_true_test,y_delta_test,input_t_head,input_t_head_test = splitData(moniData, x_name)

train_RMSE_MTEF = []
test_RMSE_MTEF = []
for i in range(100):
    DrnetM = Drnet(c_dim = x_dim)
    
    DrnetM_history = DrnetM.fit([input_t, input_c,input_t_head],input_y,epochs=epochs,batch_size=batch_size,validation_split = 0.2,verbose = 0)
    
    plt.plot(DrnetM_history.history['loss'])
    plt.plot(DrnetM_history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train_loss', 'val_loss'], loc='upper left')
    plt.show()
    
    DrnetM_predict = DrnetM.predict([input_t, input_c,input_t_head])
    
    plt.plot(DrnetM_predict[:50])
    plt.plot(input_y[:50])
    plt.legend(['Predict','True'], loc='upper left')
    plt.show()
    
    from sklearn.metrics import mean_squared_error
    ## M
    true_MTEF = y_true - y_delta
    
    DrnetM_predict_delta = DrnetM.predict([input_t - 1, input_c,input_t_head])
    predict_MTEF = DrnetM_predict - DrnetM_predict_delta
    print(f"RMSE of MTEF:{np.sqrt(mean_squared_error(predict_MTEF, true_MTEF))}")
    train_RMSE_MTEF.append(round(np.sqrt(mean_squared_error(predict_MTEF, true_MTEF)),4))
    
    DrnetM_predict_test = DrnetM.predict([input_t_test, input_c_test, input_t_head_test])
    plt.plot(DrnetM_predict_test[:50])
    plt.plot(input_y_test[:50])
    plt.legend(['encoder_test','latten'], loc='upper left')
    plt.show()
    
    true_MTEF_test = y_true_test - y_delta_test
    
    DrnetM_predict_test_delta_ = DrnetM.predict([input_t_test - 1, input_c_test,input_t_head_test])
    predict_MTEF_test = DrnetM_predict_test - DrnetM_predict_test_delta_
    print(f"RMSE of MTEF:{np.sqrt(mean_squared_error(predict_MTEF_test, true_MTEF_test))}")
    test_RMSE_MTEF.append(round(np.sqrt(mean_squared_error(predict_MTEF_test, true_MTEF_test)),4))

train_RMSE_MTEF.remove(max(train_RMSE_MTEF))
train_RMSE_MTEF.remove(min(train_RMSE_MTEF))

test_RMSE_MTEF.remove(max(test_RMSE_MTEF))
test_RMSE_MTEF.remove(min(test_RMSE_MTEF))

print('Mean Value of RMSE of MTEF on Train Set:',np.mean(train_RMSE_MTEF))
print('Mean Value of RMSE of MTEF on Test Set:',np.mean(test_RMSE_MTEF))

print('Std Value RMSE of MTEF on Train Set:',np.std(train_RMSE_MTEF))
print('Std Value RMSE of MTEF on Test Set:',np.std(test_RMSE_MTEF))