# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 09:35:05 2023

@author: yhzha
"""


# import warnings
# warnings.filterwarnings("ignore")

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from hyperopt import fmin, tpe, hp, Trials
from scipy.stats import pearsonr,spearmanr,ttest_ind,f_oneway
from scipy.stats import norm
from joblib import Parallel, delayed

import optuna
import logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

import os
# os.chdir(r'D:\phD\Code\GeneralFrame')
 

def lightgbm_optuna(X, y, y_type,is_plot = False,is_parallel = False,is_optM = True):
    
    nn = X.shape[0]
    n_feature = X.shape[1]
    # rint(nn,n_feature)
    if y_type == 'discrete':
        y_encoded = LabelEncoder().fit_transform(y)
        objective = 'multiclass' if len(set(y)) > 2 else 'binary'
        eval_metric = 'multi_logloss' if objective == 'multiclass' else 'binary_logloss'
        num_class = len(set(y)) if objective == 'multiclass' else 1
        
    elif y_type == 'continuous':
        y_encoded = y
        objective = 'regression'
        eval_metric = 'rmse'
        num_class = 1
    else:
        raise ValueError("Invalid y_type. Supported values are 'discrete' and 'continuous'.")

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=2024) # , random_state=42

    def objectiveFun(trial):
        if is_optM:
            '''
            params = {'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2),
                      'max_depth': trial.suggest_int('max_depth', 5, 10),
                      'num_leaves': trial.suggest_int('num_leaves', 40, 100),
                      # 'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', round(nn/100), round(nn/10)),
                      # 'feature_fraction': trial.suggest_uniform('feature_fraction', 0.1, 1.0),
                      # 'min_data_in_bin':trial.suggest_int('min_data_in_bin', round(nn/200), round(nn/20)),
                      # 'subsample': trial.suggest_float('subsample', 0.8, 1.0)
                      }
            '''
            params = {'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.4),
                  'max_depth': trial.suggest_int('max_depth', 3, 15),
                  'num_leaves': trial.suggest_int('num_leaves', 40, 200),
                  'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 100),
                  # 'feature_fraction': trial.suggest_uniform('feature_fraction', 0.1, 1.0),
                  'min_data_in_bin':trial.suggest_int('min_data_in_bin', 5, 100),
                  'subsample': trial.suggest_float('subsample', 0.7, 0.9)
                  }
            
        
        else:
            params = {'learning_rate': trial.suggest_float('learning_rate', 0.1, 0.4),
                      'max_depth': trial.suggest_int('max_depth', 2, 5),
                      'num_leaves': trial.suggest_int('num_leaves', 5, 15),
                      'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', round(nn/30), round(nn/10)),
                      # 'feature_fraction': trial.suggest_uniform('feature_fraction', 0.1, 1.0),
                      'min_data_in_bin':trial.suggest_int('min_data_in_bin', round(nn/10), round(nn/5)),
                      'subsample': trial.suggest_float('subsample', 0.6, 0.8)}
            
        paramsTotal = {
            'objective': objective,
            'metric': eval_metric,
            'num_class': num_class,
            'verbosity': -1,
            # 'max_bin_by_feature':max_bin_by_feature,
            #'min_data_in_bin':min_data_in_bin,
            # 'min_data_in_leaf':min_data_in_leaf,
            **params
        }
        
        
        ## Split Data
        # print(paramsTotal)
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test)
        # model = lgb.train(paramsTotal, train_data, num_boost_round=256)
        model = lgb.train(paramsTotal, train_data, valid_sets = [test_data],
                      num_boost_round=1024,
                      callbacks = [lgb.early_stopping(stopping_rounds=50,verbose = False)] # lgb.log_evaluation(period=20), 
                      )
                      
        y_pred = model.predict(X_test)
        loss = mean_squared_error(y_test, y_pred) if objective == 'regression' else \
            roc_auc_score(y_test, y_pred) if objective == 'binary' else \
            accuracy_score(y_test, y_pred.argmax(axis=1))

        return loss

    if is_parallel:
        def optimize(n_trials):
            study = optuna.load_study(study_name='joblib-quadratic', storage='sqlite:///example.db')
            study.optimize(objectiveFun, n_trials=n_trials)
    
        current_dir = os.getcwd()
        file_name = "example.db"
        full_path = os.path.join(current_dir, file_name)
         
        if os.path.exists(full_path):
            os.remove(full_path)  
        
        if y_type == 'discrete':
            study = optuna.create_study(study_name='joblib-quadratic', storage='sqlite:///example.db',direction='maximize')
        else:
            study = optuna.create_study(study_name='joblib-quadratic', storage='sqlite:///example.db')
        
        r = Parallel(n_jobs=8)([delayed(optimize)(128) for _ in range(128)])
        
        # print('Best trial:')
        trial = study.best_trial
        # print('  Value: ', trial.value)
        best_params = {key: value for key, value in trial.params.items()}
    else:
        if y_type == 'discrete':
            study = optuna.create_study(direction='maximize')
        else:
            study = optuna.create_study(direction='minimize')
        
        study.optimize(objectiveFun, n_trials=128)
        best_params = study.best_params


    total_data = lgb.Dataset(X, label=y_encoded)
    # print('best_params:',best_params)
    
    model = lgb.train({**best_params, 'objective': objective, 
                       'metric': eval_metric, 'num_class': num_class,
                       # 'min_data_in_bin': min_data_in_bin,
                       'verbosity': -1}, total_data, num_boost_round = 512)
    
    total_pred = model.predict(X)

    residual = y_encoded - total_pred
    loss = mean_squared_error(y_encoded, total_pred) if objective == 'regression' else \
        roc_auc_score(y_encoded, total_pred) if objective == 'binary' else \
        accuracy_score(y_encoded, total_pred.argmax(axis=1))
    
    if is_plot:
        plt.plot(y_encoded[0:50])
        plt.plot(total_pred[0:50])
        plt.show()

    return model, best_params, residual, loss

def lightgbm_hyperMau(X, y, y_type,is_plot = False):
    '''
    Parameters
    ----------
    X : TYPE DataFrame
        DESCRIPTION.
    y : TYPE numpy
        DESCRIPTION.
    y_type : TYPE
        DESCRIPTION.
    is_plot : TYPE, optional
        DESCRIPTION. The default is False.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    model : TYPE
        DESCRIPTION.
    params : TYPE
        DESCRIPTION.
    residual : TYPE
        DESCRIPTION.
    loss : TYPE
        DESCRIPTION.

    '''
    nn = X.shape[0]
    n_feature = X.shape[1]
    # print(nn,n_feature)
    if y_type == 'discrete':
        y_encoded = LabelEncoder().fit_transform(y)
        objective = 'multiclass' if len(set(y)) > 2 else 'binary'
        params = {
            'objective' : objective,
            'metric': 'multi_logloss' if objective == 'multiclass' else 'binary_logloss',
            'num_class': len(set(y)) if objective == 'multiclass' else 1,
            'max_bin': 100,
            'min_data_in_bin': 10,
            'max_depth': 5,
            'num_leaves': 30,
            'learning_rate':0.1,
            'min_data_in_leaf': 10,
            'subsample' : 0.9,
            
            'verbosity':-1
            }

    elif y_type == 'continuous':
        y_encoded = y
        objective = 'regression'
        params = {
            'objective' : objective,
            'metric': 'rmse',
            'num_class': 1,
            'max_bin': 100,
            'min_data_in_bin': 10,
            'max_depth': 5,
            'num_leaves': 30,
            'learning_rate':0.1,
            'min_data_in_leaf': 10,
            'subsample' : 0.9,
            
            'verbosity':-1
            }
    else:
        raise ValueError("Invalid y_type. Supported values are 'discrete' and 'continuous'.")

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=22)
    
    # train_data = lgb.Dataset(X_train, label=y_train)
    # test_data = lgb.Dataset(X_test, label=y_test)
    total_data = lgb.Dataset(X, label=y_encoded)
    
    model = lgb.train(params, total_data, valid_sets = [total_data],
                      num_boost_round=200,
                      callbacks = [lgb.log_evaluation(period=10)]
                      # callbacks = [lgb.log_evaluation(period=10), lgb.early_stopping(stopping_rounds=100)]
                      )
    

    # y_pred = model.predict(X_test)
    total_pred = model.predict(X)

    residual = y_encoded - total_pred
    loss = mean_squared_error(y_encoded, total_pred) if objective == 'regression' else \
        accuracy_score(y_encoded, total_pred.round()) if objective == 'binary' else \
        accuracy_score(y_encoded, total_pred.argmax(axis=1))
    
    if is_plot:
        plt.plot(y_encoded[0:50])
        plt.plot(total_pred[0:50])
        plt.show()

    return model, params, residual, loss

def gcm_test(X=None, Y=None, Z=None, alpha=0.05, regr_method="lightGBM", X_type='continuous', Y_type='continuous',
             plot_residuals=False, resid_XonZ=None, resid_YonZ=None,is_hypyopt = True,is_parallel = False):
    '''
    Parameters
    ----------
    X : TYPE, numpy
        DESCRIPTION. The default is None.
    Y : TYPE, numpy
        DESCRIPTION. The default is None.
    Z : TYPE, DataFrame
        DESCRIPTION. The default is None.
    alpha : TYPE, optional
        DESCRIPTION. The default is 0.05.
    regr_method : TYPE, optional
        DESCRIPTION. The default is "lightGBM".
    X_type : TYPE, optional
        DESCRIPTION. The default is 'continuous'.
    Y_type : TYPE, optional
        DESCRIPTION. The default is 'continuous'.
    plot_residuals : TYPE, optional
        DESCRIPTION. The default is False.
    resid_XonZ : TYPE, optional
        DESCRIPTION. The default is None.
    resid_YonZ : TYPE, optional
        DESCRIPTION. The default is None.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    dict
        DESCRIPTION.

    '''
  
    if Z is None:
      if resid_XonZ is None:
        resid_XonZ = X
      if resid_YonZ is None:
        resid_YonZ = Y
    else:
      if resid_XonZ is None:
        if X is None:
          raise ValueError("Either X or resid_XonZ must be provided.")
        # Could add Ztype astype to 'category'
        if is_hypyopt:
            _,_,resid_XonZ,_ = lightgbm_optuna(Z, X, X_type,is_plot = False,is_parallel = is_parallel)
        else:
            _,_,resid_XonZ,_ = lightgbm_hyperMau(Z, X, X_type,is_plot = False)
      if resid_YonZ is None:
        if Y is None:
          raise ValueError("Either Y or resid_YonZ must be provided.")
        # Could add Ztype astype to 'category'
        if is_hypyopt:
            _,_,resid_YonZ,_ = lightgbm_optuna(Z, Y, Y_type,is_plot = False,is_parallel = is_parallel)
        else:
            _,_,resid_YonZ,_ = lightgbm_hyperMau(Z, Y, Y_type,is_plot = False)
  
  
    nn = resid_XonZ.shape[0]
    R = resid_XonZ * resid_YonZ
    R_sq = R**2
    meanR = np.mean(R)
    test_statistic = np.sqrt(nn) * meanR / np.sqrt(np.mean(R_sq) - meanR**2)
    p_value = 2 * norm.sf(np.abs(test_statistic))
    
    if plot_residuals:
      import matplotlib.pyplot as plt
      plt.scatter(resid_XonZ, resid_YonZ)
      plt.title("Scatter plot of residuals")
      plt.show()
  
    return test_statistic,p_value

def perform_anova(data):
    f_statistic, p_value = f_oneway(*data.values())
    return f_statistic, p_value
    
def cov_test(x,y):
    nn = x.shape[0]
    R = x * y
    R_sq = R**2
    meanR = np.mean(R)
    test_statistic = np.sqrt(nn) * meanR / np.sqrt(np.mean(R_sq) - meanR**2)
    p_value = 2 * norm.sf(np.abs(test_statistic))
    
    return test_statistic,p_value

def anm_test(X=None, Y=None, X_type='continuous', Y_type='continuous',is_hypyopt = True):
    '''
    Parameters
    ----------
    X : TYPE, numpy
        DESCRIPTION. The default is None.
    Y : TYPE, numpy
        DESCRIPTION. The default is None.
    X_type : TYPE, optional
        DESCRIPTION. The default is 'continuous'. 'discrete' and 'continuous'
    Y_type : TYPE, optional
        DESCRIPTION. The default is 'continuous'.
    plot_residuals : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    '''
    ## X -> Y
    if X_type == 'discrete':
        X_encode = pd.DataFrame(X).astype('category')
    else:
        X_encode = pd.DataFrame(X)
    if is_hypyopt:
        _,_,resid_XonY,_ = lightgbm_optuna(X_encode, Y, Y_type,is_plot = False,is_optM = True)
    else:
        _,_,resid_XonY,_ = lightgbm_hyperMau(X_encode, Y, Y_type,is_plot = False)
    
    if X_type == 'discrete':
        if len(set(X)) > 2:
            testData = {}
            for X_ in set(X):
                testData[X_] = resid_XonY[X == X_]
            
            f_statistic, P_XtoY = perform_anova(testData)
            print("f_statistic_X:",f_statistic)
                
        else:
            '''
            group1 = resid_XonY[X == 1]
            group2 = resid_XonY[X == 0]
            t_statistic, P_XtoY = ttest_ind(group1, group2)
            '''
            t_statistic, P_XtoY = spearmanr(X, resid_XonY)
            print("t_statistic_X:",t_statistic)
    else:
        corr, P_XtoY = spearmanr(X, resid_XonY)
        # corr, P_XtoY = pearsonr(X, resid_XonY)
        # corr, P_XtoY = cov_test(X, resid_XonY)
        print("corr_X:",corr)
    
    
    ## Y -> X
    if Y_type == 'discrete':
        Y_encode = pd.DataFrame(Y).astype('category')
    else:
        Y_encode = pd.DataFrame(Y)
        
    if is_hypyopt:
        _,_,resid_YonX,_ = lightgbm_optuna(Y_encode, X, X_type,is_plot = False,is_optM = True)
    else:
        _,_,resid_YonX,_ = lightgbm_hyperMau(Y_encode, X, X_type,is_plot = False)
    
    if Y_type == 'discrete':
        if len(set(Y)) > 2:
            testData = {}
            i = 1
            for Y_ in set(Y):
                testData['{}'.format(i)] = resid_YonX[Y == Y_]
                i += 1
            
            f_statistic, P_YtoX = perform_anova(testData)
            print("f_statistic_Y:",f_statistic)
        else:
            '''
            group1 = resid_YonX[Y == 1]
            group2 = resid_YonX[Y == 0]
            t_statistic_Y, P_YtoX = ttest_ind(group1, group2)
            '''
            t_statistic_Y, P_YtoX = spearmanr(Y, resid_YonX)
            
            print("t_statistic_Y:",t_statistic_Y)
    else:
        corr_Y, P_YtoX = spearmanr(Y, resid_YonX)
        # corr_Y, P_YtoX = pearsonr(Y, resid_YonX)
        # corr_Y, P_YtoX = cov_test(Y, resid_YonX)
        
        print("corr_Y:",corr_Y)
    
    return P_XtoY,P_YtoX