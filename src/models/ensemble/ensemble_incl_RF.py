# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 19:56:37 2017

@author: Simon Stiebellehner
https://github.com/stiebels

This file implements three classifiers:
 - XGBoost (Gradient Boosted Trees)
 - Random Forest
 - AdaBoost (Decision Trees)
 
These three base classifiers are then ensembled in two ways:
 (1) Weighting and combining their individual outputs to form a 'committee vote'.
 (2) Using the outputs of the three models as inputs to a 4th classifier (Gradient Boosted Trees),
     which then outputs the final predictions (stacking).
"""

#%%
# Package Imports

import sys
sys.path.append('/home/sist/Desktop/irdm_project/repo/code/') # If module util is not found, append path to it like this before loading

import util
import random
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from mlxtend.classifier import StackingClassifier, EnsembleVoteClassifier
from skopt import gp_minimize

seed = 1
random.seed(seed)
np.random.seed(seed)




#%% Function initialization for later

# Tuning set for last ensembling weights
def get_meta_sets():
    '''
    Use a meta-set (additional holdout set) for training the meta-classifier in the stacked ensemble.
    '''
    df_ens_eval = pd.read_csv('/home/sist/Desktop/irdm_project/data/fold1_cleaned.csv')
    df_ens_sel = df_ens_eval.drop('Unnamed: 0', axis=1)
    df_sample = util.make_sample(df_ens_sel, num_q=500) # randomly sampling from the data
    X_ens, Y_ens = util.sep_feat_labels(df_sample)
    df_ens_sel.ix[:,1:] = StandardScaler().fit_transform(df_sample.ix[:,1:]) # rescaling features
    
    x_train_ens, x_dev_ens, y_train_ens, y_dev_ens = train_test_split(X_ens, Y_ens, test_size=0.2, random_state=seed)
    
    x_train_ens_qid = x_train_ens['qid'].copy()
    x_train_ens = x_train_ens.ix[:,1:].copy()
    x_dev_ens_qid = x_dev_ens['qid'].copy()
    x_dev_ens = x_dev_ens.ix[:,1:].copy()
    
    # In case of using cross_val_score we do not need separate train and dev, so let's merge it here again
    X_ens = pd.concat([x_train_ens, x_dev_ens], axis=0, join='outer', join_axes=None, ignore_index=False,
              keys=None, levels=None, names=None, verify_integrity=False,
              copy=True)
    
    Y_ens = pd.concat([y_train_ens, y_dev_ens], axis=0, join='outer', join_axes=None, ignore_index=False,
              keys=None, levels=None, names=None, verify_integrity=False,
              copy=True)
              
    X_ens_qid = pd.concat([x_train_ens_qid, x_dev_ens_qid], axis=0, join='outer', join_axes=None, ignore_index=False,
              keys=None, levels=None, names=None, verify_integrity=False,
              copy=True)
              
    return X_ens, Y_ens, X_ens_qid
    
    
def constrained_sum_sample_pos(n, total):
    dividers = sorted(random.sample(range(1, total), n - 1))
    return [a - b for a, b in zip(dividers + [total], [0] + dividers)]


#%%
# Data Import

# SPECIFY

directory = '/home/sist/Desktop/irdm_project/data/Fold1/xbb'
fold = 1 # 1 | 2 | 3 | 4 | 5
dataset = 'train' # 'train' | 'vali' | 'test'

path = directory+'Fold'+str(fold)+'/'+dataset+'_cleaned.csv' # importing CLEANED dataset

df_train = pd.read_csv(str(path), index_col=0)
df_test = pd.read_csv(directory+'Fold'+str(fold)+'/'+'test'+'_cleaned.csv', index_col=0)


#%%
# Splitting dataset
x_train, y_train = util.sep_feat_labels(df_train)
x_test, y_test = util.sep_feat_labels(df_test)

x_train.ix[:,1:] = StandardScaler().fit_transform(x_train.ix[:,1:]) # rescaling features
x_test.ix[:,1:] = StandardScaler().fit_transform(x_test.ix[:,1:]) # rescaling features

x_train_qid = x_train['qid'].copy()
x_train = x_train.ix[:,1:].copy()
x_test_qid = x_test['qid'].copy()
x_test = x_test.ix[:,1:].copy()


#%% Bayesian Optimization Functions

def ada_objective(params):
    trees, lr = params
    clf.set_params(n_estimators=trees, learning_rate=lr, random_state=1)
    return -np.mean(cross_val_score(clf, X, Y, n_jobs=-1, scoring='f1_micro', verbose=2))

def xgb_objective(params):
    max_depth, min_child, gamma, scale_pos_weight = params
    clf.set_params(scale_pos_weight=scale_pos_weight, max_depth=max_depth, min_child_weight=min_child, gamma=gamma)
    return -np.mean(cross_val_score(clf, X, Y, n_jobs=-1, scoring='f1_micro', verbose=2))
    
def rf_objective(params):
    trees = params[0]
    clf.set_params(n_estimators=trees, random_state=1)
    return -np.mean(cross_val_score(clf, X, Y, n_jobs=-1, scoring='f1_micro', verbose=2))
    
def stacking_objective(params):
    max_depth, min_child, gamma, scale_pos_weight = params
    clf = StackingClassifier(classifiers=[clf1, clf2, clf3], meta_classifier=xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, max_depth=max_depth, min_child_weight=min_child, gamma=gamma))
    return np.mean(cross_val_score(clf, X_ens, Y_ens, n_jobs=-1, scoring='f1_micro', verbose=2))

#%% 

# Dashboard
estimator = 'ensemble'
tune = False


if estimator == 'ada':
    '''
    AdaBoost (with Decision Stumps)
    '''
    clf = AdaBoostClassifier(n_estimators = 143, learning_rate = 0.9321253)
    if tune == True:
        params = [(20, 200), (0.3, 1.5)]
        res_gp = gp_minimize(ada_objective, params, n_calls = 10, random_state=1)
    '''
    Optimumg Parameters:
        n_estimators = 143
        learning_rate = 0.9321253
    '''


elif estimator == 'xgb':
    '''
    XGBoost (Gradient Boosted Trees)
    '''
    opt_max_depth = 7
    opt_min_child = 2.0550232716937149
    opt_gamma = 1
    opt_scale_pos_weight = 0
    clf = xgb.XGBClassifier(max_depth = 7, min_child_weight = 2.0550232716937149, gamma = 1, scale_pos_weight = 0)
    if tune == True:
        params = [(3, 10), (0.5, 5), (0, 5), (0, 1)]
        res_gp = gp_minimize(xgb_objective, params, n_calls = 10, random_state=1)
        '''
        Optimum Parameters:
            - max_depth: 7
            - min_child_weight: 2.0550232716937149
            - gamma: 1
            - scale_pos_weight: 0
        '''

elif estimator == 'rf':
    '''
    Random Forest
    '''
    opt_n_estimators = 139
    clf = RandomForestClassifier(n_estimators = opt_n_estimators, random_state=1)
    if tune == True:
        params = [(10,200)]
        res_gp = gp_minimize(rf_objective, params, n_calls = 10, random_state=1)
        '''
        Optimum Parameters:
            - n_estimators: 139
        '''

elif estimator == 'stacking':
    '''
    Stacked Ensemble (meta-classifier)
    '''
    clf1 = RandomForestClassifier(n_estimators = 139)
    clf2 = xgb.XGBClassifier(max_depth = 7, min_child_weight = 2.0550232716937149, gamma = 1, scale_pos_weight = 0)
    clf3 = AdaBoostClassifier(n_estimators = 143, learning_rate = 0.9321253)
    mclf = xgb.XGBClassifier(max_depth = 8, min_child_weight = 0.9155236764595901, gamma = 2, scale_pos_weight = 1)
    clf = StackingClassifier(classifiers=[clf1, clf2, clf3], meta_classifier=mclf)
    if tune == True:
        X_ens, Y_ens, X_ens_qid = get_meta_sets()
        params = [(3, 10), (0.5, 5), (0, 5), (0, 1)]
        res_gp = gp_minimize(stacking_objective, params, n_calls = 10, random_state=1)
        '''
        Optimum Parameters:
            - max_depth: 8
            - min_child_weight: 0.9155236764595901
            - gamma: 2
            - scale_pos_weight: 1
        '''
        
        
elif estimator == 'ensemble':
    '''
    Weighted Ensemble
    '''
    clf1 = RandomForestClassifier(n_estimators = 139)
    clf2 = xgb.XGBClassifier(max_depth = 7, min_child_weight = 2.0550232716937149, gamma = 1, scale_pos_weight = 0)
    clf3 = AdaBoostClassifier(n_estimators = 143, learning_rate = 0.9321253)
    clf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3], weights=[2,6,4], voting='soft')
    if tune == True:
        acc_list = []
        param_list = []
        X_ens, Y_ens, X_ens_qid = get_meta_sets()
        runs = 10
        for run in range(0, runs):
            rand_weights = constrained_sum_sample_pos(3,12)
            clf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3], weights=rand_weights)
            acc_list.append(np.mean(cross_val_score(clf, x_train, y_train, n_jobs=-1, scoring='f1_micro', verbose=2)))
            param_list.append(rand_weights)
            best_param = param_list[acc_list.index(max(acc_list))]
        print('Best Weights: '+str(best_param))
        print('... with best F1 Accuracy: '+str(max(acc_list)))
        '''
        Optimum Weights:
            - [2, 6, 4]
        '''
        

if tune == False:
    '''
    3 runs of CV on train/dev set.
    '''
    score = np.mean(cross_val_score(clf, np.array(x_train), np.array(y_train), scoring='f1_micro', verbose=2)) # mean of 3-runs of cross-validation
    print('F1 Micro Score: '+str(score))
    

#%%
# Computing some scores on test set

clf_fit = clf.fit(x_train, y_train)
preds = clf_fit.predict(x_test)
print('F1 Score --> '+str(f1_score(np.array(preds), np.array(y_test), average='micro')))
print('NDCG@k --> '+str(round(util.get_ndcg(x_dev_qid=x_dev_qid, preds=preds, y_dev=y_test)*100, 2))+' %')    

