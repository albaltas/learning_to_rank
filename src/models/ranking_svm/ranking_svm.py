#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 17:27:24 2017

@author: Simon Stiebellehner

This file implements a Support Vector Machine for Ranking tasks.
This is realized by transforming the data into pairwise representation (turning the multi-class problem
into a binary classification) and subsequently training a SVM with a linear kernel on the transformed data.
Coefficients of the trained model are then used to predict the ordering of unseen data.

"""
#%%

import sys
sys.path.append(“../../“)
import util
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import cross_val_score
from skopt import gp_minimize
from util import get_ndcg
import itertools
from collections import defaultdict
from sklearn.svm import LinearSVC


seed = 1
random.seed(seed)
np.random.seed(seed)

#%% Loading data

directory = '/specify/path/here'
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

#%%

def svm_objective(params):
    '''
    Bayesian hyperparameter optimization, 3 rounds of CV
    '''
    penalty, max_c, intercept, class_w = params 
    clf.set_params(penalty=penalty, C=max_c,fit_intercept=intercept, class_weight=class_w)
    return -np.mean(cross_val_score(clf, x_train, y_test, n_jobs=-1, scoring='f1_micro', verbose=2))

#%%

def pair_transform(x_train, y_train):
    '''
    Performs the pairwise transformation for pairwise ranking
    WARNING: Computationally expensive due to lack of vectorization
    '''
    comb_vals = []
    
    comb_iter = itertools.combinations(range(x_train.shape[0]), 2)
    for row in comb_iter:
        comb_vals.append(row)
     
    xpair = []
    ypair = []
    d = []
    o = 0
    
    for i,j in comb_vals:
        if y_train[i] != y_train[j]:
            xpair.append(x_train[i] - x_train[j])
            if (y_train[i] - y_train[j]) < 0:
                d.append(-1)
            else:
                d.append(1)
            if d[-1] < 0:
                ypair.append(-1)
            else:
                ypair.append(1)
            if ypair[-1] == (-1) ** o:
                o += 1
                continue
            else:
                o += 1
                d[-1] = np.negative(d[-1])
                xpair[-1] = np.negative(xpair[-1])
                ypair[-1] = np.negative(ypair[-1])
    
    return np.asarray(xpair), np.asarray(ypair)


def get_qdoc(df, qid):
    '''
    Extracts qid, features, labels from a dictionary
    '''
    qdoc = make_qdoc_dict(df)
    x_qid = qdoc[qid][:,2:] # features
    y_qid = qdoc[qid][:,0] # labels
    return x_qid, y_qid, qid


def make_qdoc_dict(df):
    '''
    Creates dictionary of qid: features, labels
    '''
    qdoc = defaultdict()
    qid_group = df.groupby('qid')

    for qid in list(qid_group.groups.keys()):
        docs_ix = qid_group.groups[qid]
        qdoc[qid] = np.array(train.ix[docs_ix]) 
    return qdoc

  
#%% Executing functions defined above to eventually bring the data into appropriate format

df = train

xpl = []
ypl = []

for q in df['qid'].unique():
    x, y, q = get_qdoc(df, q)
    xp, yp = pair_transform(x, y)
    xpl.append(xp)
    ypl.append(yp)
    
x_f = []
y_f = []

for x in xpl:
    for y in x:
        x_f.append(y)

for y in ypl:
    for z in y:
        y_f.append(z)
        
x_f = np.array(x_f)
y_f = np.array(y_f)


#%%

tune = False

if tune == True:
    clf = LinearSVC()
    params = [('l1', 'l2'), (0.1, 1.5), ('True', 'False'), ('balanced', None)]
    res_gp = gp_minimize(svm_objective, params, n_calls = 10, random_state=1)
    '''
    Optimum Parameters:
        - C: 0.7321432
        - penalty: l2
        - fit_intercept: True
        - class_weight: None
    '''
    
else:
    # Defining & Fitting
    clf = LinearSVC(C=0.7321432, penalty='l2', fit_intercept=True, class_weight=None)
    clf.fit(x_f, y_f)

#%%

weighted = np.dot(x_test, clf.coef_.ravel()) # Prediction

#%%

util.get_ndcg(x_test_qid, weighted, y_test, k=10) # Calling NDCG scoring function for evaluation

