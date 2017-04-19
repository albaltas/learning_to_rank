# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 10:50:44 2017
@author: sist
"""

#%% 
import subprocess
import numpy as np


#%%

tree = np.arange(500, 1500, 100)
leaf = np.arange(5, 20, 2)
shrinkage = np.linspace(0.0001, 0.5, 20)
tc = np.arange(64, 1024, 64)
mls = np.arange(0, 5, 1)


#%%

call_lib = 'java -jar ../../RankLib.jar '
params_fix = '-train ../../prediction_data_ori/train_set_small.txt -ranker 6 -tvs 0.8 '
param_metric1 = '-metric2t NDCG@10 '
param_metric2 = '-metric2T ERR@10 '

tune_tree= ' -tree '
tune_leaf = ' -leaf '
tune_shrinkage = ' -shrinkage '
tune_tc = ' -tc '
tune_mls = ' -mls '
tune_estop = ' -estop '


#%%

runs = 60

for i in range(0, runs):
    print('Run '+str(i+1)+' of '+str(runs)+':')
    print('--------------------------------')
    param_tree = str(np.random.choice(tree))
    param_leaf = str(np.random.choice(leaf))
    param_shrinkage = str(np.random.choice(shrinkage))
    param_mls = str(np.random.choice(mls))
    param_tc = str(np.random.choice(tc))

    print('++++++++++++++++++++++++++++++++')
    
    cmd_input = call_lib+params_fix+param_metric1+param_metric2+tune_tree+param_tree+tune_leaf+param_leaf+tune_shrinkage+param_shrinkage+tune_tc+param_tc+tune_mls+param_mls#+tune_lr+param_lr
    
    
    out = open('../../tuning_data/tune'+str(i+1)+'_rand'+str(np.random.randint(100000))+'.txt', 'w')
    subprocess.call(cmd_input, stdout=out, shell=True)
