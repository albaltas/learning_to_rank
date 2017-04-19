# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 10:50:44 2017
@author: Simon Stiebellehner
https://github.com/stiebels

This script does automatized random search for parameter tuning for Lemur RankLib.
"""

#%% 
import subprocess
import numpy as np

path = '/home/sist/Desktop/irdm_project/' # specify path to your irdm directory here

#%%

epoch = np.arange(20,81,5)
norm = [' -norm sum ', ' -norm zscore ', ' ']
node = np.arange(5,35,10)
layer = np.arange(1,3,1)
learning_rate = np.linspace(0.000001, 0.001, 10)

#%%

call_lib = 'java -jar /home/sist/Desktop/irdm_project/RankLib.jar '
params_fix = '-train /home/sist/Desktop/irdm_project/data/Fold1/xbb -ranker 1 -tvs 0.8 '
param_metric1 = '-metric2t NDCG@10 '
param_metric2 = '-metric2T ERR@10 '

tune_epoch = '-epoch '
tune_layer = ' -layer '
tune_node = ' -node '
tune_lr = ' -lr '


#%%

runs = 10

for i in range(0, runs):
    print('Run '+str(i+1)+' of '+str(runs)+':')
    print('--------------------------------')
    param_epoch = str(np.random.choice(epoch))
    param_norm = str(np.random.choice(norm))
    param_node = str(int(np.random.choice(node)))
    param_layer = str(int(np.random.choice(layer)))
    param_lr = str(np.random.choice(learning_rate))
    print('Epochs: '+str(param_epoch))
    print('Norm: '+str(param_norm))
    print('Layer: '+str(param_layer))
    print('Node: '+str(param_node))
    print('LR: '+str(param_lr))
    print('++++++++++++++++++++++++++++++++')
    
    cmd_input = call_lib+params_fix+param_metric1+param_metric2+tune_epoch+param_epoch+param_norm+tune_layer+param_layer+tune_node+param_node+tune_lr+param_lr
    
    
    out = open(path+'tuning_data/tune'+str(i+1)+'_rand'+str(np.random.randint(100000))+'.txt', 'w')
    subprocess.call(cmd_input, stdout=out, shell=True)
