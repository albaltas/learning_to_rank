import os
import subprocess

#
# # Terminal command
# tc = os.system("java -jar RankLib-2.1-patched.jar -train xbb.txt -ranker 2 -tvs 0.8 -metric2t NDCG@10 -metric2T ERR@10 -save mymodel1.txt")
#
# out = subprocess.getoutput(tc)
#
# subproccess.call(”out > temp.txt”)

# print(out)

# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 10:50:44 2017
@author: dvisnadi
"""

# %%
import subprocess
import numpy as np

path = '/Users/Dimitri/Downloads/IRDM/'

# %%

round = np.arange(250, 371, 2)
# norm = [' -norm sum ', ' -norm zscore ', ' ']
norm = [' -norm zscore ']
tc = np.arange(17, 30, 1)

# %%

call_lib = 'java -jar RankLib.jar '
params_fix = '-train xbb.txt -ranker 2 -tvs 0.8 '
param_metric1 = '-metric2t NDCG@10 '
param_metric2 = '-metric2T ERR@10 '

tune_round = '-round '
tune_tc = ' -tc '


# %%

runs = 400

for i in range(0, runs):
    param_round = str(int(np.random.choice(round)))
    param_tc = str(int(np.random.choice(tc)))
    param_norm = str(np.random.choice(norm))


    cmd_input = call_lib + params_fix + param_metric1 + param_metric2 + tune_round + param_round + tune_tc + param_tc + param_norm

    out = open(path + 'tuning_data/tune' + str(i + 1) + '_rand' + str(np.random.randint(100000)) + '.txt', 'w')
    subprocess.call(cmd_input, stdout=out, shell=True)
