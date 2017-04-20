
"""
@author: gompos
"""

import subprocess
import numpy as np
import pandas as pd
import os

path = '/home/andreas/Desktop/'
call_lib = 'java -jar /home/andreas/Desktop/ranklib.jar '


params_fix = '-train /home/andreas/Desktop/xbb -ranker 3 -tvs 0.8 '
param_metric1 = '-metric2t NDCG@10 '
param_metric2 = '-metric2T NDCG@10 '


#Tune hyperparameters
norms = [' -norm sum ', ' -norm zscore ', ' ']
boosting_rounds_range = [500,1000,1500]
tolerance_range=[0.0001,0.0005,0.001,0.002,0.003,0.005]
max_range=[1,2,5,8,16,32]


summ=0
for norm in norms:
    for boosting_rounds in boosting_rounds_range:
        for tolerance in tolerance_range:
            for maxx in max_range:
                
                summ+=1
                print('--------------------------------')

                param_norm = str(norm)
                param_boosting_rounds=str(boosting_rounds)
                param_tolerance=str(tolerance)
                param_max=str(maxx)
                print('Norm: '+str(param_norm))
                print('Boosting Rounds: '+str(param_boosting_rounds))
                print("Tolerance between rounds: "+str(param_tolerance))
                print("Max: "+str(param_max))

                print('++++++++++++++++++++++++++++++++')

                cmd_input = call_lib+params_fix+param_metric1+param_metric2+param_norm+'-round '+param_boosting_rounds\
                            +' -tolerance '+param_tolerance+' -max '+param_max+' -save tuning_data/model{} '.format(summ)
                    
                out = open(path+'tuning_data/tune'+str(summ)+'.txt', 'w')
                subprocess.call(cmd_input, stdout=out, shell=True)
 
    
#concatenate results of multiple files in a dataframe
tuning_data_path = "tuning_data/"

result_file = pd.DataFrame()


for filename in os.listdir('tuning_data/'):
    if filename.endswith(".txt"):
        result_list = []

        with open(tuning_data_path+filename) as f:
                for line in f:
                    if ':' in line:
                        result_list.append(line)

        result = pd.DataFrame(result_list, columns=["parameter"])
        result = pd.DataFrame(result.parameter.str.split(':',1).tolist(), columns = ['parameters',filename])
        result_file['parameters'] = result['parameters']
        result_file[filename] = result[filename].map(lambda x: x.lstrip("").rstrip('\n'))


result_file = result_file.transpose()
result_file.columns = result_file.iloc[0]
result_file = result_file[1:]
result_file.sort_values(["NDCG@10 on validation data","NDCG@10 on training data"],ascending=False,inplace=True)
result_file.to_csv("tuning_data/result_file.csv")        


#Test performance on the best model
param_metric2 = '-metric2T NDCG@10 '
cmd_input = call_lib+ ' -load tuning_data/model{} '.format(int(result_file.index[0][4:-4])) +param_metric2+' -test /home/andreas/Desktop/test.txt '
out = open(path+'tuning_data/test_performance.txt', 'w')
subprocess.call(cmd_input, stdout=out, shell=True)


param_metric2_ = ['-metric2T NDCG@10 ','-metric2T NDCG@5 ','-metric2T ERR@@10 ','-metric2T ERR@@5 ','-metric2T MAP ']
for param_metric2 in param_metric2_:
    cmd_input = call_lib+ ' -load tuning_data/model{} '.format(int(result_file.index[0][4:-4])) +param_metric2+' -test /home/andreas/Desktop/test.txt '
    out = open(path+'tuning_data/test_performance{}.txt'.format(param_metric2), 'w')
    subprocess.call(cmd_input, stdout=out, shell=True)






#test with pretrained models
#best model found from the cross validation: model83
param_metric2_ = ['-metric2T NDCG@10 ','-metric2T NDCG@5 ','-metric2T ERR@10 ','-metric2T ERR@5 ','-metric2T MAP ']

#small dataset
for param_metric2 in param_metric2_:
    cmd_input = call_lib+ ' -load tuning_data/model{} '.format(83) +param_metric2+' -test /home/andreas/Desktop/test_set_small.txt '
    out = open(path+'tuning_data/test_performance{}_small.txt'.format(param_metric2), 'w')
    subprocess.call(cmd_input, stdout=out, shell=True)
    
#large dataset  
for param_metric2 in param_metric2_:
    cmd_input = call_lib+ ' -load tuning_data/model{} '.format(83) +param_metric2+' -test /home/andreas/Desktop/test_set_large.txt '
    out = open(path+'tuning_data/test_performance{}_large.txt'.format(param_metric2), 'w')
    subprocess.call(cmd_input, stdout=out, shell=True)