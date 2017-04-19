import pandas as pd
import os

# choose the path where you save your tuning data from auto_tuning.py
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
result_file.to_csv("result_file.csv")





