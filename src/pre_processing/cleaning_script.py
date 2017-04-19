# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 01:49:46 2017

@author: Simon Stiebellehner

This script converts the MSLR files to .csv and cleans the data so that it can be used in conventional Python ML libraries.
"""

###############################################################
# Package Imports

import pandas as pd


###############################################################
# Data Cleaning Functions

# Dropping column with 100% nan values (parsing problem probably)
def drop_col(df):
    df.drop(df.columns[-1], axis=1, inplace=True)

# Cleaning values
def split_semicolon(df, do_applymap=False):
    # removes string naming pattern '*:' from values
    if do_applymap:
        for col in range(1,len(df.columns)):
            df.loc[:,col] = df.loc[:,col].apply(lambda x: str(x).split(':')[1])
    else:
        df.ix[:, 1:len(df.columns)] = df.ix[:, 1:len(df.columns)].applymap(lambda x: str(x).split(':')[1])

    df.columns = ['rel', 'qid'] + [str(x) for x in range(1, 137)]  # renaming cols

###############################################################
# Executing import & cleaning of files

# Specify directory to your folder in which the 5 Fold folders are
# Specify folds you want to clean (default: all)
# Specify datasets you want to clean (detault: all)

directory = '/home/sist/Desktop/irdm_project/data/'
folds = [1,2,3,4,5]
datasets = ['train', 'vali', 'test']

paths = [directory+'Fold'+str(fold)+'/'+sets+'.txt' for fold in folds for sets in datasets]

count = 0
for path in paths:
    count += 1
    print(str(count)+'/'+str(len(paths)))
    df = pd.read_csv(str(path), sep=" ", header = None)
    drop_col(df)
    split_semicolon(df)
    df.to_csv(path[0:-4]+'_cleaned.csv') # saves in: directory/FoldX
    del df # freeing up memory for next file
