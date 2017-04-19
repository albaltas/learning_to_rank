#%%
"""
Created on Mon Mar 27 20:35:51 2017

@author: Alexandros Baltas
"""

from util import precision_recall_f1, get_ndcg, get_mapk
import pandas as pd

df = pd.read_csv("logreg_preds_large.csv")



print("NDCG@10: "+str(get_ndcg(df.qid, df.pred, df.label, k=10)))
print("NDCG@5: "+str(get_ndcg(df.qid, df.pred, df.label, k=5)))
print("MAP: "+str(get_mapk(df.qid, df.pred, df.label)))
print("F1-Score: "+str(precision_recall_f1(df.label, df.pred)[2]))