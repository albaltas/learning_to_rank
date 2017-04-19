# ====================================================
#                    import
# ====================================================
import util
import random
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import subprocess
import pandas as pd
# ====================================================



# ====================================================
#               Loading & File prep
# ====================================================

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> TRAIN
# Only using small xbb here
# df_all = pd.read_csv('../data/Fold1/prediction_data/train_set_small_cleaned.csv') # small
df_all = pd.read_csv('../data/Fold1/prediction_data/train_set_large_cleaned.csv') # large
df_sample = df_all.ix[:,1:]

# Splitting dataset
X, Y = util.sep_feat_labels(df_sample)
X.ix[:,1:] = StandardScaler().fit_transform(X.ix[:,1:]) # rescaling features

x_train_qid = X['qid'].copy()
x_train = X.ix[:,1:].copy()
x_train = x_train.as_matrix()
y_train = Y.as_matrix()

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> TEST
# df_test = pd.read_csv('../data/Fold1/prediction_data/test_set_small_cleaned.csv') # small
df_test = pd.read_csv('../data/Fold1/prediction_data/test_set_large_cleaned.csv') # large
df_sample_test = df_test.ix[:,1:]

# Splitting dataset
Xtest, Ytest = util.sep_feat_labels(df_sample_test)
Xtest.ix[:,1:] = StandardScaler().fit_transform(Xtest.ix[:,1:]) # rescaling features

x_test_qid = Xtest['qid'].copy()
x_test = Xtest.ix[:,1:].copy()
x_test = x_test.as_matrix()
y_test = Ytest.as_matrix()

# ====================================================



# ====================================================
#                 BEST PARAMETER
# ====================================================
p_max_depth = 26
p_n_estimators = 500
p_learning_rate = 1.0
p_criterion = "gini"
p_algorithm = "SAMME.R"
# ====================================================


print("Start Model")
# ====================================================
#                    MODEL
# ====================================================

param_depth = p_max_depth
param_nest = p_n_estimators
param_lr = p_learning_rate
param_crit = p_criterion
param_algo = p_algorithm

boostedDS = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=param_depth,criterion=param_crit),
    n_estimators=param_nest,
    learning_rate=param_lr,
    algorithm=param_algo
    ).fit(x_train, y_train)


prediction = boostedDS.predict(x_test)


ndcg10 = util.get_ndcg(x_dev_qid=x_test_qid, preds=prediction, y_dev=y_test, k=10)
ndcg5 = util.get_ndcg(x_dev_qid=x_test_qid, preds=prediction, y_dev=y_test, k=5)
mapk = util.get_mapk(x_test_qid, prediction, y_test)
precision,recall,f1 = util.precision_recall_f1(np.array(y_test), np.array(prediction))
# err10 =
# err5 =
acc_score = accuracy_score(y_test, prediction)





print("NDCG@10: " + str(ndcg10))
print("NDCG@5 : " + str(ndcg5))
print("MAP : " + str(mapk))
print("Precision : " + str(precision))
print("Recall    : " + str(recall))
print("F1        : " + str(f1))
# print("ERR@10: " + str(err10))
# print("ERR@5 : " + str(err5))
# ====================================================

