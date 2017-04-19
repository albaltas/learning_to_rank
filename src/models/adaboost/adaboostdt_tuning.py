# ====================================================
#                    import
# ====================================================
import util
import random
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
# ====================================================



# ====================================================
#               Loading & File prep
# ====================================================
seed = 1
random.seed(seed)
np.random.seed(seed)

# Only using small xbb here
df_all = pd.read_csv('../data/Fold1/prediction_data/train_set_small_cleaned.csv')
df_sample = df_all.ix[:,1:]

# Splitting dataset
X, Y = util.sep_feat_labels(df_sample)
X.ix[:,1:] = StandardScaler().fit_transform(X.ix[:,1:]) # rescaling features

x_train, x_dev, y_train, y_dev = train_test_split(X, Y, test_size=0.2, random_state=seed)

x_train_qid = x_train['qid'].copy()
x_train = x_train.ix[:,1:].copy()
x_dev_qid = x_dev['qid'].copy()
x_dev = x_dev.ix[:,1:].copy()

x_train = x_train.as_matrix()
y_train = y_train.as_matrix()
x_dev = x_dev.as_matrix()
# ====================================================



# ====================================================
#                 PARAMETER TUNING
# ====================================================
p_max_depth = np.arange(25.9, 26.1, 0.02)
p_n_estimators = np.arange(499, 500, 0.1)
p_learning_rate = np.arange(0.9, 1.08, 0.02)
p_criterion = ["gini", "entropy"]
p_algorithm = ["SAMME", "SAMME.R"]
# ====================================================



# ====================================================
#                    MODEL
# ====================================================
runs = 100
with open("variable_file.txt", "a") as text_file:
    for i in range(runs):
        param_depth = int(np.random.choice(p_max_depth))
        param_nest = int(np.random.choice(p_n_estimators))
        param_lr = float(np.random.choice(p_learning_rate))

        param_crit = str(np.random.choice(p_criterion))
        param_algo = str(np.random.choice(p_algorithm))

        boostedtree = AdaBoostClassifier(
            DecisionTreeClassifier(max_depth=param_depth,criterion=param_crit),
            n_estimators=param_nest,
            learning_rate=param_lr,
            algorithm=param_algo
            ).fit(x_train, y_train)


        prediction = boostedtree.predict(x_dev)

        ndcg = round(util.get_ndcg(x_dev_qid=x_dev_qid, preds=prediction, y_dev=y_dev) * 100, 2)
        mapk = util.get_mapk(x_dev_qid, prediction, y_dev)
        precision,recall,f1 = util.precision_recall_f1(np.array(y_dev), np.array(prediction))

        acc_score = accuracy_score(y_dev, prediction)
        text = acc_score, ndcg,mapk,f1,precision,recall,param_depth,param_nest,param_lr,param_algo,param_crit

        print(text)
        text_file.write(str(text) + "\n")
# ====================================================
