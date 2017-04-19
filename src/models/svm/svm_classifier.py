import util
import random
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

import itertools
from sklearn import svm, linear_model, cross_validation
from sklearn.svm import SVC

seed = 1
random.seed(seed)
np.random.seed(seed)

# Only using small xbb here
df_all = pd.read_csv('../data/Fold1/xaa.csv')
df_sample = df_all.ix[:,1:]

# Splitting dataset
X, Y = util.sep_feat_labels(df_sample)
X.ix[:,1:] = StandardScaler().fit_transform(X.ix[:,1:]) # rescaling features

x_train, x_dev, y_train, y_dev = train_test_split(X, Y, test_size=0.2, random_state=seed)

x_train_qid = x_train['qid'].copy()
x_train = x_train.ix[:,1:].copy()
x_dev_qid = x_dev['qid'].copy()
x_dev = x_dev.ix[:,1:].copy()


#===============================================================
# Code from: https://gist.github.com/agramfort/2071994
#===============================================================

def transform_pairwise(X, y):
    """Transforms data into pairs with balanced labels for ranking
    Transforms a n-class ranking problem into a two-class classification
    problem. Subclasses implementing particular strategies for choosing
    pairs should override this method.
    In this method, all pairs are choosen, except for those that have the
    same target value. The output is an array of balanced classes, i.e.
    there are the same number of -1 as +1
    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        The data
    y : array, shape (n_samples,) or (n_samples, 2)
        Target labels. If it's a 2D array, the second column represents
        the grouping of samples, i.e., samples with different groups will
        not be considered.
    Returns
    -------
    X_trans : array, shape (k, n_feaures)
        Data as pairs
    y_trans : array, shape (k,)
        Output class labels, where classes have values {-1, +1}
    """
    X_new = []
    y_new = []
    y = np.asarray(y)
    if y.ndim == 1:
        y = np.c_[y, np.ones(y.shape[0])]
    comb = itertools.combinations(range(X.shape[0]), 2)
    for k, (i, j) in enumerate(comb):
        if y[i, 0] == y[j, 0] or y[i, 1] != y[j, 1]:
            # skip if same target or different group
            continue
        X_new.append(X[i] - X[j])
        y_new.append(np.sign(y[i, 0] - y[j, 0]))
        # output balanced classes
        if y_new[-1] != (-1) ** k:
            y_new[-1] = - y_new[-1]
            X_new[-1] = - X_new[-1]
    return np.asarray(X_new), np.asarray(y_new).ravel()


class RankSVM(svm.LinearSVC):
    """Performs pairwise ranking with an underlying LinearSVC model
    Input should be a n-class ranking problem, this object will convert it
    into a two-class classification problem, a setting known as
    `pairwise ranking`.
    See object :ref:`svm.LinearSVC` for a full description of parameters.
    """

    def fit(self, X, y):

        X_trans, y_trans = transform_pairwise(X, y)
        super(RankSVM, self).fit(X_trans, y_trans)
        return self

#===============================================================


x_train = x_train.as_matrix()
y_train = y_train.as_matrix()
x_dev = x_dev.as_matrix()

# rank_svm = RankSVM().fit(x_train, y_train) #github version, gives negative values
rank_svm = SVC().fit(x_train, y_train)
preds = rank_svm.predict(x_dev)

print('F1 Score -->  '+str(f1_score(np.array(preds), np.array(y_dev), average='micro')))
print('NDCG@k   -->  '+str(round(util.get_ndcg(x_dev_qid=x_dev_qid, preds=preds, y_dev=y_dev)*100, 2))+' %')
print('MAP@k   -->  '+str(util.MAPk(y_dev, preds, 12)))

