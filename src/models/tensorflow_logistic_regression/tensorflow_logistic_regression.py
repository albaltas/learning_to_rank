'''
    @ Alexandros Baltas

    Tensorflow implementation of Logistic Regression with L2 regularization
    using Adam Optimizer.

'''
#%%
import sys
sys.path.append(“../../“)
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from util import precision_recall_f1, get_ndcg, get_mapk

one_hot_enc = OneHotEncoder(5)
standard_scaler = StandardScaler()


df = pd.read_csv("dummy_path", index_col=0)

features = df.ix[:, df.columns != 'rel']
set_X = features.ix[:, features.columns != 'qid'].as_matrix()
qids = features.ix[:, features.columns == 'qid'].values.tolist()

set_Y = df.ix[:, df.columns == "rel"].values


train_X, dev_X, train_labels_Y, dev_labels_Y = train_test_split(set_X, set_Y, test_size=0.2)

train_Y = one_hot_enc.fit_transform(train_labels_Y).toarray()
dev_Y = one_hot_enc.fit_transform(dev_labels_Y).toarray()

standard_scaler.fit(set_X)
set_X = standard_scaler.transform(set_X)



test_df = pd.read_csv("../../prediction_data/test_set_large_cleaned.csv", index_col=0)

test_features = test_df.ix[:, test_df.columns != 'rel']
qids = test_features.ix[:, test_features.columns == 'qid'].values.tolist()
test_set_X = test_features.ix[:, test_features.columns != 'qid'].as_matrix()

test_set_Y = test_df.ix[:, test_df.columns == "rel"].values


test_oh_Y = one_hot_enc.transform(test_set_Y).toarray()
test_set_X = standard_scaler.transform(test_set_X)






number_of_training_epochs = 300
learning_rate = 0.00001
beta = 1.004

 
x = tf.placeholder(tf.float32, [None, 136])
y = tf.placeholder(tf.float32, [None, 5]) 

 
W = tf.Variable(tf.random_normal([136, 5], stddev=1e-4))
b = tf.Variable(tf.zeros([5]))


logits = tf.nn.softmax(tf.matmul(x, W) + b) 
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)) + beta*tf.nn.l2_loss(W)

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

predictions = tf.argmax(logits, 1)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(number_of_training_epochs):         
        opt = sess.run(optimizer, feed_dict={x: train_X,y: train_Y})
        train_pred = predictions.eval({x: train_X, y: train_Y})
        dev_pred = predictions.eval({x: dev_X, y: dev_Y})

        print("Epoch "+str(epoch))
        print("Training F1 Score: "+str(f1_score(train_labels_Y, train_pred, average="micro")))
        print("Dev F1 Score: "+str(f1_score(dev_labels_Y, dev_pred, average="micro")))

        print("---")
    
    test_pred = predictions.eval({x: test_set_X, y: test_oh_Y})
    
    
    predictions = pd.DataFrame({"qid" : np.array([x[0] for x in qids]), "label": np.array([x[0] for x in test_set_Y.tolist()]), "pred": test_pred})
    predictions.to_csv('logreg_preds_large.csv')
    
