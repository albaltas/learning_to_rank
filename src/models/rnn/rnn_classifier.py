# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 19:56:37 2017

@author: Simon Stiebellehner
https://github.com/stiebels

This file implements a Multi-layered (stacked), bidirectional Recurrent Neural Network for classification,
followed by two feed-forward ReLu layers and a softmax function for multiclass-classification.
It uses Dropout for regularization and batch normalization to supplement convergence speed.
Optimization of hyperparameters was performed employing Bayesian optimization using hyperopt.
"""

#%%
# Package Imports
import sys
sys.path.append(“../../“)
from util import make_sample, sep_feat_labels
import numpy as np
import random
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.metrics import f1_score
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.layers.wrappers import Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


seed = 1
random.seed(seed)
np.random.seed(seed)

#%%
# Data Import

# SPECIFY 

directory = '/specify/path/here/'
fold = 1 # 1 | 2 | 3 | 4 | 5
dataset = 'train' # 'train' | 'vali' | 'test'

path = directory+'Fold'+str(fold)+'/'+dataset+'_cleaned.csv' # importing CLEANED dataset


df_train = pd.read_csv(str(path), index_col=0)
df_test = pd.read_csv(directory+'Fold'+str(fold)+'/'+'test'+'_cleaned.csv', index_col=0)


#%%
# Splitting dataset & Pipeline
    
x_train, y_train = sep_feat_labels(df_train)
x_test, y_test = sep_feat_labels(df_test)

x_train = x_train.ix[:,1:].copy()
x_test_qid = x_test['qid'].copy()
x_test = x_test.ix[:,1:].copy()

def pipeline(x, y):
    x = preprocessing.scale(x) # also usable - doesnt really matter which: x = preprocessing.scale(x)
    x = x.reshape(x.shape[0], 1, x.shape[1]) # converting to numpy arrays
    y = np_utils.to_categorical(y, nb_classes=5) #Y.values.reshape(y_shape[0], 1) # np_utils.to_categorical(Y)
    return x, y


def make_dev_set(x_train, y_train):
    '''
    Extracts dev set from train set
    '''
    
    x_train_ens, x_dev_ens, y_train_ens, y_dev_ens = train_test_split(x_train, y_train, test_size=0.2, random_state=seed)
    
    x_train_ens_qid = x_train_ens['qid'].copy()
    x_train_ens = x_train_ens.ix[:,1:].copy()
    x_dev_ens_qid = x_dev_ens['qid'].copy()
    x_dev_ens = x_dev_ens.ix[:,1:].copy()
    
    x_dev_ens, y_dev_ens = pipeline(x_dev_ens, y_dev_ens)
    x_train_ens, y_train_ens = pipeline(x_train_ens, y_train_ens)
               
    return x_dev_ens y_dev_ens, x_dev_ens_qid, x_train_ens, y_train_ens

x_dev, y_dev, x_dev_qid, x_train_dev, y_train_dev = make_dev_set(x_train, y_train) #train/dev split

x_train, y_train = pipeline(x_train, y_train) # full train set
x_test, y_test = pipeline(x_test, y_test) # test set

#%%

def nn_objective(params):
    '''
    Bayesian optimization for neural network; more complex as many parameters to tune and due to model complexity
    Basically tunes: # of layers, # of nodes per hidden layer, dropout probability, batch size
    '''
    param_list.append(params)
    print ('Params testing: ', params)
    model = Sequential()
    print(params['units1'])
    model.add(Bidirectional(LSTM(output_dim=int(params['units1']), return_sequences=True), input_shape=(None, 136)))
    model.add(BatchNormalization())
    model.add(Dropout(params['dropout1']))
    ret = True
    if params['choice']['layers']== 'two':
        ret = False
    model.add(Bidirectional(LSTM(output_dim=int(params['units2']),return_sequences=ret)))
    model.add(BatchNormalization())
    model.add(Dropout(params['dropout2']))

    if params['choice']['layers']== 'three':
        model.add(Bidirectional(LSTM(output_dim=int(params['choice']['units3']), return_sequences=False))) 
        model.add(BatchNormalization())
        model.add(Dropout(params['choice']['dropout3']))
        
    if params['choice']['layers']== 'four':
        model.add(Bidirectional(LSTM(output_dim=int(params['choice']['units3_4']),return_sequences=True))) 
        model.add(BatchNormalization())
        model.add(Dropout(params['choice']['dropout3_4']))
        model.add(Bidirectional(LSTM(output_dim=int(params['choice']['units4']), return_sequences=False)))
        model.add(BatchNormalization())
        model.add(Dropout(params['choice']['dropout4']))
        
    model.add(Dense(output_dim=5, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Dense(output_dim=5, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Dense(output_dim=5))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=params['optimizer'], metrics=['accuracy', 'fmeasure']) # using categorical cross entropy loss, optimizer is 'adam' nd output metrics (printed) are accuracy and f1 accuracy

    model.fit(np.array(x_train_dev), np.array(y_train_dev), nb_epoch=params['nb_epochs'], batch_size=int(params['batch_size']), verbose = 1, validation_data = (np.array(x_dev), np.array(y_dev)), callbacks=callbacks) # fitting model & saving best model over n epochs
    best_model = load_model('/home/sist/Desktop/irdm_project/weights.hdf5') # loading best model
    pred_auc = best_model.predict_proba(np.array(x_dev), batch_size = 128, verbose = 0) # predicting using loaded model
    acc = f1_score(np.argmax(y_dev, axis=1), np.argmax(pred_auc, axis=1), average='micro') # computing f1 score of this run
    acc_list.append(acc) # appending f1 score to a list (for later calculation of best f1 score over all tuning runs)
    print('F1 Score of this run:'+ str(acc)) # printing f1 score of this run
    sys.stdout.flush() 
    return {'loss': -acc, 'status': STATUS_OK}

#%%
# Defining RNN CLASSIFIER with optimum params

def rnn_bi_clf_network():
    '''
    Defines model architecture
    '''
    model = Sequential()
    model.add(Bidirectional(LSTM(112, return_sequences=True), input_shape=(None, 136)))
    model.add(BatchNormalization())
    model.add(Dropout(0.27552770755333544))
    model.add(Bidirectional(LSTM(79, return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Dropout(0.15816303501152462))
    model.add(Bidirectional(LSTM(94, return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Dropout(0.15290021625759065))
    model.add(Bidirectional(LSTM(69, return_sequences=False)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2523166286852883))
    model.add(Dense(output_dim=5, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2034598375458345))
    model.add(Dense(output_dim=5, activation='relu'))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


#%%

tune = False

epochs = 30
batch_size = 47
callbacks = [ModelCheckpoint('/specify/path/to_model/here/', monitor='val_loss', save_best_only=True, verbose=1)] # picking model with least val_loss performs better than model with best val_fmeasure; i think fmeasure is computed a bit differently in keras than in sklearn
nn = KerasClassifier(build_fn=rnn_bi_clf_network, nb_epoch=epochs, batch_size=batch_size, verbose=1) # create NN object
if tune == True:
    param_list = []
    acc_list = []
    # tuning parameters and ranges (number of layers, number of nodes per layer, dropout probability, batch size)
    space = {'choice': hp.choice('num_layers',
                [ {'layers':'two', },
                {'layers':'three',
                'units3': hp.uniform('units3', 10,150), 
                'dropout3': hp.uniform('dropout3', .0,.5)},
                {'layers':'four',
                'units3_4': hp.uniform('units3_4', 10,150), 
                'dropout3_4': hp.uniform('dropout3_4', .0,.5),
                'units4': hp.uniform('units4', 10,150),
                'dropout4': hp.uniform('dropout4', .0, .5)
                }
                ]),

        'units1': hp.uniform('units1', 10,150),
        'units2': hp.uniform('units2', 10,150),

        'dropout1': hp.uniform('dropout1', .0,.5),
        'dropout2': hp.uniform('dropout2',  .0,.5),

        'batch_size' : hp.uniform('batch_size', 28,128),

        'nb_epochs' :  epochs, # always using 50 epochs since it has shown that no model needs more than that to converge
        'optimizer': hp.choice('optimizer',['adam']), # adam optimizer
        'activation': 'relu' # always using ReLu activation function
        }        
    
    trials = Trials()
    best = fmin(nn_objective, space, algo=tpe.suggest, max_evals=10, trials=trials)
    print('Best F1 Score: '+str(max(acc_list))) # printing best f1 score over tuning runs
    print('Best Parameters: '+str(param_list[acc_list.index(max(acc_list))])) # printing parameters with which the best f1 score was achieved
    '''
    Optimum Params:
     - layers: 4
     - units: (112, 79, 94, 69)
     - dropout: (0.27552770755333544, 0.15816303501152462, 0.15290021625759065, 0.2523166286852883)
     - batch_size: 124
     - optimizer: adam
     - epochs: 30
     - activation: relu
    '''

else:
    clf = nn.fit(np.array(x_train_dev), np.array(y_train_dev), batch_size=batch_size, nb_epoch=epochs, shuffle=True, verbose=1, validation_data = (np.array(x_dev), np.array(y_dev)), callbacks=callbacks) # fit model to data; includes callbacks to show score on validation set; saves best performing model over epochs
    best_model = load_model('/specify/path/to_best_model/here/') # loads best performing model previously saved
    preds = best_model.predict(np.array(x_dev)) # uses best performing model to make prediction


#%%
# Run Recurrent Neural Network on test set

clf = KerasClassifier(build_fn=rnn_bi_clf_network, nb_epoch=30, batch_size=124, verbose=1)

clf.fit(x_train, y_train)
preds = clf.predict(x_test)

# evaluate on performance metrics
print('Performance: '+str(round(f1_score(np_utils.to_categorical(preds, nb_classes=5), y_test, average='micro')*100, 2))+' % F1 Accuracy')

