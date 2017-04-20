# Information Retrieval and Data Mining
This repository contains the source code of UCL's COMPGI15: Information Retrieval and Data Mining Group Assignment (2016/2017).

### Group 6
* [Simon Stiebellehner](https://github.com/stiebels)
* [Dimitri Visnadi](https://www.linkedin.com/in/visnadi)
* Andreas Gompos
* [Alexandros Baltas](https://www.linkedin.com/in/albaltas/)


### Requirements
This project developed using Python 3.5. A list of Software dependencies follows:

* [RankLib](https://sourceforge.net/p/lemur/wiki/RankLib/)
* [scikit-learn](http://scikit-learn.org/stable/)
* [SciPy](https://www.scipy.org/)
* [TensorFlow](https://www.tensorflow.org/)
* [Keras](https://keras.io/)
* [XGBoost](https://github.com/dmlc/xgboost)
* [Hyperopt](https://github.com/hyperopt/hyperopt)
* [Scikit-optimize](https://scikit-optimize.github.io/)

### Dataset
The dataset that was used for the project is called [MSLR-WEB10K](https://www.microsoft.com/en-us/research/project/mslr/). This work contains scripts that tune ranking algorithms implemented in RankLib, and various custom models. The dataset can be used as-is for the RankLib implementations, whereas custom models require a csv format. In order to transform the data from txt to the required csv format, the script located in src/pre_processing/cleaning_script.py can be used.

### Models
This repository contains various custom models for the ranking task. They can be found in the src/models directory. The different models included in the directory are the following:

* adaboost:
Custom implementation of AdaBoost.
* ensemble:
Custom implementaion of an Ensemble Model.
* lambaMART:
Tuning script of RankLib's LamdaMART.
* rankNet:
Tuning script of RankLib's RankNet.
* ranking_svm:	
Custom implementation of a Ranking SVM.
* rnn:
Custom implementaion of an RNN classifier.
* svm:
Custom implementaion of an SVM classifier
* tensorflow_logistic_regression:
Custom implementaion of a Logistic Regression classifier.
* xgboost_&_neural_network:
Custom implementations of an XGBoost powered classifier and a deep neural network classifier.

### Evaluation metrics
Implementations of various evaluation metrics can be found in src/util.py. This file contains the following metrics:
* DSG@K
* NDSG@K
* ERR@K
* MAP
* Precision
* Recall
* F1-Score

### Miscellaneous
A statistical comparison of the different Folds of the dataset can be found in src/pre-processing/dataset_comparison.ipynb. A significance test of our results can be found in src/post-processing/significance_test.py. A script that collected our tuning results, and produce csv reports can be found in src/post-processing/results_crawler.py.

