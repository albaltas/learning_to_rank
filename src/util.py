import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def make_sample(df, num_q=50): # 1 query (qid) ~ 120 rows
    '''
    @ Simon Stiebellehner
    draws sample of complete queries
    '''
    # sampling to make coding/trying out things lighter
    # ensures to get whole q/d pairs in sample
    qids = np.random.choice(df['qid'].unique(), size=num_q)
    return df[df['qid'].isin(qids)].copy()


def sep_feat_labels(df):
    '''
    @ Simon Stiebellehner
    separates features from labels
    '''
    # returns (features, labels)
    return df.ix[:,1:].copy(), df.ix[:,0].copy()


def get_ndcg(x_dev_qid, preds, y_dev, k=10, linear=False):
    '''
    @ Simon Stiebellehner
    
    Arguments:
        - x_dev_qid: array of qid of dev set
        - preds: predictions in same order as x_dev_qid
        - y_dev: truth in same order as x_dev_qid
    Return:
        - average ndcg over queries
    '''
    
    df_eval = pd.DataFrame(index = [x for x in range(0,len(preds))])
    df_eval['qid'] = x_dev_qid.values
    df_eval['preds'] = preds
    df_eval['truth'] = y_dev.values
    
    avg_ndcg = []
    for qid in df_eval['qid'].unique():
        this_ndcg = ndcg(y_truth = df_eval[df_eval['qid']==qid]['truth'].values, y_pred = df_eval[df_eval['qid']==qid]['preds'].values, k=k, linear=linear)
        avg_ndcg.append(this_ndcg)
    return np.mean(avg_ndcg)


######### RANKING METRICS #########

def rmse_func(y_pred, y_truth):
    '''
    @ Simon Stiebellehner
    Computes RMSE score
    
        Arguments:
        - y_truth: ground truth
        - y_pred: predictions
    Return:
        - RMSE
    '''
    return abs(mean_squared_error(y_truth, y_pred))**0.5



def dcg(y_truth, y_pred, k=10, linear=False, epsilon=0.00001):
    """
    @ Simon Stiebellehner    
    
    Arguments:
        - y_truth: ground truth
        - y_pred: predictions
        - k: rank until dcg should be computed
        - linear: True if linear computation should be performed
        - epsilon: adding small constant to avoid issues with division with/by 0 when handling 0-relevance-labels
    Return:
        - DCG score
    """
    
    truth_sorted = np.array([y for _,y in sorted(list(zip(y_pred, y_truth)), reverse=True)[0:k]])
    
    if linear:
        gain = truth_sorted + epsilon
    else:
        gain = (2**truth_sorted)-1 + epsilon
           
    discount = np.log2(np.array([x for x in range(2,len(truth_sorted)+1+1)])) + epsilon
    
    return np.sum(np.divide(gain,discount))



def ndcg(y_truth, y_pred, k=10, linear=False, epsilon=0.00001):
    '''
    @ Simon Stiebellehner

    Arguments:
        - y_truth: ground truth
        - y_pred: predictions
        - k: rank until ndcg should be computed
        - linear: True if linear computation should be performed
        - epsilon: adding small constant to avoid issues with division with/by 0 when handling 0-relevance-labels
    Return:
        - NDCG score
    '''
    
    return np.divide(dcg(y_truth, y_pred, k, linear, epsilon), dcg(y_truth, y_truth, k, linear, epsilon))


def get_mapk(x_dev_qid, preds, y_dev):
    '''
    @ Dimitri Visnadi

    Arguments:
        - x_dev_qid: array of qid of dev set
        - preds: predictions in same order as x_dev_qid
        - y_dev: truth in same order as x_dev_qid
        - k: number of predicted values

    Return:
        - MAPk score
    '''

    # Check if all 4 classes are predicted. Weak models predict only 0 and 1.
    # predicts_all_classes = list(preds).count(4)
    # if predicts_all_classes == 0:
    #     return "WEAK MODEL: Not all classes predicted"

    df_eval = pd.DataFrame(index=[x for x in range(0, len(preds))])
    df_eval['qid'] = pd.Series(x_dev_qid).values
    df_eval['preds'] = preds
    df_eval['truth'] = pd.Series(y_dev).values

    # 0 relevance = irrelevant
    # everything else is somewhat relevant (1) to perfectly relevant (4)

    df_eval.ix[df_eval.preds <= 0, 'preds'] = 0
    df_eval.ix[df_eval.preds > 0, 'preds'] = 1

    df_eval.ix[df_eval.truth <= 0, 'truth'] = 0
    df_eval.ix[df_eval.truth > 0, 'truth'] = 1

    avg_precision = []
    for qid in df_eval['qid'].unique():
        pred = [x for x in tuple(df_eval[df_eval['qid'] == qid]['preds'].values)]
        true = [x for x in tuple(df_eval[df_eval['qid'] == qid]['truth'].values)]

        precision = []
        right = 0
        for pos, p in enumerate(pred):
            if p == true[pos]:
                right += 1
                precision.append(right / (pos + 1))
        try:
            avg_precision.append(sum(precision) / len(precision))
        except:
            avg_precision.append(0)

    map_score = sum(avg_precision) / len(avg_precision)

    return map_score



def getConfusionMatrix(gold_labels, predicted_labels):
    '''
    @ Alexandros Baltas

    Arguments:
        - gold_labels: list of actual class values of the instances
        - pred: mode predictions

    Returns:
        - A square np.array which represents the confusion matrix of the classification.

    '''

    number_of_classes = len(np.unique(gold_labels))
    confusion_matrix = np.zeros([number_of_classes, number_of_classes])

    for i in range(len(gold_labels)):
        confusion_matrix[predicted_labels[i]][gold_labels[i]] +=1

    return confusion_matrix

def precision_recall_f1(gold_labels, predicted_labels, average='micro'):
    '''
    @ Alexandros Baltas

    Arguments:
        - gold_labels: list of actual class values of the instances
        - pred: mode predictions
        - average: Determines the type of averaging on the data.
            * 'weighted' calculates the metrics for each class, and then computes the
              weighted average, using the number of instances for each class.
            * 'macro' calculates the metrics for each class, and then computes the mean

    Returns:
        - A list that contains the values of Precision, Recall & F1-Score.

    '''

    confusion_matrix = getConfusionMatrix(gold_labels, predicted_labels)
    number_of_classes = len(np.unique(gold_labels))

    class_precision = np.zeros(number_of_classes)
    class_recall = np.zeros(number_of_classes)
    class_f1 = np.zeros(number_of_classes)

    if average == "micro":
        diagonal = sum(np.diagonal(confusion_matrix))
        denom = sum(sum(confusion_matrix))

        return diagonal/denom, diagonal/denom, diagonal/denom

    else:

        for i in range(number_of_classes):
            if np.sum(confusion_matrix[i, :]) != 0 :
                class_precision[i] = confusion_matrix[i][i] / np.sum(confusion_matrix[i, :])
            else:
                class_precision[i] = 0
            if np.sum(confusion_matrix[:, i]) != 0:
                class_recall[i] = confusion_matrix[i][i] / np.sum(confusion_matrix[:, i])
            else:
                class_recall[i] = 0

            if (class_precision[i]+class_recall[i]) != 0:
                class_f1[i] = 2*class_precision[i]*class_recall[i]/(class_precision[i]+class_recall[i])
            else:
                class_f1[i] = 0

        class_support = [np.count_nonzero(gold_labels == x) for x in np.unique(gold_labels)]
        if average == "weighted":
            return np.average(class_precision, weights=class_support), np.average(class_recall, weights=class_support), np.average(class_f1, weights=class_support)
        elif average == "macro":
            return np.mean(class_precision), np.mean(class_recall), np.mean(class_f1)






def err(y_truth, y_pred, k=10):

    current_df = pd.DataFrame(index=range(0,len(y_truth)))
    current_df["y_pred"] = y_pred
    current_df["y_truth"] = y_truth
    current_df.sort_values(["y_pred","y_truth"],ascending=False,inplace=True)
    current_df.reset_index(inplace=True,drop=["index"])
    current_df = current_df.ix[0:k-1,:]
    
    result = 0.0
    decay = 1.0    
    for index in current_df.index:
        gain = (2**(current_df.ix[index,"y_truth"]) - 1) / 16
        result += (gain*decay) / (index+1)
        decay *= (1 - gain)

    return result

def get_err(x_dev_qid, preds, y_dev, k=10):


    '''
    @ Andreas Gompos

    Arguments:
        - x_dev_qid: array of qid of dev set
        - preds: predictions in same order as x_dev_qid
        - y_dev: truth in same order as x_dev_qid
        - k: depth of evaluation

    Return:
        - ERR@k score
    '''




    current_df = pd.DataFrame(index=range(0,len(preds)))
    current_df['qid'] = x_dev_qid
    current_df['preds'] = preds
    current_df['truth'] = y_dev

    all_errs = []
    for qid in current_df.qid.unique():
        

        current_err = err(current_df[current_df['qid'] == qid]['truth'].values,\
                              current_df[current_df['qid'] == qid]['preds'].values,\
                              k=k)
        all_errs.append(current_err)

        
    return np.mean(all_errs)
