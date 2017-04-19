import util
import numpy as np
import pandas as pd

# model_1 = pd.read_csv('fold1_boostdt.csv')
# model_1 = pd.read_csv('small_boostdt.csv')
model_2 = pd.read_csv('~/Desktop/predictions_stiebels/full/predictions_xgboost_fold1.csv', names = ["pred"])
model_1 = pd.read_csv('~/Desktop/predictions_stiebels/full/predictions_nn_fold1.csv', names = ["pred"])
# model_2 = pd.read_csv("~/Desktop/predictions_stiebels/full/predictions_nn_fold1.csv", names = ["pred"])

model_1_label = 'Neural Net'
model_2_label = 'XGBoost'

if (len(model_1) + len(model_2)) == 2400 * 2:
    df_all = pd.read_csv('small_boostdt.csv')
    qid = df_all['qid']
    true = df_all['label']
elif (len(model_1) + len(model_2)) == 4800 * 2:
    df_all = pd.read_csv('large_boostdt.csv')
    qid = df_all['qid']
    true = df_all['label']
elif (len(model_1) + len(model_2)) == 241521 * 2:
    df_all = pd.read_csv('fold1_boostdt.csv')
    qid = df_all['qid']
    true = df_all['label']


# ==============================================================================
def get_ndcg(x_dev_qid, preds, y_dev, k=10, linear=False):
    df_eval = pd.DataFrame(index=[x for x in range(0, len(preds))])
    df_eval['qid'] = x_dev_qid.values
    df_eval['preds'] = preds
    df_eval['truth'] = y_dev.values

    avg_ndcg = []
    for qid in df_eval['qid'].unique():
        this_ndcg = util.ndcg(y_truth=df_eval[df_eval['qid'] == qid]['truth'].values,
                              y_pred=df_eval[df_eval['qid'] == qid]['preds'].values, k=k, linear=linear)
        avg_ndcg.append(this_ndcg)

    return avg_ndcg


def get_map(x_dev_qid, preds, y_dev):
    df_eval = pd.DataFrame(index=[x for x in range(0, len(preds))])
    df_eval['qid'] = pd.Series(x_dev_qid).values
    df_eval['preds'] = preds
    df_eval['truth'] = pd.Series(y_dev).values

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
    return avg_precision




def err(y_truth, y_pred, k=10, max_grade=4):
    sort_by_pred = sorted(list(zip(y_pred, y_truth)), reverse=True)[0:k]
    truth_sorted = np.array([y for _, y in sort_by_pred])

    result = 0.0
    prob_step_down = 1.0

    for rank, rel in enumerate(truth_sorted):
        rank += 1
        utility = (pow(2, rel) - 1) / pow(2, max_grade)
        result += prob_step_down * utility / rank
        prob_step_down *= (1 - utility)

    return result

def get_err(x_dev_qid, preds, y_dev, k=10, linear=False):
    df_eval = pd.DataFrame(index=[x for x in range(0, len(preds))])
    df_eval['qid'] = pd.Series(x_dev_qid).values
    df_eval['preds'] = preds
    df_eval['truth'] = pd.Series(y_dev).values

    avg_err = []
    for qid in df_eval['qid'].unique():
        this_err = err(y_truth=df_eval[df_eval['qid'] == qid]['truth'].values,
                       y_pred=df_eval[df_eval['qid'] == qid]['preds'].values, k=k)
        avg_err.append(this_err)

    return avg_err


# ==============================================================================



significance = pd.DataFrame(index=[x for x in range(0, len(qid))])
significance['qid'] = qid
significance['truth'] = true
significance['predict_1'] = model_1['pred']
significance['predict_2'] = model_2['pred']


map_1_list = []
map_2_list = []
ndcg_1_list = []
ndcg_2_list = []
err_1_list = []
err_2_list = []
for uqid in significance['qid'].unique():
    filter_significance = significance[significance['qid'] == uqid].reset_index(drop=True)
    map_1 = get_map(filter_significance['qid'], filter_significance['predict_1'], filter_significance['truth'])
    map_2 = get_map(filter_significance['qid'], filter_significance['predict_2'], filter_significance['truth'])
    ndcg_1 = get_ndcg(filter_significance['qid'], filter_significance['predict_1'], filter_significance['truth'], k=10)
    ndcg_2 = get_ndcg(filter_significance['qid'], filter_significance['predict_2'], filter_significance['truth'], k=10)
    err_1 = get_err(filter_significance['qid'], filter_significance['predict_1'], filter_significance['truth'], k=10)
    err_2 = get_err(filter_significance['qid'], filter_significance['predict_2'], filter_significance['truth'], k=10)
    map_1_list.append(map_1)
    map_2_list.append(map_2)
    ndcg_1_list.append(ndcg_1)
    ndcg_2_list.append(ndcg_2)
    err_1_list.append(err_1)
    err_2_list.append(err_2)


map_1_values = [item for sublist in map_1_list for item in sublist]
map_2_values = [item for sublist in map_2_list for item in sublist]
ndcg_1_values = [item for sublist in ndcg_1_list for item in sublist]
ndcg_2_values = [item for sublist in ndcg_2_list for item in sublist]
err_1_values = [item for sublist in err_1_list for item in sublist]
err_2_values = [item for sublist in err_2_list for item in sublist]


#==========  Significance Tests =======================
from scipy.stats import ttest_ind, wilcoxon

wnt, wnp = wilcoxon(np.array(ndcg_1_values),np.array(ndcg_2_values))
tnt, tnp = ttest_ind(np.array(ndcg_1_values),np.array(ndcg_2_values))
wmt, wmp = wilcoxon(np.array(map_1_values),np.array(map_2_values))
tmt, tmp = ttest_ind(np.array(map_1_values),np.array(map_2_values))
wet, wep = wilcoxon(np.array(err_1_values),np.array(err_2_values))
tet, tep = ttest_ind(np.array(err_1_values),np.array(err_2_values))


print('\n------------------------------------------ \n')
print('NDCG')
print("wilcoxon :     t = {0:.2f}  p = {1:.4f}".format(wnt, wnp))
print("ttest_ind:     t = {0:.2f}  p = {1:.4f}".format(tnt, tnp))

print('\nERR')
print("wilcoxon :     t = {0:.2f}  p = {1:.4f}".format(wmt, wmp))
print("ttest_ind:     t = {0:.2f}  p = {1:.4f}".format(tmt, tmp))

print('\nMAP')
print("wilcoxon :     t = {0:.2f}  p = {1:.4f}".format(wet, wep))
print("ttest_ind:     t = {0:.2f}  p = {1:.4f}".format(tet, tep))
print('\n------------------------------------------ \n')



#==================  Graphs  ===========================
show_graphs = True



import matplotlib.pyplot as plt
if show_graphs == True:


    # Models Distribution
    f, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2, figsize=(8,8))
    ax1.hist(np.array(ndcg_1_values), bins=10, normed=True)
    ax1.set_title('{} (NDCG)'.format(model_1_label))
    ax2.hist(np.array(ndcg_2_values),bins=10, normed=True)
    ax2.set_title('{} (NDCG)'.format(model_2_label))
    ax3.hist(np.array(err_1_values), bins=10, normed=True)
    ax3.set_title('{} (ERR)'.format(model_1_label))
    ax4.hist(np.array(err_2_values),bins=10, normed=True)
    ax4.set_title('{} (ERR)'.format(model_2_label))
    plt.tight_layout()
    plt.show()


    # Model Difference Scatter
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))
    delta_ndcg = np.array(ndcg_1_values) - np.array(ndcg_2_values)
    ax1.scatter(range(len(delta_ndcg)),delta_ndcg, facecolors='none')
    ax1.plot([0, len(delta_ndcg)], [0, 0], 'k--')
    ax1.set_title('Topic Difference (NDCG)')
    ax1.set_xlim([0,len(delta_ndcg)])
    ax1.set_xlabel('Topic Number')
    ax1.set_ylabel('Difference in NDCG')

    delta_err = np.array(err_1_values) - np.array(err_2_values)
    ax2.scatter(range(len(delta_err)),delta_err, facecolors='none')
    ax2.plot([0, len(delta_err)], [0, 0], 'k--')
    ax2.set_title('Topic Difference (ERR)')
    ax2.set_xlim([0,len(delta_err)])
    ax2.set_xlabel('Topic Number')
    ax2.set_ylabel('Difference in ERR')
    plt.tight_layout()
    plt.show()


    # Model Difference Histogram
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    delta_ndcg = np.array(ndcg_1_values) - np.array(ndcg_2_values)
    ax1.hist(np.array(delta_ndcg),bins=10,normed=True)
    ax1.set_xlabel('Difference in NDCG')

    delta_err = np.array(err_1_values) - np.array(err_2_values)
    ax2.hist(np.array(delta_err), bins=10,normed=True)
    ax2.set_xlabel('Difference in ERR')
    plt.tight_layout()
    plt.show()



