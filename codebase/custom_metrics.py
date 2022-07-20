import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def multilabel_roc_auc(y_true, y_score, average='macro'):
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values

    if (not average) or (average == 'macro'):
        aucs = np.empty(shape=(len(y_score),), dtype=np.float64)
        
        for i in range(0, len(y_score)):
            y_score_i = y_score[i]
            y_score_i = y_score_i[:,1]
            y_true_i = y_true[:,i]
            aucs[i] = roc_auc_score(y_true_i, y_score_i)

        if average == 'macro':
            res = np.mean(aucs)
        else:
            res = aucs

    elif average == 'micro':
        y_score_micro = []
        y_true_micro = []

        for i in range(0, len(y_score)):
            y_score_i = y_score[i]
            y_score_i = y_score_i[:,1]
            y_true_i = y_true[:,i]
            y_score_micro.append(y_score_i)
            y_true_micro.append(y_true_i)
        
        y_score_micro = np.asarray(y_score_micro).flatten()
        y_true_micro = np.asarray(y_true_micro).flatten()
        res = roc_auc_score(y_true_micro, y_score_micro)
    else:
        raise NotImplementedError("average must be 'macro', 'micro', or None")

    return res
