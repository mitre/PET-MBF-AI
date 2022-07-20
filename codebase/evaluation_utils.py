import sys
import os
BASE_PATH = os.getcwd().split('e-emagin-pet')[0] + 'e-emagin-pet/'
sys.path.append(BASE_PATH)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from codebase.custom_metrics import multilabel_roc_auc

from sklearn.metrics import precision_score, recall_score, accuracy_score, balanced_accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential

def false_positive_rate(y_true, y_pred, pos_label=1):
    false_positives = (y_true != pos_label) & (y_pred == pos_label)
    fpr = false_positives.sum() / (y_true != pos_label).sum()
    return fpr

def calculate_auc(model, X_test, y_test, auc_method):
    if auc_method == None:
        raise ValueError('Must specify auc_method to calculate auc')
    if auc_method == 'prob':
        if type(model) == Sequential:
            scores = model.predict_proba(X_test)[:,0]
        else:
            scores = model.predict_proba(X_test)[:,1]
    elif auc_method == 'thresh':
        scores = model.decision_function(X_test)
    auc = roc_auc_score(y_test, scores)
    return auc

def plot_precision_recall_curve(y_true, scores, thresh, pos_label=1, title=None, ax=None, label=None, filepath=None):
    """
    Plot precision-recall curve. Only works with scores as probabilities. Does
    Not take decision thresholds like SVM does.

    Parameters
    ----------
    y_true : pd.Series
        True labels for test data
    scores : pd.Series
        Probability estimates
    thresh : float
        The threshold used to assign class predicitons from output of decision
        function. This is a probability if scores are probability estimates,
        and this is signed distance from the decision plane for models like SVM
        where decision function is not probabilistic.
    pos_label : int or string
        Label for positive class in y_true
    title : string
        Optional title of plot
    ax : matplotlib axis object
        Optional axis to plot ROC curve on. Use this to plot multiple ROC
        curves on the same graph
    label : string
        Optional label for the ROC curve. Use this when plotting multiple ROC
        curves on the same graph
    filepath : string
        Optional filepath to save image

    """
    precision_scores, recall_scores, thresholds = precision_recall_curve(y_true, scores, pos_label=pos_label)
    if not ax:
        fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(recall_scores, precision_scores, label=label)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")

    # get point on curve closest to thresh
    min_idx = np.abs(thresholds - thresh).argsort()[0]
    precision = precision_scores[min_idx]
    recall = recall_scores[min_idx]
    ax.scatter(recall, precision)

    if title:
        ax.set_title(title)
    if label:
        plt.legend()
    if filepath:
        plt.tight_layout()
        plt.savefig(filepath, dpi=300)

def plot_ROC(y_true, scores, thresh, pos_label=1, title=None, ax=None, label=None, filepath=None, color=None, label_size=None, x_label=True, y_label=True):
    """
    Plot ROC curve.

    Parameters
    ----------
    y_true : pd.Series
        True labels for test data
    scores : pd.Series
        Either probability estimates or decision function evaluations for test
        data
    thresh : float
        The threshold used to assign class predicitons from output of decision
        function. This is a probability if scores are probability estimates,
        and this is signed distance from the decision plane for models like SVM
        where decision function is not probabilistic.
    pos_label : int or string
        Label for positive class in y_true
    title : string
        Optional title of plot
    ax : matplotlib axis object
        Optional axis to plot ROC curve on. Use this to plot multiple ROC
        curves on the same graph
    label : string
        Optional label for the ROC curve. Use this when plotting multiple ROC
        curves on the same graph
    filepath : string
        Optional filepath to save image
    color : string
        Optional color to use for line in plot. A valid matplotlib color string.
    label_size : int
        Optional size for axis titles and labels

    """
    fprs, tprs, thresholds = roc_curve(y_true, scores, pos_label=pos_label)

    # get point on curve closest to thresh
    min_idx = np.abs(thresholds - thresh).argsort()[0]
    tpr = tprs[min_idx]
    fpr = fprs[min_idx]

    if not ax:
        fig, ax = plt.subplots(figsize=(5,5))

    if color:
        ax.plot(fprs, tprs, label=label, color=color)
        ax.scatter(fpr, tpr, color=color)
    else:
        ax.plot(fprs, tprs, label=label)
        ax.scatter(fpr, tpr)


    if label_size:
        if x_label:
            ax.set_xlabel("Specificity", size=label_size)
        if y_label:
            ax.set_ylabel("Sensitivity", size=label_size)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.set_xticklabels([1, .5, 0.], size=label_size)
        ax.set_yticklabels([0.0, .5, 1.0], size=label_size)
    else:
        if x_label:
            ax.set_xlabel("Specificity")
        if y_label:
            ax.set_ylabel("Sensitivity")
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xticklabels([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])

    if title:
        ax.set_title(title, size=label_size + 1)
    if label:
        plt.legend()
    if filepath:
        plt.tight_layout(pad=0.5)
        plt.savefig(filepath, dpi=300)


def decision_boundry_youdens(y_true, scores):
    """
    For binary classification problem, find the decision threshold
    that maximizes Youden's index

    y_true : np.array of ints
        True labels

    scores : np.array of floats
        Predicted probabilities of class 1

    Returns:
    --------
    The decision threshold that maximizes youden's index

    """

    _, _, thresholds = roc_curve(y_true, scores, pos_label=1)
    youdens = np.empty_like(thresholds)
    preds = np.empty_like(scores)
    for thresh_i, thresh in enumerate(thresholds):
        preds = scores.copy()
        preds[scores > thresh] = 1
        preds[scores <= thresh] = 0
        youdens[thresh_i] = balanced_accuracy_score(y_true, preds, adjusted=True)

    return thresholds[youdens.argmax()]

def decision_boundry_fixed_recall(y_true, scores, target=0.9):
    """
    For binary classification problem, find the decision threshold
    that maximizes Youden's index

    y_true : np.array of ints
        True labels

    scores : np.array of floats
        Predicted probabilities of class 1

    target : np.float
        Target recall value

    Returns:
    --------
    The decision threshold for the model at target recall value

    """

    _, _, thresholds = roc_curve(y_true, scores, pos_label=1)
    recalls = np.empty_like(thresholds)
    preds = np.empty_like(scores)
    for thresh_i, thresh in enumerate(thresholds):
        preds = scores.copy()
        preds[scores > thresh] = 1
        preds[scores <= thresh] = 0
        recalls[thresh_i] = recall_score(y_true, preds)

    dist_to_targ = np.abs(recalls - target)

    return thresholds[dist_to_targ.argmin()]

def decision_boundry_fixed_specificity(y_true, scores, target=0.1):
    """
    For binary classification problem, find the decision threshold
    that maximizes Youden's index

    y_true : np.array of ints
        True labels

    scores : np.array of floats
        Predicted probabilities of class 1

    target : np.float
        Target recall value

    Returns:
    --------
    The decision threshold for the model at target recall value

    """

    _, _, thresholds = roc_curve(y_true, scores, pos_label=1)
    specificities = np.empty_like(thresholds)
    preds = np.empty_like(scores)
    for thresh_i, thresh in enumerate(thresholds):
        preds = scores.copy()
        preds[scores > thresh] = 1
        preds[scores <= thresh] = 0
        specificities[thresh_i] = recall_score(y_true, preds, pos_label=0)

    dist_to_targ = np.abs(specificities - target)
    return thresholds[dist_to_targ.argmin()]


def confusion_matrix(y_true, y_pred, scores=np.array([None]), pos_label=1, neg_label=0, title=None, filepath=None, print_scores=True):
    """
    Plot confusion matrix and print precision, recall, accuracy, specificity.
    For binary problem, if labels not 0 or 1 specify pos and negative labels.
    Currently only tested for binary case

    Parameters
    ----------
    y_true : pd.Series
        True labels
    y_pred : pd.Series
        Predicted labels
    scores : pd.Series
        Option to pass confidence metrics for prediction -- i.e. probabilities
        or non-thresholded decision values (as returned by decision_function
        on some sklearn classifiers).
    pos_label : int or string
        label to indicate positive class
    neg_label : int or string
        label to indicate negative class
    filepath : string
        Optional filepath to save image

    Returns
    -------
    None
    """
    confusion_matrix = pd.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted']).T
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix, annot=True, vmin=0, vmax=1000, fmt='d')
    ax.set_ylim([0,2])
    if title:
        ax.set_title(title)
    if filepath:
        plt.tight_layout()
        plt.savefig(filepath, dpi=300)


    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=pos_label)
    recall = recall_score(y_true, y_pred, pos_label=pos_label)
    f1 = f1_score(y_true, y_pred, pos_label=pos_label)
    specificity = recall_score(y_true, y_pred, pos_label=neg_label)
    if scores.any():
        auc = roc_auc_score(y_true, scores)
    if print_scores:
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1: {f1}")
        print(f"Specificity: {specificity}")
        if scores.any():
            print(f"AUC: {auc}")


def evaluate_model(y_true,
                   y_pred,
                   scores,
                   problem,
                   average=None,
                   print_scores=True,
                   filepath=None,
                   pos_label=1,
                   neg_label=0,
                   outcome_names=None):
    """
        y_true must be pd.DataFrame with column names corresponding to
        outcome_names if problem is localization.
    """
    if problem == 'norm_abn':
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, pos_label=pos_label)
        recall = recall_score(y_true, y_pred, pos_label=pos_label)
        f1 = f1_score(y_true, y_pred, pos_label=pos_label)
        specificity = recall_score(y_true, y_pred, pos_label=neg_label)
        auc = roc_auc_score(y_true, scores)
        results_dict = {'accuracy': accuracy, 'precision': precision, 'recall': recall,
                        'specificity': specificity, 'f1': f1, 'auc': auc}
    elif problem == 'localization':
        if not outcome_names:
            raise ValueError("outcome_names required if problem is localization")

        # start with just AUC
        # You will have to figure out spot on AUC curve for all other metrics
        cols_msk = y_true.columns.isin(outcome_names)
        y_true = y_true.loc[:, cols_msk]
        y_pred = y_pred[:, cols_msk]
        scores = list(np.asarray(scores)[cols_msk])
        auc = multilabel_roc_auc(y_true, scores, average)

        if average is None:
            auc = dict(zip(outcome_names, list(auc)))

        results_dict = {'auc': auc}

    if print_scores:
        for metric in results_dict.keys():
            print(f"{metric}: {results_dict[metric]}")

    return results_dict

def logistic_regression_from_weights(weights, bias, penalty='none', C=0.0):
    """
        Build sklearn logistic regression with hardcoded weights, bias, and
        regularization coefficient.

        Parameters
        ----------

        weights : list of floats
            The model weights

        bias : float
            The bias/intercept term

        penalty : string
            The type of penalty term. Valid options include: 'l1', 'l2',
            'elasticnet', 'none'. See sklearn.linear_model.LogisticRegression
            documentation for more info.

        regularization_coef : float
            The inverse of the regularization strength
    """
    weights = np.array(weights).reshape(1, -1)

    lr = LogisticRegression(C=C, penalty=penalty)
    lr.classes_ = np.array([0, 1])
    lr.coef_ = weights
    lr.intercept_ = bias
    return lr
