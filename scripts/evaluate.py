"""
    A command line script to compare and evaluate the performance of different
    models. 
    
    Positional arguments consist of paths to the models to compare, and each 
    must be saved as instances of the ModelEvalWrapper class. 

    Additional required arguments include:
    --outdir    : The directory where results will be saved
    --datadir   : The path to the base directory containing the data
    --problem   : The classification problem that the models are trained to
                  predict. Valid options include 'norm_abn' and 'localization'

    Additional optional arguments can be used to specify which evaluations to
    include. 
    --perc_conf_interval        :   The percent confidence interval to
                                    calculate for AUC. Default is 95. 
    --auc_ci                    :   If this flag is present, the program will
                                    save a bar chart and csv comparing auc
                                    values with confidence intervals for all
                                    models. 
    --confusion_matrix          :   If this flag is present, the program will
                                    save confusion matricies for all models.
    --compare_performance       :   If this flag is present, the program will
                                    generate and save csv and bar plots that
                                    compare performance of all models including
                                    AUC, accuracy, precision, recall, F1, 
                                    specificity
    --roc                       :   If this flag is present, the program will
                                    plot ROC curves for all models.
    --delong                    :   If this flag is true, the program will 
                                    compute DeLong's test to compare roc curves
                                    for each pair of models and save output in
                                    a csv.
    --all                       :   If this flag is present, the program will
                                    run all evaluations.
    --final                     :   Run final model on held out test set for
                                    generalization performance.

"""

from __future__ import print_function
import sys
import os
BASE_PATH = os.getcwd().split('e-emagin-pet-export1')[0] + 'e-emagin-pet-export1/'
sys.path.append(BASE_PATH)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score 
from sklearn.metrics import precision_score, recall_score, f1_score

from codebase import data_utils
from codebase import evaluation_utils as eu
from codebase.ModelEvalWrapper import ModelEvalWrapper
from codebase.Ensemble import Ensemble

from joblib import load
import sys
import argparse
from rpy2.robjects import r, pandas2ri
from rpy2.robjects.packages import importr



def compare_conf_matricies(models, X_test, y_test, results_dir):
    """
    Display / save confusion matricies for all models on models

    Parameters
    ----------
    models : python list of ModelEvalWrapper objects
        The models to evaluate

    X_test : Pandas dataframe
        The test set features

    y_test : Pandas series or numpy array
        The labels for the test set

    results_dir : string
        The directory where the resulting confusion matricies will be saved

    Returns
    -------
    None
    """
    for model in models:
        X_test_m = X_test[model.dataset] 
        y_pred = model.predict(X_test_m)

        scores = model.predict_proba(X_test_m)[:,1]

        eu.confusion_matrix(y_test.values, y_pred, scores=scores, print_scores=False,
                            title=f"{model.model_name}", 
                            filepath=f"{results_dir}/{model.model_name_abrv}_confusion_matrix.png")

def plot_metric(metric, title=None, ax=None):
    """
    Plots bar chart comparison of single metric, +/- 1 std dev
    
    
    Parameters:
    -----------
    
    metric : pd.DataFrame
        Avg dataframe returned by avg_res()
        
    stds : pd.DataFrame
        Std dataframe returned by avg_res()
        
    title : string
        Optional title
        
    ax : matplotlib axis
        Optional axis to plot on
        
    Return: None
    -------
    """
    
    if not ax:
        fig, ax = plt.subplots()
        
    color_dict = {'LR': 'C4',
                  'Lasso LR': 'C3',
                  'MLP': 'C1',
                  'UNET': 'C2',
                  'RF': 'C0',
                  'SVM': 'C5',
                  'Dummy': 'C7'}

    model_names = metric.index.values
    max_len = max([len(name) for name in model_names])
    model_names_centered = [name.center(max_len, ' ') for name in model_names]
    colors = [color_dict[model_name] for model_name in model_names]    

    ax.bar(model_names_centered, metric, color=colors)
    
    for label in ax.get_xticklabels():
        label.set_rotation(45)
    if title:
        ax.set_title(title)
        
def plot_barchart_comp(results, problem, suptitle, outcome=None, vessel=None, filepath=None):
    """
    Plot bar chart comparison of all metrics
        - This is not designed very well -- the metric dataframes from
          avg_res are not passed as arguments but accessed as global variables.
    
    
    Parameters:
    -----------
    outcome : string
        The outcome to compare model performance for
    
    vessel : string
        The vessel territory to compare model performance for
    
    suptitle : string
        The title for the plot
    
    filepath : string
        Optional filepath to save png of plot
        
    Returns: None
    --------
    
    """
    
    fig, axs = plt.subplots(1, 5, figsize=(12, 3.5), sharey=True)
    metrics = ['auc', 'accuracy', 'precision', 'recall', 'specificity']
    metrics_printable = ['AUC', 'Accuracy', 'Precision / PPV', 'Recall / Sensitivity',
                        'Specificity']

    if problem == 'localization':
        for i, (metric, metric_printable) in enumerate(zip(metrics, metrics_printable)):
            plot_metric(results[metric].loc[:,(outcome, vessel)],
                        metric_printable,
                        ax=axs[i])

        all_metrics = [results[metric].loc[:,(outcome, vessel)] for metric in metrics]
    elif problem == 'norm_abn':
        for i, (metric, metric_printable) in enumerate(zip(metrics, metrics_printable)):
            plot_metric(results[metric],
                        metric_printable,
                        ax=axs[i])

        all_metrics = [results[metric] for metric in metrics]
    
    
    all_metrics = np.hstack(all_metrics)
    axs[4].set_ylim(bottom=all_metrics.min() - 0.1)
    plt.suptitle(suptitle)
    fig.tight_layout()
    if filepath:
        fig.savefig(filepath, dpi=300)
    plt.plot()

def compare_model_performance_norm_abn(models, X_test, y_test, results_dir):
    """
    Compares performance of models, recording accuracy, precision, recall,
    specificity, f1, auc. Plots bar chart, saves comparison and bar chart in 
    results_dir

    Parameters
    ----------
    models : python list of ModelEvalWrapper objects
        The models to evaluate

    X_test : Pandas dataframe
        The test set features

    y_test : Pandas series or numpy array
        The labels for the test set

    results_dir : string
        The directory where the resulting confusion matricies will be saved

    Returns
    -------
    None
    """
    accuracies = [] 
    precisions = []
    recalls = []
    specificities = []
    f1s = []
    aucs = []
    model_names = []

    results_lists = [accuracies, precisions, recalls, specificities,
                     f1s, aucs]
    results_keys = ['accuracy', 'precision', 'recall', 'specificity',
                    'f1', 'auc']

    for model in models:
        X_test_m = X_test[model.dataset] 

        scores = model.predict_proba(X_test_m)[:,1]
        y_pred = model.predict(X_test_m)
        res = eu.evaluate_model(y_test, y_pred, scores=scores, problem='norm_abn', print_scores=False)
        model_names.append(model.model_name_abrv)
        for res_list, res_key in zip(results_lists, results_keys):
            res_list.append(res[res_key])

    results_dict = dict(zip(results_keys, results_lists))
    res_df = pd.DataFrame.from_dict(results_dict)
    res_df.index = model_names
    res_df.to_csv(f"{results_dir}/model_performance.csv")

    fig, axes = plt.subplots(2, 3, figsize=(9, 6), sharey=True)
    for ax, col in zip(axes.flatten(), res_df.columns):
        sns.barplot(x=res_df.index, y=res_df[col], ax=ax)
        ax.set_title(col)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/performance_barcharts.png", dpi=300)

    plot_barchart_comp(res_df,
                       problem='norm_abn',
                       suptitle='Per Person Normal/Abnormal Detection',
                       filepath=f"{results_dir}/barchart_comp.png")


def compare_model_performance_loc(models, X_test, y_test, results_dir):
    """
    Compares performance of models for localization problem. Saves two CSVs,
    one for AUC values and the other for accuracy. Each CSV has results by
    outcome (abnormal, ischemia, scar) and by region or per vessel (per vessel,
    lcx, rca, lad)
    

    Parameters
    ----------
    models : python list of ModelEvalWrapper objects
        The models to evaluate

    X_test : Pandas dataframe
        The test set features

    y_test : Pandas series or numpy array
        The labels for the test set

    results_dir : string
        The directory where the resulting confusion matricies will be saved

    Returns
    -------
    None
    """
    model_names = []
    # make index
    for model in models:
        model_names.append(model.model_name_abrv)

    outcomes = ['abnormal', 'scar', 'ischemia']
    vessels = ['per_vessel', 'lad', 'rca', 'lcx']
    num_models = len(models)

    multi_idx = pd.MultiIndex.from_product([outcomes, vessels])


    accuracies = pd.DataFrame(np.full((num_models, 12), -1),
                              index=model_names,
                              columns=multi_idx)
    aucs = pd.DataFrame(np.full((num_models, 12), -1),
                        index=model_names,
                        columns=multi_idx)
    precisions = pd.DataFrame(np.full((num_models, 12), -1),
                              index=model_names,
                              columns=multi_idx)
    recalls = pd.DataFrame(np.full((num_models, 12), -1),
                           index=model_names,
                           columns=multi_idx)
    f1s = pd.DataFrame(np.full((num_models, 12), -1),
                       index=model_names,
                       columns=multi_idx)
    specificities = pd.DataFrame(np.full((num_models, 12), -1),
                                 index=model_names,
                                 columns=multi_idx)
    print("Compare Model Performance", file=sys.stderr)
    for model in models:
        X_test_m = X_test[model.dataset]
        for outcome in outcomes:
            for vessel in vessels:
                print('model: \t', model.model_name_abrv,
                      'outcome: \t', outcome, 
                      'vessel: \t', vessel,
                      file=sys.stderr)
                if vessel == 'per_vessel':
                    vessel_i = None
                else:
                    vessel_i = vessel
                
                preds, _ =  preds_per_vessel(model,
                                             X_test_m,
                                             y_test,
                                             outcome,
                                             vessel=vessel_i,
                                             probs=False)
    
                probs, labels = preds_per_vessel(model,
                                                 X_test_m,
                                                 y_test,
                                                 outcome,
                                                 vessel=vessel_i,
                                                 probs=True)

                # specificity is recall with pos label flipped
                accuracy = accuracy_score(labels, preds)
                auc = roc_auc_score(labels, probs) 
                precision = precision_score(labels, preds)
                recall = recall_score(labels, preds, average='binary', pos_label=1)
                f1 = f1_score(labels, preds)
                specificity = recall_score(labels, preds, average='binary', pos_label=0)

                accuracies.loc[model.model_name_abrv,
                               (outcome, vessel)] = accuracy
                aucs.loc[model.model_name_abrv,
                         (outcome, vessel)] = auc
                precisions.loc[model.model_name_abrv,
                               (outcome, vessel)] = precision
                recalls.loc[model.model_name_abrv,
                            (outcome, vessel)] = recall
                f1s.loc[model.model_name_abrv,
                        (outcome, vessel)] = f1
                specificities.loc[model.model_name_abrv,
                                  (outcome, vessel)] = specificity
                

    accuracies.to_csv(f"{results_dir}/accuracy.csv")
    aucs.to_csv(f"{results_dir}/auc.csv")
    precisions.to_csv(f"{results_dir}/precision.csv")
    recalls.to_csv(f"{results_dir}/recall.csv")
    f1s.to_csv(f"{results_dir}/f1.csv")
    specificities.to_csv(f"{results_dir}/specificity.csv")

    results = {}
    results['auc'] = aucs
    results['accuracy'] = accuracies
    results['precision'] = precisions
    results['recall'] = recalls
    results['specificity'] = specificities

    plot_barchart_comp(results,
                       problem='localization',
                       suptitle='Per Vessel Normal/Abnormal Classification',
                       outcome='abnormal',
                       vessel='per_vessel',
                       filepath=f"{results_dir}/barchart_comp.png")



def compare_model_performance(models, X_test, y_test, results_dir, problem):
    print("\nComparing model performance\n")
    if problem == 'norm_abn':
        compare_model_performance_norm_abn(models, X_test, y_test, results_dir)
    elif problem == 'localization':
        compare_model_performance_loc(models, X_test, y_test, results_dir)
    else:
        raise ValueError("problem must be 'norm_abn' or 'localization'")



def preds_per_vessel(model, X, y, outcome, vessel=None, probs=False):
    """
    outcome must be scar, ischemia, or abnormal
    

    Parameters:
    -----------

    model : ModelEvalWrapper object
    
    X : pd.DataFrame
        The features

    y : pd.Series or np.array
        The labels

    outcome : string
        Must be 'abnormal', 'scar', or 'ischemia'

    vessel : string    
        Must be lcx, lad, rca, or None. If None, then return per vessel
        predicted probabilities with no distinction by region. 

    probs : bool
        If True, return predicted probabilities. Otherwise, return predictions.

    Return:
    -------
    (preds, labels)
    """
    col_names = list(y.columns) 
    cols = []
    for col in col_names:
        # col is in format {abn_type}_{vessel}
        if (((vessel == col.split('_')[1]) | (not vessel)) & 
           ((outcome == col.split('_')[0]) | (outcome == 'abnormal'))):
            cols.append(col)

    preds = []
    labels = []
    for i, col in enumerate(cols):
        if probs:
            preds.append(model.predict_proba(X)[col_names.index(col)][:,1])
        else:
            preds.append(model.predict(X)[:,col_names.index(col)]) 

        labels.append(y[col])
    
    preds = np.array(preds).flatten()
    labels = np.array(labels).flatten()
    return preds, labels

def compare_roc_curves(models, X_test, y_test, results_dir, problem):
    """
    Plot roc curves for each model in models and save plot to results_dir

    Parameters
    ----------
    models : python list of ModelEvalWrapper objects
        The models to evaluate

    X_test : Pandas dataframe
        The test set features

    y_test : Pandas series or numpy array
        The labels for the test set

    results_dir : string
        The directory where the resulting confusion matricies will be saved

    Returns
    -------
    None
    """
    models = models.copy()
    models = [m for m in models if m.model_name_abrv != 'Dummy']
    color_dict = {'LR': 'C4',
                  'Lasso LR': 'C3',
                  'MLP': 'C1',
                  'UNET': 'C2',
                  'RF': 'C0',
                  'SVM': 'C5'}

    if problem == 'norm_abn':
        fig, ax = plt.subplots(figsize=(3.3,3.3))

        for model in models:
            # extract correct dataset for model
            X_test_m = X_test[model.dataset]
            scores = model.predict_proba(X_test_m)[:,1]
            print(model.model_name_abrv, '\tdecision boundries:', model.decision_boundries, '\tindex:', 0)
            eu.plot_ROC(y_test, scores, pos_label=1, ax=ax, thresh=model.decision_boundries[0],
                        filepath=f"{results_dir}/ROC.png",
                        color=color_dict[model.model_name_abrv], label_size=7.5)


    elif problem == 'localization':
        print('COMP ROC CURVES', file=sys.stderr)
        print('Figure 1', file=sys.stderr)
        outcome_names = y_test.columns
        # n_models plots, each with curves for each vessel+abn_type pair
        for model in models:
            print('model: ', model.model_name_abrv, file=sys.stderr)
            # extract correct dataset for model
            X_test_m = X_test[model.dataset]
            fig, ax = plt.subplots(figsize=(3.3,3.3))
            for i, outcome in enumerate(outcome_names): 
                scores = model.predict_proba(X_test_m)[i][:,1]
                print('scores shape:', scores.shape)
                print(model.model_name_abrv, '\tdecision boundries:', model.decision_boundries, '\tindex:', i, file=sys.stderr)
                eu.plot_ROC(y_test[outcome],
                            scores,
                            pos_label=1,
                            ax=ax,
                            label=outcome,
                            thresh=model.decision_boundries[i],
                            title=model.model_name_abrv + ' ROC Comparison',
                            filepath=f"{results_dir}/{model.model_name_abrv}_ROC.png",
                            color=color_dict[model.model_name_abrv],
                            label_size=9)
    
       
        # 6 plots, 1 for each vessel+abn type with AUC curves from all models
        print('Figure 2', file=sys.stderr)
        fig, axs = plt.subplots(2, 3, figsize=(9, 6))
        for outcome_i, outcome in enumerate(outcome_names):
            abn = outcome.split('_')[0]
            ves = outcome.split('_')[1]
            row = 0 if abn == 'scar' else 1
            if ves  == 'lad': 
                col = 0
            elif ves  == 'rca': 
                col = 1
            elif ves  == 'lcx': 
                col = 2
            ax = axs[row][col]

            for model in models:
                # extract correct dataset for model
                print('model: ', model.model_name_abrv, file=sys.stderr)
                X_test_m = X_test[model.dataset]
                scores = model.predict_proba(X_test_m)[outcome_i][:,1]
                eu.plot_ROC(y_test[outcome],
                            scores,
                            pos_label=1,
                            ax=ax,
                            label=model.model_name_abrv,
                            thresh=model.decision_boundries[outcome_i],
                            title=outcome + ' ROC Comparison',
                            filepath=f"{results_dir}/per_output_ROC.png",
                            color=color_dict[model.model_name_abrv],
                            label_size=10)
                
#         for scar, ischemia, abnormal:
#         1 figure, 4 plots: micro all vessels; lad; lcx; rca
        # START WITH SCAR
        print('Figure 3', file=sys.stderr)
        vessels = ['lad', 'rca', 'lcx']
        for outcome in ['scar', 'ischemia', 'abnormal']:
            fig, axs = plt.subplots(2, 2, figsize=(3.3, 3.3))
            for model in models:
                # extract correct dataset for model
                print('model: ', model.model_name_abrv, file=sys.stderr)
                X_test_m = X_test[model.dataset]
                scores, labels = preds_per_vessel(model,
                                                  X_test_m,
                                                  y_test,
                                                  outcome,
                                                  vessel=None,
                                                  probs=True)
                eu.plot_ROC(labels,
                            scores,
                            pos_label=1,
                            ax=axs[0][0],
                            thresh=0.0,
                            title='Pooled',
                            filepath=f"{results_dir}/{outcome}_ROC.png",
                            color=color_dict[model.model_name_abrv],
                            label_size=7.5,
                            y_label=True,
                            x_label=False)


                for i, vessel in enumerate(vessels):
                    scores, labels = preds_per_vessel(model,
                                                      X_test_m,
                                                      y_test,
                                                      outcome,
                                                      vessel=vessel,
                                                      probs=True)
                    if i == 0:
                        row = 0
                        col = 1
                    else:
                        row = 1
                        col = i % 2

                    # Find the idx in model.decision_boundries corresponding to
                    # outcome x vessel territory combo
                    dec_bound_idxs = {'scar': {'lcx': 2, 'rca': 1, 'lad': 0},
                                      'ischemia': {'lcx': 5, 'rca': 4, 'lad': 3}}

                    if outcome == 'abnormal':
                        thresh = 0.0
                    else:
                        thresh = model.decision_boundries[dec_bound_idxs[outcome][vessel]]
                   
                    x_label = True if row == 1 else False
                    y_label = True if col == 0 else False 
                    
                    eu.plot_ROC(labels,
                                scores,
                                pos_label=1,
                                ax=axs[row][col],
                                thresh=thresh,
                                title=vessel.upper(),
                                filepath=f"{results_dir}/{outcome}_ROC.png",
                                color=color_dict[model.model_name_abrv],
                                label_size=7.5,
                                x_label=x_label,
                                y_label=y_label)


def plot_auc_with_ci(results, models, results_dir, problem, outcome, vessel, axs, axs_title=None):
    """
    Plot bar chart of AUC values with confidence intervals

    results : pd.DataFrame
        Results df for DeLong's test for problem, outcome, vessel
    """

    fig, ax = plt.subplots(figsize=(5.5, 5))

    model_names = [model.model_name_abrv for model in models]
    # We don't include Dummy
    if 'Dummy' in model_names:
        model_names.remove('Dummy')
    first_in_pair = results['models'].apply(lambda x : x.split('-')[0])
    second_in_pair = results['models'].apply(lambda x : x.split('-')[1])

    cis = []
    aucs = []
    for model in model_names:
        try:
            ci = results[first_in_pair == model]['model 1 95% ci'].values[0]
            ci = eval(ci)
            auc = results[first_in_pair == model]['model 1 auc'].values[0]
        except IndexError:
            ci = results[second_in_pair == model]['model 2 95% ci'].values[0]
            ci = eval(ci)
            auc = results[second_in_pair == model]['model 2 auc'].values[0]
        cis.append(ci)
        aucs.append(auc)
    
    # Make plot
    y_pos = np.arange(len(aucs))
    xerr = np.abs(np.array(aucs) - np.array(cis).T)
    color_dict = {'LR': 'C4',
                  'Lasso LR': 'C3',
                  'MLP': 'C1',
                  'UNET': 'C2',
                  'RF': 'C0',
                  'SVM': 'C5'}
    colors = [color_dict[m_name] for m_name in model_names]
    elinewidth = 4 if problem == 'localization' else 2
    fontsize = 36 if problem == 'localization' else 28

    ax.barh(y_pos, aucs, xerr=xerr, align='center', capsize=9, color=colors, error_kw={'elinewidth':elinewidth})
    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_names, fontsize=fontsize)
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_xlim(left=0.8)

    if problem == 'norm_abn':
        title = f"Per Patient Normal / Abnormal"
    elif problem == 'localization':
        vessel_str = vessel.upper() if vessel else 'All'
        title = f"{outcome.capitalize()} {vessel_str}" 

    for x, y, model, auc in zip(list(xerr[1] + np.array(aucs)), y_pos, model_names, aucs):
        ax.text(x + .02, y, "{:.2f}".format(auc), color='black', va='center', fontsize=fontsize)
    ax.set_frame_on(False)
    ymax = len(model_names) - 0.5
    ax.vlines(x=0, ymin=-0.5, ymax=ymax, color='black')
    
    if problem == 'norm_abn':
        filepath = os.path.join(results_dir,
                                f"auc_ci.png")
    elif problem == 'localization':

        filepath = os.path.join(results_dir,
                                f"auc_ci_{outcome}_{vessel_str.lower()}.png")

    fig.tight_layout()
    fig.savefig(filepath, dpi=300, transparent=True, bbox_inches='tight')


def delongs_test_by_task(models, X_test, y_test, results_dir, problem, outcome=None, vessel=None, perc_conf_interval=95, axs=None, axs_title=None):
    """
    Computes DeLong's test to compare roc curves between all pairs of models
    in models. Saves output to 'DeLongs.csv' in results_dir

    Parameters
    ----------
    models : python list of ModelEvalWrapper objects
        The models to evaluate

    X_test : Pandas dataframe
        The test set features

    y_test : Pandas series or numpy array
        The labels for the test set

    results_dir : string
        The directory where the resulting confusion matricies will be saved

    outcome : string
        The outcome that we want to compare models for. Only used when problem
        is 'localization'. Valid options include 'abnormal', 'scar', 'ischemia'.

    vessel : string
        The vessel territory on which we will compare model performance. Only
        used when problem is 'localization'. If None, then will compare per
        vessel performance accross all territories. Valid options include None,
        'rca', 'lcx', 'lad'

    perc_conf_interval : int
        The width of the confidence interval to estimate

    Returns
    -------
    None
    """

    # Activate pandas2ri for easy use of pandas datastructures with rpy2
    pandas2ri.activate()
    importr('pROC')
    
    results_dict = {'models': [],
                    'model 1 auc': [],
                    f"model 1 {perc_conf_interval}% ci": [],
                    'model 2 auc': [],
                    f"model 2 {perc_conf_interval}% ci": [],
                    'p value': []}

    # Memoize predictions in probs_dict so each model only runs once
    probs_dict = {}
    print("DeLong's Test")
    for i, model_i in enumerate(models):                                            
        for j, model_j in enumerate(models[i + 1:]): 

            # Don't do DeLong's test with Dummy classifier
            if ((model_i.model_name_abrv == 'Dummy') or
                (model_j.model_name_abrv == 'Dummy')):
                continue
            print("model i: \t", model_i.model_name_abrv,
                  "model j: \t", model_j.model_name_abrv)
            X_test_i = X_test[model_i.dataset]
            X_test_j = X_test[model_j.dataset]
            
            if problem != 'localization':
                # Memoization
                if model_i.model_name_abrv not in probs_dict:
                    model_i_probs = model_i.predict_proba(X_test_i)[:,1]
                    probs_dict[model_i.model_name_abrv] = model_i_probs
                else:
                    model_i_probs = probs_dict[model_i.model_name_abrv]

                if model_j.model_name_abrv not in probs_dict:
                    model_j_probs = model_j.predict_proba(X_test_j)[:,1]
                    probs_dict[model_j.model_name_abrv] = model_j_probs
                else:
                    model_j_probs = probs_dict[model_j.model_name_abrv]

                y_labels = y_test

            else:
                # as long as outcome is the same in preds_per_vessel,
                # y_labels will be the same for both calls

                # Memoization
                if model_i.model_name_abrv not in probs_dict:
                    model_i_probs, _ = preds_per_vessel(model_i,
                                                        X_test_i,
                                                        y_test,
                                                        outcome,
                                                        vessel=vessel,
                                                        probs=True)
                    probs_dict[model_i.model_name_abrv] = model_i_probs
                else:
                    model_i_probs = probs_dict[model_i.model_name_abrv]

                if model_j.model_name_abrv not in probs_dict:
                    model_j_probs, y_labels = preds_per_vessel(model_j,
                                                               X_test_j,
                                                               y_test,
                                                               outcome,
                                                               vessel=vessel,
                                                               probs=True)
                    probs_dict[model_j.model_name_abrv] = model_j_probs
                else:
                    model_j_probs = probs_dict[model_j.model_name_abrv]


            roc_i = r['roc'](response=y_labels, predictor=model_i_probs)
            roc_j = r['roc'](response=y_labels, predictor=model_j_probs)

            delong_test = r['roc.test'](roc_i, roc_j, paired=True)
            delong_test = dict(zip(delong_test.names, list(delong_test)))
            p = delong_test['p.value'][0]

            ci_i = r['ci.auc'](roc_i, 0.01 * perc_conf_interval, method='bootstrap')
            auc_i = ci_i[1]
            ci_i = f"[{ci_i[0] :.4f}, {ci_i[2] :.4f}]"

            ci_j = r['ci.auc'](roc_j, 0.01 * perc_conf_interval, method='bootstrap')
            auc_j = ci_j[1]
            ci_j = f"[{ci_j[0] :.4f}, {ci_j[2] :.4f}]"

            results_dict['models'].append(f"{model_i.model_name_abrv}-{model_j.model_name_abrv}")
            results_dict['p value'].append(p)
            results_dict['model 1 auc'].append(auc_i)
            results_dict[f"model 1 {perc_conf_interval}% ci"].append(ci_i)
            results_dict['model 2 auc'].append(auc_j)
            results_dict[f"model 2 {perc_conf_interval}% ci"].append(ci_j)

    if problem != 'localization':
        results = pd.DataFrame.from_dict(results_dict)
        results.to_csv(f"{results_dir}/DeLongs.csv", index=False)
        if axs is not None:
           plot_auc_with_ci(results, models, results_dir, problem, outcome, vessel, axs)
    else:
        if vessel is None:
            filename = f"{results_dir}/DeLongs_{outcome}_all.csv"
        else:
            filename = f"{results_dir}/DeLongs_{outcome}_{vessel}.csv"
        results = pd.DataFrame.from_dict(results_dict)
        results.to_csv(filename, index=False)
        if axs is not None:
           plot_auc_with_ci(results, models, results_dir, problem, outcome, vessel, axs, axs_title)

def delongs_test(models, X_test, y_test, results_dir, problem, perc_conf_interval=95):
    """
    Computes DeLong's test to compare roc curves between all pairs of models
    in models. Saves output to 'DeLongs.csv' in results_dir

    Parameters
    ----------
    models : python list of ModelEvalWrapper objects
        The models to evaluate

    X_test : Pandas dataframe
        The test set features

    y_test : Pandas series or numpy array
        The labels for the test set

    results_dir : string
        The directory where the resulting confusion matricies will be saved

    perc_conf_interval : int
        The width of the confidence interval to estimate

    Returns
    -------
    None
    """

    if problem == 'norm_abn':
        fig, axs = plt.subplots(figsize=(5,5))
        delongs_test_by_task(models,
                             X_test,
                             y_test,
                             results_dir,
                             problem,
                             outcome=None,
                             vessel=None,
                             perc_conf_interval=95,
                             axs=axs)
         
        
    elif problem == 'localization':
        vessels = [None, 'lad', 'lcx', 'rca']
        outcomes = ['abnormal', 'scar', 'ischemia']
        for outcome in outcomes:
            fig, axs = plt.subplots(2, 2, figsize=(10,10))
            axs_title=None
            print(outcome)
            for vessel in vessels:
                delongs_test_by_task(models,
                                     X_test,
                                     y_test,
                                     results_dir,
                                     problem,
                                     outcome=outcome,
                                     vessel=vessel,
                                     perc_conf_interval=95,
                                     axs=axs,
                                     axs_title=axs_title)


def parse_args():
    """
    Parse command line arguments 

    Parameters
    ----------
    None

    Returns
    -------
    arg_dict : dictionary
        Dictionary containing values extracted from command line arguments
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate and compare model performance')
    parser.add_argument('models', type=str, nargs='+',
                        help='The file paths to the models to evaluate. Models must be objects of type ModelEvalWrapper, and they must be stored as .joblib files.')
    parser.add_argument('--outdir', type=str, 
                        help='The path to the output directory where results will be saved')
    parser.add_argument('--datadir', type=str, 
                        help='The path to the base directory containing the data')
    parser.add_argument('--problem', type=str, choices=['norm_abn', 'localization'],
                        help="""The classification problem. Valid options
                                include 'norm_abn' and 'localization'""")
    parser.add_argument('--perc_conf_interval', nargs='?', type=int, default=95, 
                        help='Percent confidence interval to calculate. Default is 95')
    parser.add_argument('--auc_ci', action='store_true',
                        help='Save bar chart and csv comparing auc values with confidence intervals for all models')
    parser.add_argument('--confusion_matrix', action='store_true',
                        help='Create confusion matricies for all models')
    parser.add_argument('--compare_performance', action='store_true',
                        help='Full performance comparison between models including accuracy, precision, recall, F1, specificity')
    parser.add_argument('--roc', action='store_true',
                        help='Plot roc curves for all models')
    parser.add_argument('--delong', action='store_true',
                        help="Compute DeLong's test to compare roc curves for each pair of models")
    parser.add_argument('--all', action='store_true',
                        help="Run all evaluations")
    parser.add_argument('--final', action='store_true',
                        help="""Run final model on held out test set for
                                generalization performance""")
    parser.add_argument('--dataset', type=str, choices=['3_vessel', '17_segment'],
                        help="The dataset to train on. Valid choices include '3_vessel' and '17_segment'")
    args = parser.parse_args()

    return args


def main():
    # Parse arguments
    args = parse_args()
    model_paths = args.models
    dataset = args.dataset
    results_dir = args.outdir[:-1] if args.outdir[-1] == '/' else args.outdir
    data_dir = args.datadir
    problem = args.problem
    auc_ci = args.auc_ci
    perc_conf_interval = args.perc_conf_interval
    confusion_matrix = args.confusion_matrix
    compare_performance = args.compare_performance
    roc = args.roc
    delong = args.delong
    run_all = args.all
    final = args.final

    # Load models
    print('Loading models')
    models = []
    datasets = set()
    for model_path in model_paths:
        model = load(model_path)
        models.append(model) 
        datasets.add(model.dataset)
    
    if final:
        if len(models) != 1:
            raise AttributeError("""In order to get good estimate of
                                    generalization performance, you should not
                                    pass more than one model if running
                                    --final""")


    # y_test will be the same for same problem regardless of dataset 
    print('Loading dataset')
    if final:
        split = 'test'
        print('split: ', split)
    else:
        split = 'val'
    X_test = {}
    for dataset in list(datasets):
        data = data_utils.load_dataset(data_dir, dataset, problem, split)
        X_test[dataset], y_test = data['X'], data['y']

    if final:
        print("Starting model performance comparison")
        compare_model_performance(models, X_test, y_test, results_dir, problem) 
        print("Completed model performance comparison\n")
        print("Starting ROC curve comparison")
        compare_roc_curves(models, X_test, y_test, results_dir, problem) 
        print("Completed ROC curve comparison\n")
    else:
        # run evaluations
        if run_all:
            if problem == 'norm_abn':
                print("Creating confusion matricies")
                compare_conf_matricies(models, X_test, y_test, results_dir)
                print("Completed confusion matriciesi\n")
            print("Starting model performance comparison")
            compare_model_performance(models, X_test, y_test, results_dir, problem) 
            print("Completed model performance comparison\n")
            print("Starting ROC curve comparison")
            compare_roc_curves(models, X_test, y_test, results_dir, problem) 
            print("Completed ROC curve comparison\n")
            print("Starting DeLong's test")
            delongs_test(models, X_test, y_test, results_dir, problem)
            print("Completed DeLong's test")
        else:
            if confusion_matrix:
                print("Creating confusion matricies")
                compare_conf_matricies(models, X_test, y_test, results_dir)
                print("Completed confusion matricies\n")
            if compare_performance:
                print("Starting model performance comparison")
                compare_model_performance(models, X_test, y_test, results_dir, problem)
                print("Completed model performance comparison\n")
            if roc:
                print("Starting ROC curve comparison")
                compare_roc_curves(models, X_test, y_test, results_dir, problem) 
                print("Completed ROC curve comparison\n")
            if delong:
                print("Starting DeLong's test")
                delongs_test(models, X_test, y_test, results_dir, problem)
                print("Completed DeLong's test")


if __name__ == "__main__":
    main()
