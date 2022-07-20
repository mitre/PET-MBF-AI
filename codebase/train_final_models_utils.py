"""
Functions to train final models and save them as ModelEvalWrapper instances
in the appropriate subdirectory of Saved_Models
"""
# Add to sys path so codebase modules can be found
import sys
import os
BASE_DIR = os.getcwd().split('e-emagin-pet')[0] + 'e-emagin-pet/'
sys.path.append(BASE_DIR)

import pandas as pd
import numpy as np
import json

from codebase import data_utils
from codebase.ModelEvalWrapper import ModelEvalWrapper
from codebase import MLP
from codebase import unet

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score, make_scorer
from sklearn.metrics import f1_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import tensorflow as tf
from tensorflow import keras

from joblib import dump, load
from datetime import datetime


def train_final_svm(X_train, y_train, X_val, y_val, dataset, problem, hyperparams, db_method='youdens', recall_target=None, saved_models_path=None):
    """
    Trains final SVM and saves model in ModelEvalWrapper class instance for
    compatibility with evaluate.py script for easy model comparison. Should
    be called with the top hyperparameters found in model selection notebook.

    X_train : pd.DataFrame
        Training data for final model. This should be split 0 from 
        data_utils.load_dataset where val_split is 'nn_val_split'

    y_train : pd.Series
        Corresplonding labels to X_train 

    dataset : string
        The dataset that the model is trained on. Valid options include
        '3_vessel', '17_segment', 'polar_plot'.

    problem : string
        The prediction problem the model is trained to predict. Options include
        'norm_abn', 'localization'

    hyperparams : python dictionary
        Dictionary containing the top hyperparameter values identified in the
        model selection notebook.
    
    db_method : str
        Method used to determine decision boundries. Valid options are
        'youdens' or 'fixed_recall'
        
    recall_target : float
        Target recall value if method is 'fixed_recall', otherwise must be None

    saved_models_path : string
        Optional path to directory where model will be saved. If None, then 
        default is /{BASE_DIR}/Saved_Models/{dataset}/{problem}
    """
    kernel = hyperparams['kernel'] 
    C = hyperparams['C']
    gamma = hyperparams['gamma']

    pl_svm = Pipeline(steps=[('scaler', StandardScaler()),
                             ('svc', SVC(random_state=1,
                                         probability=True,
                                         kernel=kernel,
                                         C=C,
                                         gamma=gamma))])
    pl_svm.fit(X_train, y_train)

    # save as ModelEvalWrapper
    feature_cols = X_train.columns.values.tolist()
    pl_svm = ModelEvalWrapper(pl_svm, 'Support Vector Machine', 'SVM', 'sklearn', dataset, feature_cols=feature_cols)
    if db_method is not None:
        pl_svm.set_decision_boundries(X_val, y_val, method=db_method, target=recall_target)

    if not saved_models_path:
        saved_models_path = os.path.join(BASE_DIR,
                                         'Saved_Models',
                                         dataset,
                                         problem)

    os.makedirs(saved_models_path, exist_ok=True)
    dump(pl_svm, os.path.join(saved_models_path,'svm.joblib'))



def train_final_rf(X_train, y_train, X_val, y_val, dataset, problem, hyperparams, db_method='youdens', recall_target=None, saved_models_path=None):
    """
    Trains final RF and saves model in ModelEvalWrapper class instance for
    compatibility with evaluate.py script for easy model comparison. Should
    be called with the top hyperparameters found in model selection notebook.

    X_train : pd.DataFrame
        Training data for final model. This should be split 0 from 
        data_utils.load_dataset where val_split is 'nn_val_split'

    y_train : pd.Series
        Corresplonding labels to X_train 

    dataset : string
        The dataset that the model is trained on. Valid options include
        '3_vessel', '17_segment', 'polar_plot'.

    problem : string
        The prediction problem the model is trained to predict. Options include
        'norm_abn', 'localization'

    hyperparams : python dictionary
        Dictionary containing the top hyperparameter values identified in the
        model selection notebook.

    db_method : str
        Method used to determine decision boundries. Valid options are
        'youdens' or 'fixed_recall'
        
    recall_target : float
        Target recall value if method is 'fixed_recall', otherwise must be None

    saved_models_path : string
        Optional path to directory where model will be saved. If None, then 
        default is /{BASE_DIR}/Saved_Models/{dataset}/{problem}
    """
    n_estimators = hyperparams['n_estimators'] 
    max_depth = hyperparams['max_depth']
    min_samples_split = hyperparams['min_samples_split']
    min_samples_leaf = hyperparams['min_samples_leaf']
    max_features = hyperparams['max_features']
    random_state = hyperparams['random_state']

    rf = RandomForestClassifier(random_state=random_state,
                                n_estimators=n_estimators, 
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                max_features=max_features)
    rf.fit(X_train, y_train)

    # save as ModelEvalWrapper
    feature_cols = X_train.columns.values.tolist()
    rf = ModelEvalWrapper(rf, 'Random Forest', 'RF', 'sklearn', dataset, feature_cols=feature_cols)
    if db_method is not None:
        rf.set_decision_boundries(X_val, y_val, method=db_method, target=recall_target)

    if not saved_models_path:
        saved_models_path = os.path.join(BASE_DIR,
                                         'Saved_Models',
                                         dataset,
                                         problem)
    os.makedirs(saved_models_path, exist_ok=True)
    dump(rf, os.path.join(saved_models_path, 'rf.joblib'))



def train_final_mlp(X_train, y_train, X_val, y_val, outdir, dataset, problem, hyperparams, db_method='youdens', recall_target=None, saved_models_path=None):
    """
    Trains final MLP and saves model in ModelEvalWrapper class instance for
    compatibility with evaluate.py script for easy model comparison. Should
    be called with the top hyperparameters found in model selection notebook.

    X_train : pd.DataFrame
        Training data for final model. This should be split 0 from 
        data_utils.load_dataset where val_split is 'nn_val_split'

    y_train : pd.Series
        Corresplonding labels to X_train 

    X_train : pd.DataFrame
        Validation data for final model. This should be split 1 from 
        data_utils.load_dataset where val_split is 'nn_val_split'

    y_train : pd.Series
        Corresplonding labels to X_val 

    outdir : string
        The path to the base output directory.

    dataset : string
        The dataset that the model is trained on. Valid options include
        '3_vessel', '17_segment', 'polar_plot'.

    problem : string
        The prediction problem the model is trained to predict. Options include
        'norm_abn', 'localization'

    hyperparams : python dictionary
        Dictionary containing the top hyperparameter values identified in the
        model selection notebook.

    db_method : str
        Method used to determine decision boundries. Valid options are
        'youdens' or 'fixed_recall'
        
    recall_target : float
        Target recall value if method is 'fixed_recall', otherwise must be None

    saved_models_path : string
        Optional path to directory where model will be saved. If None, then 
        default is /{BASE_DIR}/Saved_Models/{dataset}/{problem}
    
    Returns: None
    """
    # parameters for make_model
    if dataset == '3_vessel':
        input_shape = 12
    elif dataset == '17_segment':
        input_shape = 68
    else:
        raise NotImplementedError()

    if problem == 'norm_abn':
        num_classes = 2
        num_outputs = 1
    elif problem == 'localization':
        num_classes = 2
        num_outputs = 6

    # unpack hyperparameters
    epochs = hyperparams['epochs']
    checkpoint_metric = hyperparams['checkpoint_metric']
    hidden_sizes = hyperparams['hidden_sizes'] 
    learning_rate = hyperparams['learning_rate']
    drop_prob = hyperparams['drop_prob']
    reg = hyperparams['reg']

    # tf.datasets for keras MLP
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train.values, y_train.values)).batch(1000)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val.values, y_val.values)).batch(1000)
    
    # tensorboard log directory 
    now = datetime.now().strftime("%Y%m%d-%H%M%S") 
    logdir = os.path.join(outdir,
                          dataset,
                          problem,
                          'mlp',
                          'logs/tensorboard_final_model/scalars/',
                          now)

    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    tf.keras.backend.clear_session()
    

    # use model checkpointing to save the model at the epoch at which it
    # achieves greatest validation accuracy. This helps prevent overfitting.
    checkpoint_dir = os.path.join(outdir,
                                  dataset,
                                  problem,
                                  'mlp',
                                  'logs/checkpoint_final_model/',
                                  now)

    checkpoint_callback = keras.callbacks.ModelCheckpoint(
                               checkpoint_dir,
                               monitor=checkpoint_metric,
                               mode='max',
                               verbose=1,
                               save_best_only=True,
                               save_weights_only=True,
                           )

    model = MLP.make_model(input_shape=input_shape,
                           num_classes=num_classes,
                           num_outputs=num_outputs,
                           hidden_sizes=hidden_sizes,
                           learning_rate=learning_rate,
                           drop_prob=drop_prob,
                           reg=reg,
                           X_train=X_train.values)

    model.fit(train_dataset,
              epochs=epochs,
              validation_data=val_dataset,
              callbacks=[tensorboard_callback,
                         checkpoint_callback])

    # load weights from the best model saved in checkpoint
    model.load_weights(checkpoint_dir)

    # save final model
    if not saved_models_path:
        saved_models_path = os.path.join(BASE_DIR,
                                         'Saved_Models',
                                         dataset,
                                         problem)
    os.makedirs(saved_models_path, exist_ok=True)
    keras_model_path = os.path.join(saved_models_path, 'mlp.HDF5')
    model.save(keras_model_path)

    # save as ModelEvalWrapper
    feature_cols = X_train.columns.values.tolist()
    mlp = ModelEvalWrapper(None,
                          'Multilayer Perceptron',
                          'MLP',
                          'keras',
                          dataset,
                          feature_cols=feature_cols,
                          keras_model_path=keras_model_path)
    if db_method is not None:
        # get decision boundries and then reconstruct ModelEvalWrapper instance
        # with correct decision boundries because of very weird pickling error.
        # For whatever reason, this resolves the issue. 
        decision_boundries = mlp.calc_decision_boundries(X_val, y_val, method=db_method, target=recall_target)
        mlp = ModelEvalWrapper(None,
                              'Multilayer Perceptron',
                              'MLP',
                              'keras',
                              dataset,
                              feature_cols=feature_cols,
                              keras_model_path=keras_model_path,
                              decision_boundries=decision_boundries)

    print('\ndecision boundries: ', mlp.decision_boundries, '\n')
    dump(mlp, os.path.join(saved_models_path, 'mlp.joblib'))


def train_final_unet(X_train, y_train, X_val, y_val, outdir, dataset, problem, hyperparams, db_method='youdens', recall_target=None, saved_models_path=None):
    """
    Trains final MLP and saves model in ModelEvalWrapper class instance for
    compatibility with evaluate.py script for easy model comparison. Should
    be called with the top hyperparameters found in model selection notebook.

    X_train : pd.DataFrame
        Training data for final model. This should be split 0 from 
        data_utils.load_dataset where val_split is 'nn_val_split'

    y_train : pd.Series
        Corresplonding labels to X_train 

    X_train : pd.DataFrame
        Validation data for final model. This should be split 1 from 
        data_utils.load_dataset where val_split is 'nn_val_split'

    y_train : pd.Series
        Corresplonding labels to X_val 

    outdir : string
        The path to the base output directory.

    dataset : string
        The dataset that the model is trained on. Valid options include
        '3_vessel', '17_segment', 'polar_plot'.

    problem : string
        The prediction problem the model is trained to predict. Options include
        'norm_abn', 'localization'

    hyperparams : python dictionary
        Dictionary containing the top hyperparameter values identified in the
        model selection notebook.

    db_method : str
        Method used to determine decision boundries. Valid options are
        'youdens' or 'fixed_recall'
        
    recall_target : float
        Target recall value if method is 'fixed_recall', otherwise must be None

    saved_models_path : string
        Optional path to directory where model will be saved. If None, then 
        default is /{BASE_DIR}/Saved_Models/{dataset}/{problem}

    
    Returns: None
    """
    assert(dataset == 'polar_plot')
    # parameters for make_model
    if dataset == '3_vessel':
        input_shape = 12
    elif dataset == '17_segment':
        input_shape = 68
    else:
        input_shape = X_train[0].shape

    if problem == 'norm_abn':
        num_classes = 2
        num_outputs = 1
    elif problem == 'localization':
        num_classes = 2
        num_outputs = 6

    # unpack hyperparameters
    epochs = hyperparams['epochs']
    checkpoint_metric = hyperparams['checkpoint_metric']
    f = hyperparams['f']
    k = hyperparams['k']
    l = hyperparams['l']
    hidden_sizes = hyperparams['fc_hidden_sizes'] 
    learning_rate = hyperparams['learning_rate']
    drop_prob = hyperparams['drop_prob']
    reg = hyperparams['reg']

    # tf.datasets for keras unet
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(128)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(128)
    
    # tensorboard log directory 
    now = datetime.now().strftime("%Y%m%d-%H%M%S") 
    logdir = os.path.join(outdir,
                          dataset,
                          problem,
                          'unet',
                          'logs/tensorboard_final_model/scalars/',
                          now)

    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    tf.keras.backend.clear_session()
    

    # use model checkpointing to save the model at the epoch at which it
    # achieves greatest validation accuracy. This helps prevent overfitting.
    checkpoint_dir = os.path.join(outdir,
                                  dataset,
                                  problem,
                                  'unet',
                                  'logs/checkpoint_final_model/',
                                  now)

    checkpoint_callback = keras.callbacks.ModelCheckpoint(
                               checkpoint_dir,
                               monitor=checkpoint_metric,
                               mode='max',
                               verbose=1,
                               save_best_only=True,
                               save_weights_only=True,
                           )

    # f, k, l, fc_hidden_sizes, learning_rate, drop_prob, reg
    model = unet.make_model(input_shape=input_shape,
                            num_classes=num_classes,
                            num_outputs=num_outputs,
                            f=f,
                            k=k,
                            l=l,
                            fc_hidden_sizes=hidden_sizes,
                            learning_rate=learning_rate,
                            drop_prob=drop_prob,
                            reg=reg) 

    model.fit(train_dataset,
              epochs=epochs,
              validation_data=val_dataset,
              callbacks=[tensorboard_callback,
                         checkpoint_callback])

    # load weights from the best model saved in checkpoint
    model.load_weights(checkpoint_dir)

    # save final model
    if not saved_models_path:
        saved_models_path = os.path.join(BASE_DIR,
                                         'Saved_Models',
                                         dataset,
                                         problem)
    os.makedirs(saved_models_path, exist_ok=True)
    keras_model_path = os.path.join(saved_models_path, 'unet.HDF5')
    model.save(keras_model_path)

    # save as ModelEvalWrapper
    # Set model to None for ModelEvalWrapper for keras models and instead use
    # keras_model_path.
    # Must set decision boundries in constructor. It creates pickeling issue 
    # to use set_decision_boundries_youdens for some reason with keras models
    unet_wrapped = ModelEvalWrapper(None,
                          'UNET',
                          'UNET',
                          'keras',
                          dataset,
                          feature_cols=None,
                          keras_model_path=keras_model_path)
    if db_method is not None:
        # get decision boundries and then reconstruct ModelEvalWrapper instance
        # with correct decision boundries because of very weird pickling error.
        # For whatever reason, this resolves the issue. 
        decision_boundries = unet_wrapped.calc_decision_boundries(X_val, y_val, method=db_method, target=recall_target)
        unet_wrapped = ModelEvalWrapper(None,
                                       'UNET',
                                       'UNET',
                                       'keras',
                                       dataset,
                                       feature_cols=None,
                                       keras_model_path=keras_model_path,
                                       decision_boundries=decision_boundries)

    print('\ndecision boundries: ', unet_wrapped.decision_boundries, '\n')
    dump(unet_wrapped, os.path.join(saved_models_path, 'unet.joblib'))

def train_dummy(X_train, y_train, X_val, y_val, dataset, problem, saved_models_path=None):
    """
    Trains dummy classifier and saves model in ModelEvalWrapper class instance
    for compatibility with evaluate.py script for easy model comparison. 

    X_train : pd.DataFrame
        Training data for final model. This should be split 0 from 
        data_utils.load_dataset where val_split is 'nn_val_split'

    y_train : pd.Series
        Corresplonding labels to X_train 

    dataset : string
        The dataset that the model is trained on.  Always should be '17_segment'
        for dummy classifier.

    problem : string
        The prediction problem the model is trained to predict. Options include
        'norm_abn', 'localization'

    hyperparams : python dictionary
        Dictionary containing the top hyperparameter values identified in the
        model selection notebook.

    saved_models_path : string
        Optional path to directory where model will be saved. If None, then 
        default is /{BASE_DIR}/Saved_Models/{dataset}/{problem}
    """
    dc = DummyClassifier(strategy='stratified')
    dc.fit(X_train, y_train)
    if problem == 'localization':
        decision_boundries = [0.5] * 6
    elif problem == 'norm_abn':
        decision_boundries = [0.5] 


    # save as ModelEvalWrapper
    dc = ModelEvalWrapper(model=dc,
                          model_name='Dummy Classifier',
                          model_name_abrv='Dummy',
                          model_type='sklearn',
                          dataset=dataset,
                          decision_boundries=decision_boundries)

    if not saved_models_path:
        saved_models_path = os.path.join(BASE_DIR,
                                         'Saved_Models',
                                         dataset,
                                         problem)

    os.makedirs(saved_models_path, exist_ok=True)
    dump(dc, os.path.join(saved_models_path,'dummy.joblib'))

def save_models(datadir,
                outdir,
                dataset,
                problem,
                svm_hyperparams=None,
                rf_hyperparams=None,
                mlp_hyperparams=None,
                unet_hyperparams=None,
                dummy=False,
                db_method='youdens',
                recall_target=None,
                saved_models_path=None):
    # Train all models on the same part of training set that nn will be 
    # trained on for a fair comparison between all models
    data = data_utils.load_dataset(datadir,
                                   dataset,
                                   problem,
                                   'train',
                                   'nn_val_split')

    X, y, val_split = data['X'], data['y'], data['val_split']
    X_train, y_train = X[val_split == 0], y[val_split == 0]
    X_val, y_val = X[val_split == 1], y[val_split == 1]


    # Train and save models with top hyperparameters identified in model 
    # selection notebook
    
    if dummy:
        train_dummy(X_train, y_train, X_val, y_val, dataset, problem, saved_models_path)
    if svm_hyperparams:
        train_final_svm(X_train, y_train, X_val, y_val, dataset, problem, svm_hyperparams, db_method, recall_target, saved_models_path)  
        print('\nSVM saved\n')
    if rf_hyperparams:
        train_final_rf(X_train, y_train, X_val, y_val, dataset, problem, rf_hyperparams, db_method, recall_target, saved_models_path)
        print('\nRF saved\n')
    if mlp_hyperparams:
        train_final_mlp(X_train, y_train, X_val, y_val, outdir, dataset, problem, mlp_hyperparams, db_method, recall_target, saved_models_path)  
        print('\nMLP saved\n')
    if unet_hyperparams:
        data = data_utils.load_dataset(datadir,
                                       'polar_plot',
                                       problem,
                                       'train',
                                       'nn_val_split')

        X, y, val_split = data['X'], data['y'], data['val_split']
        X_train, y_train = X[val_split == 0], y[val_split == 0]
        X_val, y_val = X[val_split == 1], y[val_split == 1]
        train_final_unet(X_train, y_train, X_val, y_val, outdir, 'polar_plot', problem, unet_hyperparams, db_method, recall_target, saved_models_path)  
        print('\nUNET saved\n')

