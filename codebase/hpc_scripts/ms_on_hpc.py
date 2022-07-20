import sys
import os
BASE_PATH = os.getcwd().split('e-emagin-pet')[0] + 'e-emagin-pet/'
sys.path.append(BASE_PATH)

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score, make_scorer
from sklearn.metrics import f1_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras import initializers, regularizers
from tensorflow.keras.callbacks import TensorBoard

from codebase import data_utils, tune_utils
from codebase import model_selection_utils as ms
from codebase import evaluation_utils as ev
from codebase.custom_metrics import multilabel_roc_auc 

import argparse
from datetime import datetime
from functools import partial
import joblib
import json

import ray
from ray import tune
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.logger import DEFAULT_LOGGERS
from hyperopt import hp




"""
*******************************************************************************
*
*       Below is the function train_for_tune which is used by tune to train
*       models throughout the hyperparameter tuning process. 
*       Because of a quirk of Ray, this function must remain in this file. 
*       Moving to another file and importing will cause an issue with Ray. 
*
*******************************************************************************
"""

def train_for_tune(config, checkpoint_dir=None, model=None, data_dir=None, dataset=None, problem=None, base_path=None, epochs=150, search_alg_abrv='ho'):
    """
        This is the function that Tune calls every time a model is trained. 
        This function will be converted to a trainable class automatically by
        Tune. Config and checkpoint_dir are the only arguments that are a part
        of the official Tune trainable interface. When passed to Tune.run(),
        this function must be modified such that all other arguments are set 
        through functools.partial().

        A tunable function must load the data, build the model by setting
        hyperparameters according to the specifications in the config 
        dictionary, and train the model. For Tune to track the hyperparameter 
        specifications and training and validation metrics, when the model is 
        fit, callbacks must be specified such that Tune.report() is called 
        (this occurs in TuneReporterCallback).

        config : python dictionary
            Dictionary of hyperparameters specifying the architecture of the
            model to be trained. This will be generated and passed by Ray Tune.
            Hyperparameter space / potential values for config are specified in
            the hp_spec file which, if using the interface in
            model_selection_utils, was created and saved in that file. 

            TODO FORMAT OF THE CONFIG FILE / FIGURE OUT WHERE THIS SHOULD
            BE SPECIFIED

        checkpoint_dir : string
            Not currently used. An optional part of the tunable interface
            to specify a directory to save checkpoints for checkpointing
            during training. 

        model : string
            Name of the model type to be trained. 'mlp' or 'unet' are valid options

        dataset : str
            The dataset to train on. Options are '3_vessel' or '17_segment'.

        data_dir : string
            A path to the directory which holds the data files. 

        problem : string
            Valid options (currently) include 'norm_abn'

        base_path : string
            The path to the 'e-emagin-pet' directory--the base directory of
            this repository. 
        
        epochs : int
            The number of epochs to train for.

        search_alg_abrv : string
            Either 'ho' for HyperOpt or 'gs' for grid search.
    """
    import sys
    import os
    sys.path.append(base_path)
    import tensorflow as tf
    from codebase import data_utils
    from codebase import MLP
    from codebase import unet
    from codebase import tune_utils

    assert(dataset in ['3_vessel', '17_segment', 'polar_plot'])

    # Whether to use all 4 features for polar plot (i.e. rest, stress, reserve,
    # difference) or whether to use just rest and stress.

#     polar_plot_features = 'rest_stress'
    polar_plot_features = 'rest_stress_reserve_difference'


    # Load data                   
    data = data_utils.load_dataset(data_dir,
                                   dataset,
                                   problem,
                                   'train',
                                   val_col='nn_val_split')

    X, y, val_split = data['X'], data['y'], data['val_split']
    X_train, y_train = X[val_split == 0], y[val_split == 0]
    X_val, y_val = X[val_split == 1], y[val_split == 1]

    if dataset != 'polar_plot':
        X_train, y_train = X_train.values, y_train.values
        X_val, y_val = X_val.values, y_val.values
    else:
        if polar_plot_features == 'rest_stress':
            X_train = X_train[:,:,:,:2]
            X_val = X_val[:,:,:,:2]
    

    batch_size=256
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)

    # Initialize NN with the parameters from config    

    # The name of the hyperparameter is different in config for UNET and MLP
    # to be more explicit in the interface
    if model == 'unet':
        layers_hp = 'fc_layers'
        num_layers_hp = 'num_fc_layers'
        # when X_train is array of images
        input_shape = X_train[0].shape
    elif model == 'mlp':
        layers_hp = 'layers'
        num_layers_hp = 'num_layers'
        # when X_train is feature matrix
        input_shape = X_train.shape[1]

    # Set hidden_sizes. config dictionary is in different format when search
    # alg is HyperOpt vs grid search
    if search_alg_abrv == 'ho':
        # For HyperOpt
        hidden_sizes=[config[layers_hp][f"l{i + 1}"] for i in range(config[layers_hp][num_layers_hp])]
    elif search_alg_abrv == 'gs':
        # For GridSearch
        hidden_sizes=[config[f"l{i + 1}"] for i in range(config[num_layers_hp])]

    print("\n\n input shape: ", input_shape, '\n\n')
    if problem == 'norm_abn':
        num_outputs = 1
    elif problem == 'localization':
        num_outputs = 6

    if model == 'mlp':
        nn = MLP.make_model(input_shape=input_shape,
                            num_classes=2,
                            num_outputs=num_outputs,
                            hidden_sizes=hidden_sizes,
                            learning_rate=config["lr"],
                            drop_prob=config["drop_prob"],
                            reg=config["reg"],
                            X_train=X_train)
    elif model == 'unet':
        f = config['f_k']['f']
        k = config['f_k']['k']
        print("input_shape:", input_shape)
        nn = unet.make_model(input_shape=input_shape,
                             num_classes=2,
                             num_outputs=num_outputs,
                             f=f,
                             k=k,
                             l=config['l'],
                             fc_hidden_sizes=hidden_sizes,
                             learning_rate=config['lr'],
                             drop_prob=config['drop_prob'],
                             reg=config['reg'])
    else:
        raise NotImplementedError("model must be 'mlp' or 'unet'")
  
    nn.fit(train_dataset,
           epochs=epochs,
           validation_data=val_dataset,
           verbose=0,
           callbacks=[tune_utils.TuneReporterCallback()])

"""
*******************************************************************************
*
*       Below are the functions related to running this command line script and
*       executing the various hyperparameter tuning approaches. 
*       
*******************************************************************************
"""


def parse_args():
    """
    Parse command line arguments
    """

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model', type=str, choices=['svm', 'rf', 'mlp', 'unet'],
                        help="""The type of model to train. Valid choices
                              include 'svm', 'rf', 'mlp', and 'unet'""")
    parser.add_argument('--problem',
                        type=str,
                        choices=['norm_abn', 'localization'],
                        help="""The classification problem to solve. Valid
                              choices include 'norm_abn' or 'localization'""")
    parser.add_argument('--data_dir',
                        required=True,
                        type=str,
                        help="""Path to the directory contianing the dataset""")
    parser.add_argument('--dataset', type=str, choices=['3_vessel', '17_segment', 'polar_plot'],
                        help="""The dataset to train on. Valid choices include
                              '3_vessel', '17_segment', or 'polar_plot'""")
    parser.add_argument('--hp_spec', type=str,
                        help='Path to the hyperparameter space specification file')
    parser.add_argument('--results_dir', type=str,
                        help='Path to directory to save results.')
    parser.add_argument('--job_name', type=str,
                        help='Job name')
    parser.add_argument('--project_name', type=str, nargs='?', default=None,
                        help="""Only used with Keras RandomSearch Tuner. 
                                Optional project_name to be passed to Tuner.
                                If not specified, a unique project name will 
                                be generated by combining job_name and 
                                timestamp""")
    parser.add_argument('--num_samples',
                        required='ho' in sys.argv,
                        type=int,
                        default=None, 
                        help="""When nn_search_alg is 'ho, num samples is the 
                                number of trials to run in Bayesian optimization.
                                When nn_search_alg is 'gs' this is the number
                                of times to resample the gridsearch (i.e.
                                num_samples=2 will run the gridsearch twice).
                                This can be helpful as random initializations
                                neural networks can alter results. Required 
                                when nn_search_alg is 'ho'""") 
    parser.add_argument('--epochs', required='mlp' in sys.argv, type=int, default=None, 
                        help="Number of epochs to run in random search. Required if model is 'mlp', otherwise not used.")
    parser.add_argument('--redis_password', type=str, default=None, 
                        help="""Redis password used in connection to Ray head
                              Node. Required if model is 'mlp', otherwise not
                              used.""")
    parser.add_argument('--nn_search_alg', required='mlp' in sys.argv, type=str, default=None,
                        choices=['ho', 'gs'],
                        help="""The algorithm to use for hyperparmeter tuning for
                              a neural network. Valid options include 'ho' for
                              HyperOpt and 'gs' for grid search.""")
    parser.add_argument('--gpus_per_node', type=int, nargs='?', default=0, 
                        help="""Number of GPUs per node requested in SLURM job.
                              This is used to determine whether to instruct Ray
                              to use GPU in hyperparameter tuning of MLP""")

    parser.add_argument('--local',
                        action='store_true',
                        help="""Use if the hyperparameter search is run
                                locally rather than on SLURM""")
    return parser.parse_args()


def ray_tune(model, data_dir, dataset, problem, hp_spec, job_name, results_dir, num_samples, epochs, redis_password=None, gpus_per_node=0, search_alg_abrv='ho', local=False):
    """
    Launch the hyperparameter tuning job with Ray Tune. A dataframe of the 
    results of the hyperparameter search.  The results directory will be 
    created and contain
    (1) results in the distributed format used by Ray Tune and 
    (2) TensorBoard logs for the hyperparameter search. This folder
    is saved in results_dir. See README for output directory structure. 

    Parameters:
    -----------

    model : string
        Name of the model type to be trained. 'mlp' or 'unet' are valid options

    data_dir : string
        Path to directory contianing data.

    dataset : string
        Valid options include '3_vessel', '17_segment'

    problem : string
        Valid options (currently) include 'norm_abn'

    hp_spec : string
        Path to the file containing the definition of the hyperparameter space.

    job_name : string
        The name of the job

    num_samples : int
        If search algorithm is HyperOpt, this is number of trials to run 
        HyperOpt for. Required when search algorithm is HyperOpt. If algorithm 
        is grid search, this is the number of times to try each configuration 
        in the grid search. Optional when search algorithm is grid search, with
        default num_samples=2.

    epochs : int
        The number of epochs to train for.

    redis_password : str
        The password for redis server. This is necessary for Tune. If called
        from run_ms_on_hpc.py, the redis password is randomly generated in
        the script.

    gpus_per_node : int
        The number of gpus allocated by SLURM. This is used to direct Ray in
        what resources to use per trial. If gpus_per_node > 0, then ray is 
        directed to use 1 gpu per trial.

    dataset : str
        The dataset to train on. Valid options are '3_vessel' or '17_segment'

    search_alg_abrv : str
        The search algorithm to use. Valid options are 'ho' for HyperOpt or
        'gs' for GridSearch

    local : bool
        Indicates whether or not the model seleciton operation is being
        run locally or in a SLURM job.

    Returns:
    --------
    pd.DataFrame of results of hyperparameter search
    """
    base_path = os.getcwd().split('e-emagin-pet')[0] + 'e-emagin-pet/'

    assert(search_alg_abrv == 'ho' or search_alg_abrv == 'gs')
    space = joblib.load(hp_spec)

    if search_alg_abrv == 'ho':
#         algo = HyperOptSearch( space, metric="val_auc", mode="max")
        algo = HyperOptSearch( space, metric="val_auc", mode="max", n_initial_points=100)
        num_samples=num_samples
    elif search_alg_abrv == 'gs':
        algo = None
        if num_samples == None:
            num_samples=2

    print("gpu: ", gpus_per_node) 
    ray.shutdown()
    if local:
        now_str = datetime.now().strftime("%Y%m%d-%H%M%S") 
        res_name = job_name + '-' + now_str 
        ray.init()
    else:
        job_id = os.environ['SLURM_JOB_ID']
        res_name = job_name + '-job' + job_id
        ray.init(address=os.environ["ip_head"], _redis_password=redis_password)

    print("Nodes in the Ray cluster:")
    print(ray.nodes())

    if gpus_per_node == 0:
        gpu = 0
    else:
        gpu = 1

    # Must pass partial of train for tune so that the only parameters
    # in the tunable function passed to tune.run are config, checkpoint_dir
    res = tune.run(
        partial(train_for_tune,
                model=model,
                data_dir=data_dir,
                dataset=dataset,
                problem=problem, 
                base_path=base_path,
                epochs=epochs,
                search_alg_abrv=search_alg_abrv),
        resources_per_trial={'gpu': gpu},
        config=space,
        search_alg=algo,
        name=res_name,
        loggers=DEFAULT_LOGGERS,
        local_dir=results_dir,
        num_samples=num_samples)

    res = res.results_df
    return res

def make_scorers(problem):
    """
    Parameters:
    -----------
    problem : str
        The classification problem to create scorers for

    Returns:
    --------
    The ppropriate scorers for the given problem for sklearn gridsearch
    """
    if problem == 'norm_abn':
        average = 'binary'
        auc_prob_scorer = make_scorer(roc_auc_score, needs_proba=True)
        precision_scorer = make_scorer(precision_score, pos_label=1, zero_division=0, average=average)
        recall_scorer = make_scorer(recall_score, pos_label=1, average=average)
        accuracy_scorer = make_scorer(accuracy_score)
        # specificity is sensitivity/recall with pos_label flipped!
        specificity_scorer = make_scorer(recall_score, pos_label=0, average=average)
        f1_scorer = make_scorer(f1_score, pos_label=1, average=average)

        scorers = {'precision': precision_scorer, 
                   'recall': recall_scorer, 
                   'accuracy': accuracy_scorer,
                   'specificity': specificity_scorer,
                   'f1': f1_scorer,
                   'auc': auc_prob_scorer}

    elif problem == 'localization':
        average = 'macro'
        auc_prob_scorer = make_scorer(multilabel_roc_auc, needs_proba=True, average='macro')
        precision_scorer = make_scorer(precision_score, pos_label=1, zero_division=0, average=average)
        recall_scorer = make_scorer(recall_score, pos_label=1, average=average)
        accuracy_scorer = make_scorer(accuracy_score)
        # specificity is sensitivity/recall with pos_label flipped!
        specificity_scorer = make_scorer(recall_score, pos_label=0, average=average)
        f1_scorer = make_scorer(f1_score, pos_label=1, average=average)


        scorers = {'precision': precision_scorer, 
                   'recall': recall_scorer, 
                   'accuracy': accuracy_scorer,
                   'f1': f1_scorer,
                   'auc': auc_prob_scorer}
    return scorers

def gridsearch_sklearn(X_train, y_train, val_split, problem, model, hp_spec, job_name):
    """
    Run gridsearch on a scikit learn model with GridSearchCV. Return the
    dataframe of the cv_results with precision, recall, accuracy, specificity,
    f1, and auc. Currently implemented for scikit learn
    implementations of SVM and RF models.
    
    Parameters:
    -----------

    X_train : pd.DataFrame
        The feature matrix.

    y_train : pd.Series
        The labels

    val_split : pd.Series
        Vector containing group assignments for each study. Used to create
        the folds for cross validation. 

    problem : string
        Vaid options include 'norm_abn' or 'localization'

    model : str
        The model to run gridsearch on. Either valid options are 'svm' or 'rf'

    hp_spec : str
        Path to the file containing the definition of the search space. This
        file should contain a jsonified python dictionary in the format 
        of scikit learn's model_selection.GridSearchCV's param_grid argument.
    
    job_name : str
        The name of the job.

    Returns:
    --------
    pd.DataFrame of grid search results
    """

    with open(hp_spec) as json_file:
        params = json.load(json_file)

    # copy params to new dict with keys suitable for sklearn pipeline
    params_pipeline_formatted = {}
    for key in params.keys():
        params_pipeline_formatted[model + '__' + key] = params[key]

    params = params_pipeline_formatted

    metric_region = pd.Series(X_train.columns).apply(lambda x : np.array(x.split('_', maxsplit=1)))
    metrics = np.unique(np.vstack(metric_region.values)[:,0])
    regions = np.unique(np.vstack(metric_region.values)[:,1])
    print('metrics: ', metrics)
    print('regions: ', regions)
    print('label cols: ', y_train.columns)

    if model == 'svm':
        pl = Pipeline(steps=[('scaler', StandardScaler()), (model, SVC(probability=True, random_state=1))])
    elif model == 'rf':
        pl = Pipeline(steps=[(model, RandomForestClassifier(random_state=1))])  

    scorers = make_scorers(problem)

    # Because the gamma parameter is not used with the linear kernel, we increase efficiency of
    # Our grid search by dividing our search into two calls to GridSearchCV--one for
    # linear kernel and one for rbf kernel.
    if 'kernel' in np.array(list(map(lambda x : str.split(x, '__'), list(params.keys()))))[:,1]:
        prefix = np.array(list(map(lambda x : str.split(x, '__'), list(params.keys()))))[0,0]
        key = prefix + '__kernel'
        kernel_list = params[key]
        if ('linear' in kernel_list) and 'rbf' in kernel_list:


            # for linear kernel, don't vary over gamma
            params_linear = params.copy()
            params_linear[prefix + '__gamma'] = [0]
            params_linear[prefix + '__kernel'] = ['linear']

            # for rbf kernel, vary over both C and gamma
            params_rbf = params.copy()
            params_rbf[prefix + '__kernel'] = ['rbf']
        
            # gridsearch linear kernel
            gs = GridSearchCV(estimator=pl, param_grid=params_linear, scoring=scorers,
                              cv=LeaveOneGroupOut(), refit=False, verbose=1, n_jobs=-1)
            gs.fit(X_train, y_train, groups=val_split)
            res_linear = pd.DataFrame(gs.cv_results_)


            # gridsearch rbf kernel
            gs = GridSearchCV(estimator=pl, param_grid=params_rbf, scoring=scorers,
                              cv=LeaveOneGroupOut(), refit=False, verbose=1, n_jobs=-1)

            gs.fit(X_train, y_train, groups=val_split)
            res_rbf = pd.DataFrame(gs.cv_results_)

            # combine results
            res = pd.concat([res_linear, res_rbf],ignore_index=True)

        else:
            gs = GridSearchCV(estimator=pl, param_grid=params, scoring=scorers,
                              cv=LeaveOneGroupOut(), refit=False, verbose=1, n_jobs=-1)
            gs.fit(X_train, y_train, groups=val_split)
            res = pd.DataFrame(gs.cv_results_)
    else:
        gs = GridSearchCV(estimator=pl, param_grid=params, scoring=scorers,
                          cv=LeaveOneGroupOut(), refit=False, verbose=1, n_jobs=-1)
        gs.fit(X_train, y_train, groups=val_split)
        res = pd.DataFrame(gs.cv_results_)
        
    return res



def main():
    args = parse_args()
    model = args.model
    dataset = args.dataset
    data_dir = args.data_dir
    hp_spec = args.hp_spec
    results_dir = args.results_dir
    job_name = args.job_name
    num_samples = args.num_samples
    epochs = args.epochs
    redis_password = args.redis_password
    gpus_per_node = args.gpus_per_node
    nn_search_alg = args.nn_search_alg
    local = args.local
    problem = args.problem
    
    # Run the appropriate hyperparameter tuning job
    if model == 'svm' or model == 'rf':
        data = data_utils.load_dataset(data_dir,
                                       dataset,
                                       problem,
                                       'train',
                                       val_col='cv_splits')
        X_train, y_train, val_split = data['X'], data['y'], data['val_split']
        res = gridsearch_sklearn(X_train,
                                 y_train,
                                 val_split,
                                 problem,
                                 model,
                                 hp_spec,
                                 job_name)

    if (model == 'mlp') or (model == 'unet'):
        res = ray_tune(model,
                       data_dir,
                       dataset,
                       problem,
                       hp_spec,
                       job_name,
                       results_dir,
                       num_samples,
                       epochs,
                       redis_password=redis_password,
                       gpus_per_node=gpus_per_node,
                       search_alg_abrv=nn_search_alg,
                       local=local)

    if local:
        now_str = datetime.now().strftime("%Y%m%d-%H%M%S") 
        res_name = job_name + '-' + now_str + '.csv'
    else:
        job_id = os.environ['SLURM_JOB_ID']
        res_name = job_name + '-job' + job_id + '.csv'
    res.to_csv(os.path.join(results_dir, res_name))


if __name__ == "__main__":
    main()
