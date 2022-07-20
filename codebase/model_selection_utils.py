import pandas as pd
import numpy as np
from IPython.display import display, HTML
from datetime import datetime
from ast import literal_eval
from sklearn.ensemble import RandomForestClassifier
import itertools
import subprocess
import os
import json
import joblib

def try_literal_eval(s):
    """
    Evaluates string as python code
    
    Parameters
    ----------
    s : string
        A string to evaluate as python code
        
    Returns
    -------
    Either evaluation of s as python code or s if this raises an error.
    """
    try:
        return literal_eval(s)
    except ValueError:
        return s
    
def load_gs_results(path):
    """
    Load pd dataframe with formatted gridsearch results from a csv file
    containing grid search results. Converts params column from 
    string format, as read from csv, to Python dictionary objects.
    
    Parameters
    ----------
    path : string
        filepath to csv containing gridsearch results
        
    Returns
    -------
    gs : pd.DataFrame
        dataframe containing formatted gridsearch results
    """
    gs = pd.read_csv(path)
    gs = gs.drop('Unnamed: 0', axis=1)
    gs = gs.applymap(try_literal_eval)
    return gs

def print_top_n_hyperparams(cv_results, metrics, n=5):
    """
    Displays in ipython notebook the top n scores and hyperparameter settings
    for each metric in the provided metrics

    Parameters
    ----------
    cv_results : pd.DataFrame
        cv_results_ object from sklearn's GridSearchCV as a DataFrame
    metrics : list of strings
        list of metrics
    n : int
        number of hyperparameter settings to display for each metric
    """
    metrics_stds = [('mean_test_' + metric, 'std_test_' + metric) for metric in metrics]
    # flatten metrics_stds
    metrics_stds = list(itertools.chain.from_iterable(metrics_stds))

    # the metrics sort results by
    metrics = ['mean_test_' + metric for metric in metrics]

    params = cv_results['params'].iloc[0].keys()
    params = ['param_' + param for param in params]
    for metric in metrics:
        print(f"Top 5 param settings {metric}:")
        top_5 = cv_results.sort_values(by=metric)[::-1][:n]
        display(top_5[metrics_stds + params])
        print('\n')


def avg_depth(rf):
    """
    Returns the average depth of individual base estimators in a trained
    random forest classifier

    Parameters
    ----------
    rf : scikit learn RandomForestClassifier
        Trained random forest classifier

    Returns
    -------
    avg_depth : int
        Average depth of individual base estimators in rf
    
    """
    avg_depth = np.mean([estimator.get_depth() for estimator in rf.estimators_]) 
    return avg_depth

def avg_depths(X, y, max_depths_list, min_samples_splits_list, min_samples_leafs_list, max_features_list):
    """
    Returns dataframe containing the average depth of base learners for random
    forest trained over the parameter space created by lists of hyperparameter
    values in parameters.

    Parameters
    ----------
    X : pandas dataframe
        The feature matrix used to train the random forest classifiers
    
    y : pandas series
        The labels used to train the random forest classifiers 

    max_depths_list : list of ints
        List of max_depths values

    min_samples_splits_list : list of ints
        List of min samples split values

    min_samples_leafs_list : list of ints
        List of min samples leaf values

    max_features_list : list of ints
        List of max feature values

    Returns
    -------
    dataframe containing the average depth of base learners for random forest
    classifiers trained over all combonations of rf hyperparameters passed as
    parameters to this function
    """
    max_depths = []
    min_samples_splits = []
    min_samples_leafs = []
    max_featuress = []
    avg_depths = []


    for max_depth in max_depths_list:
        for min_samples_split in min_samples_splits_list:
            for min_samples_leaf in min_samples_leafs_list:
                for max_features in max_features_list:
                    rf = RandomForestClassifier(max_depth=max_depth, min_samples_split=min_samples_split,
                                                min_samples_leaf=min_samples_leaf, n_estimators=200,
                                                max_features=max_features, random_state=1, n_jobs=-1)
                    rf.fit(X, y)


                    max_depths.append(max_depth)
                    min_samples_splits.append(min_samples_split)
                    min_samples_leafs.append(min_samples_leaf)
                    max_featuress.append(max_features)
                    avg_depths.append(avg_depth(rf))

    return pd.DataFrame(data=np.array([max_depths,
                                       min_samples_splits,
                                       min_samples_leafs,
                                       max_featuress,
                                       avg_depths]).T,
                        columns=['max_depth',
                                 'min_samples_split',
                                 'min_samples_leaf',
                                 'max_features',
                                 'avg_depth']).sort_values('avg_depth')


def gs_results_file_path(prefix, base_path='./Grid_Search_Results'):
    """
    Generate a unique file path in form {base_path}/{prefix}_{timestamp}.csv
    for grid search results so that results csv is never accidentally 
    overwritten when grid search is run multiple times

    Parameters
    ----------
    prefix : string
        Prefix for file name
    base_path : string
        Path to directory where file should be saved

    Returns
    -------
    file_path : string
       Path under which the grid search results will be saved
    """
    now = datetime.now()
    now_str = now.strftime("%m_%d_%Y_%H_%M_%S")
    file_path = base_path + '/' + prefix + '_' + now_str + '.csv' 
    return file_path



def build_output_file_structure(model,
                                 out_dir,
                                 dataset,
                                 problem):
    """
    Helper function to ms_on_hpc and launch_nn_ray_tune. Creates
    the file structure for the directory where output will be saved.
    Returns jobname.
    
    Parameters:
    -----------

    model : string
        The model to run gridsearch for.

    out_dir : string
        Path to directory to hold output.

    dataset : string
        The dataset to train on.

    problem : string
        The classification problem the model is being trained for.

    ms_round : int
        A number indicating the current round of grid search. This is used in
        automatically generated file names.

    job_name_suffix : string
        Optional argument for a suffix to append to the end of filenames.

    Return:
    -------
    The job name to be used for the SLURM job.
    
    """
    hp_spec_dir = os.path.join(out_dir, dataset, problem, model, "hp_specs")
    log_dir = os.path.join(out_dir, dataset, problem, model, "logs")
    results_dir = os.path.join(out_dir, dataset, problem, model, "results")

    os.makedirs(hp_spec_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    return (hp_spec_dir, log_dir, results_dir)



def build_job_name(model,
                   dataset,
                   ms_round,
                   job_name_suffix=None):
    """
    Helper function to ms_on_hpc and launch_nn_ray_tune. Generates job name
    for SLURM job.
    
    Parameters:
    -----------

    model : string
        The model to run gridsearch for.

    dataset : string
        The dataset to train on.

    ms_round : int
        A number indicating the current round of grid search. This is used in
        automatically generated file names.

    job_name_suffix : string
        Optional argument for a suffix to append to the end of filenames.

    Return:
    -------
    The job name to be used for the SLURM job.
    
    """
    if dataset == '17_segment':
        ds_abrv = '17s' 
    elif dataset == '3_vessel':
        ds_abrv = '3v'
    elif dataset == 'polar_plot':
        ds_abrv = 'pp'
    else:
        raise NotImplementedError(f"{dataset} is invalid option for dataset")
    job_name = model + '_' + ds_abrv + '_r' + str(ms_round)
    if job_name_suffix:
        job_name = job_name + '_' + job_name_suffix 
    
    return job_name


def ms_on_hpc(model, 
              data_dir,
              out_dir,
              hp_spec,
              dataset,
              problem,
              ms_round,
              job_name_suffix=None,
              mem='10GB',
              time=None,
              cpus=None,
              local=False):
    """
    Launches a gridsearch job on HPC for SVM or RF. This function will create 
    various directories for any directory path passed as an argument that does
    not exist at the time the function is run. The function will create files 
    for: the result of the grid search; the dictionary of hyperparameter values
    to be used in the grid search (as a JSON file); the log file for the SLURM
    job; and the environment variables for the SLURM job. This function is an 
    interface for the command line program run_ms_on_hpc.py. 

    Parameters
    ----------
    model : string
        The model to run gridsearch for. Valid options are 'svm' or 'rf'

    data_dir : string
        Path to directory contianing data.

    out_dir : string
        Path to directory to hold output. If this is an empty directory or if
        the directory does not exist, the directory and subdirectories will
        be created according to organization outlined in README for hpc_scripts.
        Output will include slurm logs, hyperparameter specification file,
        and results files.

    hp_spec : python dictionary
        The dictionary of hyperparameters specifying which hyperparameters and values 
        to gridsearch over. This is in the format of the param_grid argument
        of scikit learn's model_selection.GridSearchCV 
    
    dataset : string
        The dataset to train on. Valid options are '3_vessel' or '17_segment'

    problem : string
        The classification problem the model is being trained for. Currently
        only valid option is 'norm_abn'.

    ms_round : int
        A number indicating the current round of grid search. This is used in
        automatically generated file names.

    job_name_suffix : string
        Optional argument for a suffix to append to the end of filenames.

    mem : string
        Optional argument specifying the amount of memory to request for the
        SLURM job. The string should be formatted as '{amount}GB'.
        Defaut value is '10GB'.

    time : string
        Optional argument specifying the time limit for the SLURM job. The
        string should be formatted as '{hours}:{minutes}:{seconds}'. Default
        value is '05:00'. Requesting a lower time limit will give your job
        higher priority in the queue, but a time limit that is too low could
        cause the job to terminate before it is complete.

    cpus : int
        The number of cpus to request for the SLURM job. Default value is 1. 

    local : bool
        If true, ms_on_hpc will run the model selection operation locally
        rather than launch a job with SLURM.
    
    Returns
    -------
        This function returns None. 

        If there is no filename_suffix passed, job_name will be:
            {model}_{dataset abreviation}_r(ms_round}
        otherwise jobname will be:
            {model}_{dataset abreviation}_r(ms_round}_{optional filename_suffix}

        If any directory path that is passed to this funciton does not exist at
        the time it is run, that directory will be created. This function will
        create the following files:
    
        
        The results of the gridsearch: 
            - {grid_search_results_dir}/{jobname}_{timestamp}.csv
        
        JSON containing dictionary passed as params:
            - {param_dir}/r{ms_round}_params.json

        SLURM log files:
            - {log_dir}/{username}-{jobname}-job{SLURM job num}.out

        SLURM environment variables:
            - {log_dir}/slurm_{jobname}.txt
    """
    # validate these arguments so no incorrect directories are created
    if dataset not in ['3_vessel', '17_segment']:
        raise ValueError("dataset must be '3_vessel' or '17_segment'")
    if problem not in ['norm_abn', 'localization']:
        raise ValueError("problem must be 'norm_abn' or 'localization'")
    if model not in ['svm', 'rf']:
        raise ValueError("model must be one of 'svm', 'rf'")

    base_dir = os.getcwd().split('e-emagin-pet')[0] + 'e-emagin-pet/'
    hpc_launcher_path = base_dir + "codebase/hpc_scripts/run_ms_on_hpc.py"

    # build path to log dir, path to hp_spec dir, path to results dir    
    hp_spec_dir, log_dir, results_dir = build_output_file_structure(model, out_dir, dataset, problem)
    job_name = build_job_name(model, dataset, ms_round, job_name_suffix)

    hp_spec_name = f"{job_name}_hp_spec.json"
    hp_spec_path = os.path.join(hp_spec_dir, hp_spec_name)
    
    #convert all entries in hp_spec dict to lists of hyperparam values
    for key in hp_spec.keys():
        if type(hp_spec[key]) != list:
            hp_spec[key] = list(hp_spec[key])
    
    # save as json in hp_spec_path
    with open(hp_spec_path, 'w') as outfile:
        json.dump(hp_spec, outfile)

    cmd = f"python {hpc_launcher_path} --model {model} --problem {problem} --data_dir {data_dir} --dataset {dataset} --hp_spec {hp_spec_path} --job_name {job_name}"
    if local:
        cmd = cmd + " --local"
    optional_params = [log_dir, results_dir, mem, time, cpus]
    optional_param_flag = ['--log_dir', '--results_dir', '--mem', '--time', 
                           '--cpus_per_task']
    
    # build command with optional params
    for param, flag in zip(optional_params, optional_param_flag):
        if param:
            cmd = cmd + " " + flag + " " + str(param)

    # Run command line program
    proc = subprocess.Popen(cmd,                                                       
        shell=True,                                                                    
        stdout = subprocess.PIPE,                                                      
        stderr = subprocess.PIPE,                                                      
    )                                                                                  
    stdout, stderr = proc.communicate()                                                
    print(stdout.decode("utf-8"))                                                      
    print(stderr.decode("utf-8"))



def launch_nn_ray_tune_hpc(model,
                           data_dir,
                           out_dir,
                           hp_spec,
                           dataset,                                         
                           problem,
                           ms_round,                                        
                           nn_search_alg='ho',
                           num_samples=10,                                 
                           epochs=50,                                       
                           job_name_suffix=None,                            
                           mem='10GB',                                      
                           time=None,                                       
                           cpus=1,                                          
                           gpus=0,
                           local=False):
    """
    Parameters
    ----------
    model : string
        The model to run gridsearch for. Currently, only valid option is 'mlp'

    data_dir : string
        Path to directory contianing data.

    out_dir : string
        Path to directory to hold output. If this is an empty directory or if
        the directory does not exist, the directory and subdirectories will
        be created according to organization outlined in README for hpc_scripts.
        Output will include slurm logs, hyperparameter specification file,
        and results files.

    hp_spec : dictionary
        TODO: Specify space formats. Different for gs and ho

    dataset : string
        The dataset to train on. Valid options are '3_vessel' or '17_segment'

    problem : string
        The classification problem the model is being trained for. Currently
        only valid option is 'norm_abn'.

    ms_round : int
        A number indicating the current round of grid search. This is used in
        automatically generated file names.

    nn_search_alg : string
        The algorithm to use for hyperparmeter tuning for a neural network. 
        Valid options include 'ho' for HyperOpt and 'gs' for grid search.
    
    num_samples : int
        Optional. When nn_search_alg is 'ho, num samples is the number of 
        trials to run in Bayesian optimization.  When nn_search_alg is 'gs' 
        this is the number of times to resample the gridsearch (i.e. 
        num_samples=2 will run the gridsearch twice).  This can be helpful as 
        random initializations neural networks can alter results.  

    epochs : int

    job_name_suffix : string
        Optional argument for a suffix to append to the end of filenames.

    mem : string
        Optional argument specifying the amount of memory to request for the
        SLURM job. The string should be formatted as '{amount}GB'.
        Defaut value is '10GB'.

    time : string
        Optional argument specifying the time limit for the SLURM job. The
        string should be formatted as '{hours}:{minutes}:{seconds}'. Default
        value is '05:00'. Requesting a lower time limit will give your job
        higher priority in the queue, but a time limit that is too low could
        cause the job to terminate before it is complete.

    cpus : int
        The number of cpus to request for the SLURM job. Default value is 1. 
    
    gpus : int 
        The number of gpus to request for the SLURM job. Default value is 1. 

    local : bool
        If true, ms_on_hpc will run the model selection operation locally
        rather than launch a job with SLURM.

    Returns
    -------
    None

        If there is no filename_suffix passed, job_name will be:
            {model}_{dataset abreviation}_r(ms_round}
        otherwise jobname will be:
            {model}_{dataset abreviation}_r(ms_round}_{optional filename_suffix}

    """
    # validate these arguments so no incorrect directories are created
    if dataset not in ['3_vessel', '17_segment', 'polar_plot']:
        raise ValueError("dataset must be '3_vessel', '17_segment', or 'polar_plot'")
    if problem not in ['norm_abn', 'localization']:
        raise ValueError("problem must be 'norm_abn' or 'localization'")
    if model not in ['mlp', 'unet']:
        raise ValueError("model must be 'mlp' or 'unet'")

    base_dir = os.getcwd().split('e-emagin-pet')[0] + 'e-emagin-pet/'
    hpc_launcher_path = base_dir + "codebase/hpc_scripts/run_ms_on_hpc.py"

    # build path to log dir, path to hp_spec dir, path to results dir    
    hp_spec_dir, log_dir, results_dir = build_output_file_structure(model, out_dir, dataset, problem)
    job_name = build_job_name(model, dataset, ms_round, job_name_suffix)

    hp_spec_name = f"{job_name}_hp_spec.joblib"
    hp_spec_path = os.path.join(hp_spec_dir, hp_spec_name)

    # save as joblib in hp_spec_path
    joblib.dump(hp_spec, hp_spec_path)

    cmd = f"python {hpc_launcher_path}"
                                                                                
    required_params = [model, problem, dataset, data_dir, hp_spec_path, job_name, 
                       log_dir, results_dir]
    required_param_flag = ['--model', '--problem', '--dataset', '--data_dir', '--hp_spec',
                           '--job_name', '--log_dir', '--results_dir']
    
    # build command string with required params
    for param, flag in zip(required_params, required_param_flag):               
        cmd = cmd + " " + flag + " " + str(param)                           

    optional_params = [mem,  time, cpus, gpus, num_samples, epochs,
                       nn_search_alg]
    optional_param_flag = ['--mem', '--time', '--cpus_per_task',
                           '--gpus_per_node', '--num_samples', '--epochs', 
                           '--nn_search_alg'] 

    # build command string with optional params
    for param, flag in zip(optional_params, optional_param_flag):               
        if param:                                                               
            cmd = cmd + " " + flag + " " + str(param)                           
 
    if local:
        cmd = cmd + ' --local'
    # Run command line program                                                  
    proc = subprocess.Popen(cmd,                                                       
        shell=True,                                                                    
        stdout = subprocess.PIPE,                                                      
        stderr = subprocess.PIPE,                                                      
    )                                                                                  
    stdout, stderr = proc.communicate()                                                
    print(stdout.decode("utf-8"))                                                      
    print(stderr.decode("utf-8"))
