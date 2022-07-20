"""
    This script trains and saves the SVM, RF, MLP, and U-Net with the top
    hyperparameters identified in the model selection notebook for the models
    trained on the 17 segment datasets on the detection problem

    Tensorboard logs are created for the final MLP and U-Net.
    This script assumes the output directory structure already exists.
"""
import sys
import os
BASE_DIR = os.getcwd().split('e-emagin-pet-export1')[0] + 'e-emagin-pet-export1/'
sys.path.append(BASE_DIR)

from codebase import train_final_models_utils as fm
from codebase import data_utils
from codebase.Ensemble import Ensemble
from codebase.ModelEvalWrapper import ModelEvalWrapper
from joblib import load, dump
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser(description="""Create the stratified datasets
                                            for 17 segment localization, norm/abn
                                            and 3 vessel norm/abn outcomes.""")
    parser.add_argument('--datadir', type=str,
                        help="""Path to data directory""")
    parser.add_argument('--outdir', type=str,
                        help="""Path to directory to where model training output will
                                be saved""")
    parser.add_argument('--saved_models_path', type=str,
                        help="""Path to directory where final models will be saved.""")
    parser.add_argument('--hyperparam_path', type=str,
                        help="""Path to json file contianing hyperparameter settings
                                for svm, rf, mlp, and unet. Format should match
                                the example json files in config_files""")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    datadir = args.datadir
    outdir = args.outdir
    saved_models_base_path = args.saved_models_path
    hyperparameters_json_path = args.hyperparam_path

    with open(hyperparameters_json_path, 'r') as fp:
        hp = json.load(fp) 
    
    svm_hyperparams = hp['svm']
    rf_hyperparams = hp['rf'] 
    mlp_hyperparams = hp['mlp'] 
    unet_hyperparams = hp['unet'] 

#     # top SVM hyperparameters identified in model selection notebook
#     svm_hyperparams = {'kernel': 'rbf',
#                        'C': 1428.284902,
#                        'gamma': 0.001496}
# 
#     # top RF hyperparameters identified in model selection notebook
#     rf_hyperparams = {'n_estimators': 300,
#                       'max_depth': 31,
#                       'min_samples_split': 2,
#                       'min_samples_leaf': 1,
#                       'max_features': 0.67,
#                       'random_state': 0}
# 
#     # top MLP hyperparameters identified in model selection notebook
#     mlp_hyperparams = {'epochs': 450,
#                        'checkpoint_metric': 'val_auc',
#                        'hidden_sizes': [256, 288, 40],
#                        'learning_rate': .001,
#                        'drop_prob': 0.0,
#                        'reg': 0.0} 
# 
#     # top UNET hyperparameters identified in model selection notebook (DF)
#     unet_hyperparams = {'epochs': 500,
#                         'checkpoint_metric': 'val_auc',
#                         'f': 32,
#                         'k': 8,
#                         'l': 2,
#                         'fc_hidden_sizes': [32, 64],
#                         'learning_rate': 7e-05,
#                         'drop_prob': 0.25,
#                         'reg': 500}


    num_trainings = 11
    for i in range(0, num_trainings):
        if i == 0:
            fm.save_models(datadir=datadir,
                           outdir=outdir,
                           dataset='17_segment',
                           problem='norm_abn',
                           svm_hyperparams=svm_hyperparams,
                           rf_hyperparams=rf_hyperparams,
                           mlp_hyperparams=mlp_hyperparams,
                           unet_hyperparams=unet_hyperparams,
                           dummy=True,
                           db_method='youdens',
                           recall_target=None,
                           saved_models_path=os.path.join(saved_models_base_path,
                                                          f"models{i}"))

        else:
            fm.save_models(datadir=datadir,
                           outdir=outdir,
                           dataset='17_segment',
                           problem='norm_abn',
                           mlp_hyperparams=mlp_hyperparams,
                           unet_hyperparams=unet_hyperparams,
                           dummy=False,
                           db_method='youdens',
                           recall_target=None,
                           saved_models_path=os.path.join(saved_models_base_path,
                                                          f"models{i}"))

    # Save Ensembles
    os.makedirs(os.path.join(saved_models_base_path,
                             "ensembles"),
                exist_ok=True) 

    model_types = ['mlp', 'unet']
    model_name_abrvs = ['MLP', 'UNET']
    model_names = ['Multilayer Percetpron Ensemble',
                   'U-Net Ensemble']
    datasets = ['17_segment', 'polar_plot']

    for model_type, model_name_abrv, model_name, dataset in zip(model_types,
                                                                model_name_abrvs,
                                                                model_names,
                                                                datasets):
        models = []
        for i in range(0, num_trainings):
            models.append(load(os.path.join(saved_models_base_path,
                                            f"models{i}/{model_type}.joblib")))
        ensemble = Ensemble(models)
        ensemble = ModelEvalWrapper(model=ensemble,
                                    model_name=model_name,
                                    model_name_abrv=model_name_abrv,
                                    model_type='sklearn',
                                    dataset=dataset,
                                    decision_boundries=None,
                                    feature_cols=None)
        
        # Load data to calculate youden's index
        data = data_utils.load_dataset(datadir,
                                       dataset,
                                       'norm_abn',
                                       'train',
                                       'nn_val_split')

        X, y, val_split = data['X'], data['y'], data['val_split']
        X_train, y_train = X[val_split == 0], y[val_split == 0]
        X_val, y_val = X[val_split == 1], y[val_split == 1]

        # Calculate decision boundries with youden's index on validation set
        # for hyperparameter tuning
        decision_boundry = ensemble.calc_decision_boundries(X_val, y_val, method='youdens')
        decision_boundries = [decision_boundry]
        
        # Reload to clear loaded keras models from base ModelEvalWrappers
        # to avoid pickeling error
        models = []
        for i in range(0, num_trainings):
            models.append(load(os.path.join(saved_models_base_path,
                                            f"models{i}/{model_type}.joblib")))
        ensemble = Ensemble(models)
        ensemble = ModelEvalWrapper(model=ensemble,
                                    model_name=model_name,
                                    model_name_abrv=model_name_abrv,
                                    model_type='sklearn',
                                    dataset=dataset,
                                    decision_boundries=decision_boundries,
                                    feature_cols=None)

        dump(ensemble, os.path.join(saved_models_base_path,
                                    f"ensembles/{model_type}_e.joblib"))



if __name__ == "__main__":
    main()
