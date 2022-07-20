import sys
import os
BASE_DIR = os.getcwd().split('e-emagin-pet-export1')[0] + 'e-emagin-pet-export1/'
sys.path.append(BASE_DIR)

import numpy as np
import pandas as pd
from codebase import data_utils as du
from codebase import split_utils
import argparse

"""
This script creates stratified datasets for 17 segment per-patient detection
and per-vessel localizaiton and classification of abnormality type. Data is stratified
by outcome, and all studies for a given patient fall within the same split. This
script splits the full formatted dataset into training data, validation data 
for hyperparameter tuning, validation data for model comparison, and test data.
This script will save the split datasets in the following directory structure.

data
    3_vessel
        norm_abn
            3v_norm_abn_train.csv
            3v_norm_abn_val.csv
            3v_norm_abn_test.csv
            3v_norm_abn_full.csv
    17_segment
        norm_abn
            17s_norm_abn_train.csv
            17s_norm_abn_val.csv
            17s_norm_abn_test.csv
            17s_norm_abn_full.csv
        localization
            17s_loc_train.csv
            17s_loc_val.csv
            17s_loc_test.csv
            17s_loc_full.csv

If script does not converge, try increasing max_class_dev. See argument description
with --help. 

"""



def parse_args():
    parser = argparse.ArgumentParser(description="""Create the stratified datasets
                                            for 17 segment localization, norm/abn
                                            and 3 vessel norm/abn outcomes.""")
    parser.add_argument('--outdir', type=str,
                        help="""Base path to directory that will containe the
                              datasets. Will be created if it does not exist""")
    parser.add_argument('--in_datapath', type=str,
                        help="""Path to formatted data csv with exclusions applied.
                                This should contain all feature columns for 17 segment
                                and 3 vessel datasets; it should contain all target columns
                                for detection and localization problems; it should
                                contain study_no and pt_id columns.""")
    parser.add_argument('--max_class_dev', type=float, default=0.009,
                        help="""For any class in any given split, the maximum that the proportion of
                                class / total in split can deviate from the respective proportion of
                                class / total in the whole dataset. A smaller fraction means the resulting
                                class distribution in splits will be more closely reflective of the
                                overall class distribution, but the algorithm may take longer to / never
                                converge. A larger fraction means the resulting class distribution in 
                                splits will be more approximate in reflecting the overall class distribution,
                                and the splitting algorithm will more easily converge. Depending on the 
                                total size of the dataset, the number of splits, and the number of 
                                repeat studies from the same patient, and the distribution of classes, 
                                the algorithm may not converge to find patient wise splits that are
                                stratified by class.""")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    outdir = args.outdir
    datapath = args.in_datapath
    max_class_dev = args.max_class_dev

    
    # Build column and feature lists
    regions = ['basal_anterior', 'basal_anteroseptal', 'basal_inferoseptal',    
               'basal_inferior', 'basal_inferolateral', 'basal_anterolateral',    
               'mid_anterior', 'mid_anteroseptal', 'mid_inferoseptal',          
               'mid_inferior', 'mid_inferolateral', 'mid_anterolateral',        
               'apical_anterior', 'apical_septal', 'apical_inferior',           
               'apical_lateral', 'apex']                                        

    rest_cols = ['rest_' + reg for reg in regions]                              
    stress_cols = ['stress_' + reg for reg in regions]                          
    reserve_cols = ['reserve_' + reg for reg in regions]                        
    difference_cols = ['difference_' + reg for reg in regions]  

    feature_cols_17s = rest_cols + stress_cols + reserve_cols + difference_cols

    vessel_territories = ['lad', 'rca', 'lcx']                              
    rest_cols = ['rest_' + ves for ves in vessel_territories]               
    stress_cols = ['stress_' + ves for ves in vessel_territories]           
    reserve_cols = ['reserve_' + ves for ves in vessel_territories]         
    difference_cols = ['difference_' + ves for ves in vessel_territories]

    feature_cols_3v = rest_cols + stress_cols + reserve_cols + difference_cols

    target_cols_loc = ['scar_lad', 'scar_rca', 'scar_lcx', 'ischemia_lad', 'ischemia_rca', 'ischemia_lcx'] 
    target_cols_detection = ['abnormal']



    # Load formatted data with exclusions applied including all feature cols, all target cols, pt_id, and study_no
    pet = pd.read_csv(datapath)

    unique_class = list(map(lambda x: tuple(x), pet[['scar_lad',             
                                                     'scar_rca',                 
                                                     'scar_lcx',                 
                                                     'ischemia_lad',             
                                                     'ischemia_rca',             
                                                     'ischemia_lcx']].values))

    ids = pd.DataFrame({'unique_class': unique_class,
        'pt_id': pet['pt_id'],
        'study_no': pet['study_no']})


    class_cnts = ids.unique_class.value_counts()                             

    # # Group all of the least frequently appearing unique classes together       
    # # All that appear less than 10 times
    infrequent_classes = class_cnts[class_cnts < 10].index                      
    infreq_series = ids['unique_class'].isin(infrequent_classes)             
    ids.loc[infreq_series, 'unique_class'] = -1


    # train, val_for_model_comp, test
    stratified_splits = split_utils.make_stratified_splits(
        [80, 10, 10],
        ids,
        'unique_class',
        'pt_id',
        max_class_dev=max_class_dev,
        seed=3)

    train = stratified_splits[stratified_splits.split == 0]                     
    cv_splits = split_utils.make_stratified_splits(
        [10] * 10,
        train,
        'unique_class',
        'pt_id',
        max_class_dev=max_class_dev,
        seed=3)
    nn_val_split = split_utils.make_stratified_splits(
        [87.5, 12.5],
        train,
        'unique_class',
        'pt_id',
        max_class_dev=max_class_dev,
        seed=3)


    split_utils.make_datasets(data=pet,
                  stratified_splits=stratified_splits,
                  cv_splits=cv_splits,
                  nn_val_split=nn_val_split,
                  dataset='3_vessel',
                  problem='norm_abn',
                  data_dir=outdir,
                  feature_columns=feature_cols_3v,
                  output_columns=target_cols_detection)

    split_utils.make_datasets(data=pet,
                  stratified_splits=stratified_splits,
                  cv_splits=cv_splits,
                  nn_val_split=nn_val_split,
                  dataset='17_segment',
                  problem='localization',
                  data_dir=outdir,
                  feature_columns=feature_cols_17s,
                  output_columns=target_cols_loc)

    split_utils.make_datasets(data=pet,
                  stratified_splits=stratified_splits,
                  cv_splits=cv_splits,
                  nn_val_split=nn_val_split,
                  dataset='17_segment',
                  problem='norm_abn',
                  data_dir=outdir,
                  feature_columns=feature_cols_17s,
                  output_columns=target_cols_detection)

if __name__ == "__main__":
    main()
