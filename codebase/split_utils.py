import sys
import os
BASE_DIR = os.getcwd().split('e-emagin-pet')[0] + 'e-emagin-pet/'
sys.path.append(BASE_DIR)

import numpy as np
import pandas as pd
from codebase import data_utils as du


def make_stratified_splits(split_scheme, data, label_col, pt_id_col, max_class_dev=0.009, seed=0):
    """
    Returns data stratified according to split_scheme. Data will be stratified
    patient-wise and by class frequency (i.e. observations from the same patient
    will not appear in different splits, and each split will have the same proportion
    of observations from each class as does the full dataset. Returned dataset
    will have the additional int column 'split' denoting which split,
    corresponding to index of split_scheme, that a given row belongs to.
    
    split_scheme : list of ints
        Each int represents the percentage for that split. I.e. [80, 10, 10]
        will make 3 splits, the first 80%, second and third 10%
    
    data : pd.DataFrame
        Dataset to be split into stratified patient wise splits. Should contain
        a column containing the label for each row (the name of this col will be
        passed in label_col). Should contain a column with the pt_ids 
        corresponding with each observation (the name of this col will be passed
        in pt_id_col).

    label_col : string
        Name of the column containing class labels

    pt_id_col : string
        Name of the column containing patient_ids corresponding with each observation

    max_class_dev : float
        For any class in any given split, the maximum that the proportion of
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
        stratified by class.
        
    seed : int
        Random seed to be used to make repeatable splits
    """
    np.random.seed(seed)
    data = data.copy()
    total_studies = data.shape[0]
    
    # add a column to indicate what split the row is in
    # -1 indicates has not been selected yet
    data['split'] = -1

    unique_classes = data[label_col].unique()
    class_proportions = data[label_col].value_counts() / total_studies

    # split_end_percents is the percentage of data corresponding to split i
    split_end_percents = split_scheme.copy()
    # Splits is the # of the split
    splits = list(range(0,len(split_scheme)))
    
    # max_dev_in_splits keeps track of the maximum deviation from the population class proportion
    # for each split
    max_dev_in_splits = np.array([np.inf] * len(split_scheme))
    
    # If this does not converge, try increasing the constant to make more lenient
    # the approximate stratification accross splits by outcome
#     while max_dev_in_splits.max() > 0.009:
    while max_dev_in_splits.max() > max_class_dev:
        # If not the first run, reset the 3 least balanced splits to -1 and try to resplit them
        if not (max_dev_in_splits == np.inf).any():
            splits = max_dev_in_splits.argsort()[::-1][:3]
            split_end_percents = [split_scheme[i] for i in splits]
            print('worst 3 splits', [max_dev_in_splits[i] for i in splits],
                  'worst 3 splits i', splits)
            
            # reset data to unsplit for those splits
            data.loc[data['split'].isin(splits), 'split'] = -1
    
        for i, (split_end_perc, split) in enumerate(zip(split_end_percents, splits)):
            if i == len(split_end_percents) - 1:
                data.loc[data['split'] == -1, 'split'] = split
                in_split_mask = data['split'] == split
                total_in_split = in_split_mask.astype(int).sum()
            else:
                total_in_split = 0

            class_proportions = data[label_col].value_counts() / total_studies
            target_counts = class_proportions * total_studies * (split_end_perc/100)
            dev_from_targ = dict(zip(list(target_counts.index),  list(-1 * target_counts)))
            target_counts = target_counts.to_dict()
            class_proportions = class_proportions.to_dict()


            while ((total_in_split < ((split_end_perc/100) * total_studies))
                   & ((data['split'] == -1).astype(int).sum() > 0)):

                # Keep track of whether you pick for each class in unique_classes where there are options left.
                # Mark as picked if no members of a class left.
                # Mark as picked if you sample for the class.
                # If there was nothing sampled for any class, then check if within certain percentage of target. 
                sampled = dict(zip(unique_classes, [False] * len(unique_classes)))
                hyp_dev_from_perc = {}
                hyp_dev_from_perc_remaining = {}
                pt_ids_dict = {}

                for class_i in unique_classes:
                    # Calculate hypothetical if you do add
                    pt_ids = data.loc[(data['split'] == -1) & (data[label_col] == class_i),
                                      pt_id_col].unique()

                    if (pt_ids.size > 0):
                        # add a person
                        perc_pick=0.4
                        num_samples = int(np.ceil(abs(perc_pick * (-1 * dev_from_targ[class_i]) + 1e-5)))
                        num_samples = abs(num_samples)
                        while num_samples > pt_ids.size:
                            num_samples -= 1

                        pt_id = np.random.choice(pt_ids, size=num_samples, replace=False)
                        data.loc[data[pt_id_col].isin(pt_id), 'split'] = split


                        diff_from_targ = abs(((split_end_perc/100) * total_studies) - total_in_split)
                        # update variables
                        in_split_mask = data['split'] == split
                        class_i_mask = data[label_col] == class_i
                        class_i_in_split_mask = in_split_mask & class_i_mask
                        total_in_split = in_split_mask.astype(int).sum()
                        total_class_i_in_split = (class_i_in_split_mask).astype(int).sum()
                        perc_class_i_in_split = total_class_i_in_split / total_in_split
                        dev_from_targ_i = total_class_i_in_split - target_counts[class_i]

                        hyp_dev_from_perc[class_i] = perc_class_i_in_split - class_proportions[class_i]

                        remaining = data[data['split'] == -1]
                        perc_remaining = remaining[remaining[label_col] == class_i].shape[0] / remaining.shape[0]
                        hyp_dev_from_perc_remaining[class_i] = perc_remaining - class_proportions[class_i]

                        pt_ids_dict[class_i] = pt_id

                        # adding another pushes over the cap
                        diff_from_targ_new = abs(((split_end_perc/100) * total_studies) - total_in_split)
                        
                        # if it goes further over than it was under, than you don't want to add it. otherwise you can add it. 
                        (diff_from_targ_new < diff_from_targ)
                        if (abs(dev_from_targ_i) < abs(dev_from_targ[class_i])) & (diff_from_targ_new < diff_from_targ):
                            dev_from_targ[class_i] = dev_from_targ_i
                            sampled[class_i] = True
                        else:
                            data.loc[data[pt_id_col].isin(pt_id), 'split'] = -1

                if not np.array(list(sampled.values())).any():
                    # stwitch to add the hypothetical scenario that will make it off by the smallest percent

                    # Choose from the class that the remainder of unassigned studies is most overweight with. 
                    hyp_dev_from_perc_remaining = pd.Series(hyp_dev_from_perc_remaining)
                    idx = hyp_dev_from_perc_remaining.argmax()
                    class_i = hyp_dev_from_perc_remaining.index[idx]
                    pt_id = pt_ids_dict[class_i]
                    data.loc[data[pt_id_col].isin(pt_id), 'split'] = split

                    # update variables
                    in_split_mask = data['split'] == split
                    class_i_mask = data[label_col] == class_i
                    class_i_in_split_mask = in_split_mask & class_i_mask
                    total_in_split = in_split_mask.astype(int).sum()
                    total_class_i_in_split = (class_i_in_split_mask).astype(int).sum()
                    dev_from_targ_i = total_class_i_in_split - target_counts[class_i]
                    dev_from_targ[class_i] = dev_from_targ_i

                # recalculate total_in_split for while statement
                in_split_mask = data['split'] == split
                total_in_split = in_split_mask.astype(int).sum()

            perc_in_split = data.loc[data['split'] == split, label_col].value_counts() / total_in_split
            max_dev_in_split = (perc_in_split - (data[label_col].value_counts() / total_studies)).abs().max()
            max_dev_in_splits[split] = max_dev_in_split

    return data




def make_dataset(data, dataset, stratified_splits=None, cv_splits=None, nn_val_split=None):
    """
    
    Parameters:
    -----------
    data : pd.DataFrame
        Contains features, outcome columns, and study_no column
    
    dataset : string
        Options include: 'train', 'val_for_model_comparison', 'test'
    
    stratified_splits : pd.DataFrame
        Dataframe with 'study_no', 'pt_id', and 'split' where split
        denotes the split that the given study belongs to out of train (split==0),
        val for model comp (split==1), and test (split==2).
        This argument is required when dataset is 'val_for_model_comparison' or 'test',
        otherwise, it should be None. 
    
    cv_splits : pd.DataFrame
        Dataframe with 'study_no', 'pt_id', outcome columns, and 'split' where split
        denotes the split that the study is in. Training data split int 10 for cv.
        This argument is required when dataset is 'train', otherwise, it should
        be None.
    
    nn_val_split : pd.DataFrame
        Dataframe with 'study_no', 'pt_id', outcome columns, and 'split' where split
        denotes the split that the study is in. Training data split in 2 for nn split.
        This argument is required when dataset is 'train', otherwise, it should
        be None.

    
    Returns:
    --------
    Dataframe with only included studies for the given split. If dataset is
    train, the returned dataframe will contain two additional columns:
    cv_splits and nn_val_split. cv_splits will contain integers 0-9 denoting
    a cross validation fold associated with each study. nn_val_split will contain
    integers 0, 1 where 0 denotes training and 1 denotes validation for hyperparameter
    tuning for the neural networks.
    
    """
    if dataset == 'train':
        assert((cv_splits.size != 0) & (nn_val_split.size != 0))

        data = data.merge(cv_splits[['study_no', 'split', 'pt_id']], 
                                        left_on=['study_no', 'pt_id'],
                                        right_on=['study_no', 'pt_id'])
        data = data.rename({'split': 'cv_splits'}, axis=1)
        data = data.merge(nn_val_split[['study_no', 'split', 'pt_id']], 
                                                      left_on=['study_no', 'pt_id'],
                                                      right_on=['study_no', 'pt_id'])
        data = data.rename({'split': 'nn_val_split'}, axis=1)
    elif dataset == 'val_for_model_comparison':
        included = stratified_splits.loc[stratified_splits['split'] == 1,
                                         'study_no']
        data = data.merge(included)
    elif dataset == 'test':
        included = stratified_splits.loc[stratified_splits['split'] == 2,
                                         'study_no']
        data = data.merge(included)
    return data

def make_datasets(data, stratified_splits, cv_splits, nn_val_split, dataset, problem, data_dir, feature_columns, output_columns):
    """
    Parameters:
    -----------
    data : pd.DataFrame
        Contains features, outcome columns, and study_no column
    
    stratified_splits : pd.DataFrame
        Dataframe with 'study_no', 'pt_id', and 'split' where split
        denotes the split that the given study belongs to out of train (split==0),
        val for model comp (split==1), and test (split==2).
    
    cv_splits : pd.DataFrame
        Dataframe with 'study_no', 'pt_id', outcome columns, and 'split' where split
        denotes the cross validation split that the study is in. Training data is split into 10
        folds for cross validation.
    
    nn_val_split : pd.DataFrame
        Dataframe with 'study_no', 'pt_id', outcome columns, and 'split' where split
        denotes the validation split for hyperparameter tuning that the study is in.
        A single validation split is used for nn_split.
    
    dataset : str
        Options include 17_segment and 3_vessel

    problem : str
        Options include norm_abn and localization 

    data_dir : str
        Path to output base directory that will hold all data files. This will
        be created if it does not already exist.

    feature_columns : list of str
        List of the feature columns to include in the dataset

    output_columns : list of str
        List of the output columns to include in the dataset

    Return:
    ------
    None
    """
#     print("data shape:",data[feature_columns + output_columns + ['study_no', 'pt_id']].copy().shape)
    data = data[feature_columns + output_columns + ['study_no', 'pt_id']].copy() 
    data_train = make_dataset(data,
                              'train',
                              stratified_splits=None, 
                              cv_splits=cv_splits,
                              nn_val_split=nn_val_split)
    data_val = make_dataset(data,
                            'val_for_model_comparison',
                            stratified_splits)
    data_test = make_dataset(data,
                             'test',
                             stratified_splits)
    
    file_path = os.path.join(data_dir, dataset, problem)
    
    # Save to csvs
    os.makedirs(file_path, exist_ok=True)
    file_name = du.data_filename(dataset, problem, 'train')
    data_train.to_csv(os.path.join(file_path, file_name), index=False)
    
    file_name = du.data_filename(dataset, problem, 'val')    
    data_val.to_csv(os.path.join(file_path, file_name), index=False)
    
    file_name = du.data_filename(dataset, problem, 'test')
    data_test.to_csv(os.path.join(file_path, file_name), index=False)
    
    file_name = du.data_filename(dataset, problem, 'full')
    data.to_csv(os.path.join(file_path, file_name), index=False)
