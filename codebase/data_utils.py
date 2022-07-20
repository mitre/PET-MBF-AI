import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import re


def polar_data(polar_data_path, study_nos):
    """
    Return the polar plot stress, rest, reserve, and difference measures for
    specified studies. This function assumes that both rest and stress files
    are present for each specified study. This function assumes that all
    images follow the naming convention STUDYNO_STUDYTYPE.csv where STUDYTYPE 
    is either 'rest' or 'stress'.
    
    polar_data_path : str
        Path to polar data directory
    
    study_nos : list of strings
        List of study numbers for which data will be returned
        
    Returns
    -------
    scans : np.array
        Array of image data in the format (N, H, W, 4) where N is the
        number of data instances. The last dimension is the image channel, and
        will contain, in order, stress_scan, rest_scan, reserve, difference.
        This array will keep the same order as study_nos--scans[i] will
        correspond to study_nos[i] for all i 
    
    """
    
    # Create the array before hand and fill it up at the corresponding index
    study_nos = list(study_nos)
    study_no_to_idx = dict(zip(study_nos, [i for i in range(0, len(study_nos))]))
    
    rest_filenames = pd.Series(study_nos).astype(str) + '_rest.csv'
    stress_filenames = pd.Series(study_nos).astype(str) + '_stress.csv'
    filenames = rest_filenames.append(stress_filenames)
    filenames = list(filenames.sort_values().values)
    
    # get shape of a single scan (must be square) for empty array
    scan = pd.read_csv(os.path.join(polar_data_path, rest_filenames[0]), header=None).values
    assert(scan.shape[0] == scan.shape[1])
    scan_dim = scan.shape[0]
    scans = np.empty(shape=(len(study_nos), 4, scan_dim, scan_dim), dtype=float)

    i = 0
    while i < len(filenames) - 1:
        filename_rest, filename_stress = filenames[i], filenames[i + 1]
        study_no_rest = int(str.split(filename_rest, '_')[0])
        study_no_stress = int(str.split(filename_stress, '_')[0])

        assert(str.split(filename_rest, '_')[1][:-4] == 'rest')
        assert(str.split(filename_stress, '_')[1][:-4] == 'stress')
        assert((study_no_rest == study_no_stress))
        
        study_no = study_no_rest
        rest = pd.read_csv(os.path.join(polar_data_path, filename_rest), header=None).values
        stress = pd.read_csv(os.path.join(polar_data_path, filename_stress), header=None).values
        
        assert(stress.shape == rest.shape)
        
        reserve = np.divide(stress, rest, out=np.zeros_like(stress, dtype=float), where=stress != 0)
        diff = stress - rest

        scans_i = np.array([stress, rest, reserve, diff])

        scans[study_no_to_idx[study_no]] = scans_i    
        i += 2

    return np.moveaxis(scans, 1, -1)


def data_filename(dataset, problem, split):
    """
    Return the file name of and file path to dir that contains
    dataset specified by parameters

    Parameters:
    -----------
    dataset : string
        Valid options include '3_vessel', '17_segment'
        
    problem : string
        Valid options include (currently) 'norm_abn'
        
    split : string
        Valid options include 'train', 'val', 'test', 'full'

    Returns: 
    --------
    (file_path, file_name)
    """
    if dataset == '3_vessel':
        ds_abv = '3v'
    elif dataset == '17_segment':
        ds_abv = '17s'
    else:
        raise NotImplementedError("dataset must be '3_vessel' or '17_segment'")

    filename = ds_abv + '_' + problem + '_' + split + '.csv'
    return filename

def load_dataset(data_dir, dataset, problem, split, val_col=None):
    """
    Structure of data must be:
        data:
            3_vessel:
                norm_abn:
                    3v_norm_abn_train.csv
                    3v_norm_abn_val.csv
                    3v_norm_abn_test.csv
            17_segment:
                norm_abn:
                    17s_norm_abn_train.csv
                    17s_norm_abn_test.csv
                    17s_norm_abn_val.csv
                abn_type:
                    17s_abn_type_train.csv
                    17s_abn_type_val.csv
                    17s_abn_type_test.csv
    
    Parameters
    ----------
    data_dir : string
        The directory holding the datasets
    
    dataset : string
        Valid options include '3_vessel', '17_segment', 'polar_plot'
        
    problem : string
        Valid options include 'norm_abn', 'localization'
        
    split : string
        Valid options include 'train', 'val', 'test'
        
    val_col : string
        Required only when split is 'train'. The column containing 
        validation split groups for hyperparameter tuning. Valid options
        include: 'cv_splits' and 'nn_val_split'
        
    returns
    -------
    {'X': feature_matrix, 'y': label_vector, 'val_split' : val_split,
     'study_no': study_no, 'pt_id': pt_id}
    val_col_vector will be None if not specified

    If 'X' is image data, it will be in format (N, H, W, C) where N is the
    number of data instances.
    
    """
    if split == 'train':
        assert(val_col)
    
    if dataset == 'polar_plot':
        # get 17 segment data to use labels and split cols for image data. 
        # tabular data will be disregarded
        filename = data_filename('17_segment', problem, split)
        file_path = os.path.join(data_dir, 'splits', '17_segment', problem, filename)
#         file_path = os.path.join(data_dir, '17_segment', problem, filename)
    else:
        filename = data_filename(dataset, problem, split)
        file_path = os.path.join(data_dir, 'splits', dataset, problem, filename)
#         file_path = os.path.join(data_dir, dataset, problem, filename)

    print(file_path)
    data = pd.read_csv(file_path)
    if problem == 'norm_abn':
        y = data['abnormal'].copy()
        data = data.drop('abnormal', axis=1)
    elif problem == 'localization':
        y = data[['scar_lad', 'scar_rca',
                  'scar_lcx', 'ischemia_lad',
                  'ischemia_rca', 'ischemia_lcx']]
        data = data.drop(['scar_lad', 'scar_rca',
                          'scar_lcx', 'ischemia_lad',
                          'ischemia_rca', 'ischemia_lcx'], axis=1)
    else:
        raise NotImplementedError("problem must be 'norm_abn' or 'localization'")

    if val_col:
        val_split = data[val_col]
        data = data.drop(val_col, axis=1)
        if val_col == 'cv_splits':
            data = data.drop('nn_val_split', axis=1)
        elif val_col == 'nn_val_split':
            data = data.drop('cv_splits', axis=1)
    else:
        val_split = None
    
    pt_id = data['pt_id']
    data = data.drop('pt_id', axis=1)
    study_no = data['study_no']
    data = data.drop('study_no', axis=1)
    
    if dataset == 'polar_plot':
#         polar_data_path = os.path.join(data_dir, 'polar_plot')
        polar_data_path = os.path.join(data_dir, 'raw', 'polar_plot')
        # Use the study_no column to get the appropriate images
        scans = polar_data(polar_data_path, study_no) 
        X = scans
    else:
        X = data

    return {'X': X, 'y': y, 'val_split': val_split,
            'study_no': study_no, 'pt_id': pt_id}


def plot_scan(scan):
    """
    Plots Stress, Rest, Reserve, and Difference
    
    Parameters:
    -----------
    scan : np.array
        a 4-d np.array where all dimensions are the same size and
        scan[0] is the map for stress values, scan[1] is the map for
        rest values, scan[2] is the map for reserve, and scan[3] is
        the map for difference
    """
    stress, rest, reserve, difference = scan[0], scan[1], scan[2], scan[3]
    max_val = scan.max()
    fig, axs = plt.subplots(2, 2, figsize=(12,10))
    labels = ["Stress", "Rest", "Reserve", "Difference"]
    images = [stress, rest, reserve, difference]
    
    for image, label, ax in zip(images, labels, axs.flatten()):
        ax.set_title(label)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        sns.heatmap(image, cmap="YlGnBu", vmin=0, vmax=max_val, ax=ax)


def get_features_and_labels(data, feature_cols, label_col, label_map=None):
    """
    Split a dataframe into one pd.DataFrame containing specified feature columns,
    and one pd.Series containing corresponding lables with an option
    to rename labels according to label_map.

    Parameters
    ----------
    data : pd.DataFrame
        contains a dataset, potentially with more features available
        than those specified in feature_cols

    feature_cols : list
        list of column names

    label_col : string
        column name

    label_map : dictionary
        key value pairs to rename the existing labels in label_col

    Returns
    -------
        (feature_df, labels) where feature_df is a dataframe with all
        features as columns, and labels is a pd.series containing
        corresponding labels.

    """
    feature_df = data[feature_cols].copy()
    labels = data[label_col].copy()

    if label_map:
        label_type = type(list(label_map.values())[0])

        # rename target variables
        for key, val in label_map.items():
            labels[labels == key] = val
        labels = labels.astype(label_type)

    return (feature_df, labels)


def format_col_names(pet):
    """

    Parameters
    ----------
    pet : pd.DataFrame
        multiindex DataFrame containing small dataset as read by pd.readexcel

    Returns
    -------
    pet : pd.DataFrame
        multiindex DataFrame with reformatted col names
    """
    pet.columns = pd.MultiIndex.from_tuples(map(lambda x: (str.lower(x[0]), str.lower(x[1])), pet.columns))
    pet.columns = pd.MultiIndex.from_tuples(map(lambda x: (str.strip(x[0]), str.strip(x[1])), pet.columns))
    pet.columns = pd.MultiIndex.from_tuples(map(lambda x: (re.sub(r'unnamed.*','', x[0]),
							   re.sub(r'unnamed.*','', x[1])), pet.columns))
    pet.columns = pd.MultiIndex.from_tuples(map(lambda x: (re.sub(r' - 17 segment','', x[0]),
							   re.sub(r' - 17 segment','', x[1])), pet.columns))
    pet.columns = pd.MultiIndex.from_tuples(map(lambda x: (x[0].replace(' ', '_'), x[1].replace(' ', '_')), pet.columns))
    return pet

def exclude(pet):
    """
    Excludes all scans that do not meet inclusion criteria below:

        Include only scans which:
        - have no calcium or calcium not commented
        - is not a transplant patients
        - does not presence of artifact or inconclusive perfusion
        - has interpretation == DefNorm, DefAbn, or DefAbnHR
        - does not have normal interpretation with indication of scar or
          ischemia
        - does not have an abnormal interpretation without indication of either
          scar or ischemia



    Parameters
    ----------
    pet : pd.DataFrame
       DataFrame formatted by format_col_names() and flatten_multiindex()

    Returns
    -------
    pet : pd.DataFrame
        The updated DataFrame
    """
    pet = pet[(pet['interp'] == 'DefAbn') | (pet['interp'] == 'DefAbnHR') | (pet['interp'] == 'DefNorm')]
    pet = pet[pet['calcium_present'] != 'Present']
    pet = pet[pet['transpt'] == 0]
    try:
        pet = pet[pet['inconclusive_perfusion'] == 0]
    except(KeyError):
        pet = pet[pet['perfusion_token_inconclusive_perfusion'] == 0]
    try:
        pet = pet[pet['artifact'] == 0]
    except(KeyError):
        pet = pet[pet['perfusion_token_artifact'] == 0]
    pet = pet[(pet['rest_pm_mean'] <= 4) &
              (pet['stress_pm_mean'] <= 8) &
              (pet['reserve_pm_mean'] <= 8)]
    pet = pet[~((pet['interp'] != 'DefNorm') &
                (pet['scar_present'] == 0) &
                (pet['ischemia_present'] == 0))]
    pet = pet[~((pet['interp'] == 'DefNorm') &
                    ((pet['scar_present'] == 1) |
                     (pet['ischemia_present'] == 1))
                )]
    return pet


def flatten_multiindex(pet):
    """
    Flattens a multiindex dataframe into single index and renames all columns
    accordingly

    Parameters
    ----------
    pet : pd.DataFrame
        Multiindex DataFrame to be flattened

   Returns
   ------
   pet : pd.DataFrame
        Flattened DataFrame
    """

    cols = []

    for outer, inner in pet.columns:
        if inner:
            if inner in ['calcium', 'ischemia', 'scar']:
                cols.append(outer + '_present')
            else:
                if outer == '':
                    cols.append(inner)
                else:
                    cols.append(outer + '_' + inner)
        else:
            cols.append(outer)
    pet.columns = cols
    return pet


def format_dataset(data, dataset, problem):
    """
    Formats data from raw data to inclue only relevant feature and label columns
    according to dataset and problem.
    
    data : pd.dataframe
        Unformatted data read in from original data file:
        '/q/PET-MBF/data/17segment_2008to2019 + PETReport (2020-10-26) + Pt-ID-Deidentified.xlsx'
    
    dataset : string
        Valid options are '17_segment' or '3_vessel', 'Polar Plot', None
        If dataset is None or 'Polar Plot', no features will be returned --
        only label columns, study_no, and patient_id
    
    problem : string
        Valid options are 'norm_abn', 'localization', and 'labels_only'.
        If norm_abn, label column: 'abnormal'
        If localization, label columns for all combinations of scar/ischemia
        and vessel territory.
        If labels_only, label columns include all from localization as well
        as abnormal, and abnormal for each vessel territory.
    
    """
    if dataset == '17_segment':
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

    elif dataset == '3_vessel':
        vessel_territories = ['lad', 'rca', 'lcx']
        rest_cols = ['rest_' + ves for ves in vessel_territories]                                 
        stress_cols = ['stress_' + ves for ves in vessel_territories]                      
        reserve_cols = ['reserve_' + ves for ves in vessel_territories]                  
        difference_cols = ['difference_' + ves for ves in vessel_territories]
    elif (dataset == None) or (dataset =='polar_plot'):
        rest_cols, stress_cols, reserve_cols, difference_cols = [], [], [], []
    else:
        raise ValueError()
    
    
    feature_cols = rest_cols + stress_cols + reserve_cols + difference_cols
    target_cols = ['interp', 'scar_lad', 'scar_rca', 'scar_lcx',
                   'ischemia_lad', 'ischemia_rca', 'ischemia_lcx']
    id_cols = ['study_#', 'pt-id']
    included_cols = feature_cols + target_cols + id_cols
    
    data = data[included_cols].copy()
    data = data.rename({'study_#': 'study_no',
                        'pt-id': 'pt_id'}, axis=1)


    
    data.loc[data['interp'] == 'DefNorm', 'abnormal'] = 0
    data.loc[data['interp'] == 'DefAbn', 'abnormal'] = 1
    data.loc[data['interp'] == 'DefAbnHR', 'abnormal'] = 1
    data['abnormal'] = data['abnormal'].astype(int)
    data = data.drop('interp', axis=1)
    
    data['abnormal_lad'] = (data['scar_lad'].astype(bool) | data['ischemia_lad'].astype(bool)).astype(int)
    data['abnormal_rca'] = (data['scar_rca'].astype(bool) | data['ischemia_rca'].astype(bool)).astype(int)
    data['abnormal_lcx'] = (data['scar_lcx'].astype(bool) | data['ischemia_lcx'].astype(bool)).astype(int)
    
    data['scar_lad'] = (data['scar_lad'] > 0).astype(int)
    data['scar_rca'] = (data['scar_rca'] > 0).astype(int)
    data['scar_lcx'] = (data['scar_lcx'] > 0).astype(int)
    data['ischemia_lad'] = data['ischemia_lad'].astype(int)
    data['ischemia_rca'] = data['ischemia_rca'].astype(int)
    data['ischemia_lcx'] = data['ischemia_lcx'].astype(int)
 
    if problem == 'norm_abn':
        target_cols = ['abnormal']
    elif problem == 'localization':
        target_cols = ['scar_lad', 'scar_rca', 'scar_lcx',
                       'ischemia_lad', 'ischemia_rca', 'ischemia_lcx']
    elif problem == 'labels_only':
        target_cols = ['abnormal', 'abnormal_lad', 'abnormal_rca', 'abnormal_lcx',
                       'scar_lad', 'scar_rca', 'scar_lcx', 'ischemia_lad',
                       'ischemia_rca', 'ischemia_lcx']
    else:
        assert(False)
    
    included_cols = rest_cols + stress_cols + reserve_cols + difference_cols\
                    + ['study_no', 'pt_id'] + target_cols
    
    
    return data[included_cols].copy()
