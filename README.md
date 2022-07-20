# Repo Structure

Analysis_notebooks
- These notebooks contain analysis. Mainly model selection and hyperparameter
  optimization for the various models, datasets, and prediction problems. 

codebase
- This contains reusable code and scripts that are often leveraged accross
  notebooks and scripts and includes scripts for training final models.

config_files
- This contains json files describing optimal hyperparameters for the final
  models

data
- This contians the raw data and the data split into files based on predictive
  task (localization / detection) and train / validation / test splits

logs
- This contians log files associated with the trainings of final models (i.e.
  model checkpoints and tensorboard logs for keras models)

models
- Contains final trained models. For mlp and unet models that are ensembled, the 
  individual trainings are saved in models/model0 - models/model11. The final
  ensemble is stored in models/ensembles. All other models are stored in models/model0

results
- Contains the results from comparison of models and from evaluation of final
  models (one model for detection and one for localization) for generalization
  performance.

scripts
- Contains the scripts used to create patientwise splits stratified by outcome,
  to train final models, and to evaluate models

Singularity
- .def and .sif files for the Singularity environment

run_analysis.sh
- A file that shows how to run all stages of analysis (besides iterative 
  hyperparameter tuning) with an example dataset of random values.

# Model selection experiments
A tool was written to streamline the hyperparameter tuning process and help
keep hyperparameter tuning organized. This tool launches hyperparameter tuning
experiments to SLURM from a simple interface. This makes it easy to adjust
hyperparameter ranges searched and keep track of these changes from a notebook
or single script, rather than having to create many versions of similar scripts
and submitting each to SLURM. This tool also makes it easy to keep code and output
organized by enforcing naming conventions and file structure for the files
created during hyperparameter tuning.

There is an option to use this same interface to launch model selection jobs
locally if one is not running this code on a cluster managed by SLURM. The
structure of the code that creates this tool is held within
codebase/hpc_scripts. See Analysis_notebooks/17_segment_hpc/model_selection_17_segment_localization.ipynb
for an example using this tool through the interfaces in codebase/model_selection_utils.py.


### Model selection SLURM job launcher overview
The directory codebase/hpc_scripts contains reusable scripts to help launch model selection / hyperparameter
tuning jobs on the HPC cluster managed with SLURM for jobs related to the
MITRE UOHI collaboration PET imaging project. These reusable scripts allow for
the launching of hyperparameter tuning jobs for scikit-learn's svm and random forest
models using scikit-learn's [GridSearchCV] (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html).
The scripts also allow for the launching of 
hyperparameter tuning jobs for a keras fully connected neural network and a keras U-Net with 
[Bayesian optimization] (http://hyperopt.github.io/hyperopt/) and
[grid search] (http://hyperopt.github.io/hyperopt/). The script run's
 HyperOpt's implementation of Tree-structured Parzen Estimators for the Bayesian
optimization algorithm.  Both the Bayesian optimization and gridsearch for the 
fully connected neural network are run through [Ray Tune] (https://docs.ray.io/en/latest/tune/index.html).

Currently, these algorithms are implemented for use with the classification
of normal/abnormal scans with the 3 vessel, 17 segment, and polar plot datasets.
The scripts were designed to be easily expanded to handle the localization and
classification-scar/ischemia problems.

### Files

There are three files that are used together to launch these gridsearch jobs
to SLURM: run_ms_on_hpc.py, ms_on_hpc.py, model_selection_utils.py. run_ms_on_hpc.py
and ms_on_hpc.py are command line programs. ms_on_hpc.py contains the code
for hyperparameter tuning for both sklearn and keras jobs. run_ms_on_hp.py 
requests resources and submits a hyperparameter tuning job to SLURM by running
ms_on_hpc.py. model_selection_utils.py contains the functions 
launch_mlp_hyper_opt_hpc and ms_on_hpc. These functions provide a clean interface
to launch hyperparameter tuning jobs on slurm from within python scripts or 
jupyter notebooks. These functions also help organize the names and locations
of files created as a part of the hyperparamter tuning process.

In order for these scripts to be completely reusable--requiring no code changes
to launch a new hyperparameter tuning job--the hyperparameter search space
to be searched must be specified in either a .json file or a .joblib file 
(depending on if the model is sklearn or keras), and this file will be read in
by ms_on_hpc.py where the search is executed. The interfaces in 
model_selection_utils will generate, save, and name these hyperparameter
space specification files and will pass the appropriate file path to 
run_ms_on_hpc.py.

To launch a new hyperparameter tuning job, the only functions you will
need are launch_mlp_hyper_opt_hpc and ms_on_hpc from model_selection_utils.py.

### Hyperparameter space specification

The format of the hyperparameter space passed to them model selection SLURM job
launcher will depend on model type. The hyperparameter space for scikit-learn
models will be in the format of the param_grid argument of scikit learn's
model_selection.GridSearchCV. The hyperparameter space for keras models will
be in the format of hyperopt search space when the search algorithm is hyperopt
and it will be in the format of the Ray tune.gridsearch interface.

For scikit learn models, hyperparameter names can be found in scikit learn
documentation. 

Please see Analysis_Notebooks/17_segment/model_selection_17_segment_localiation.ipynb
for description of the included hyperparameters and hyperparameter names and
examples of correctly formatted search spaces. 

### Output

When run, various output will be saved depending on the model type.

sklearn models:
- results: a csv with various model
configurations and performance for each according to various metrics.

keras models:
- results: a csv with various model
configurations and performance for each according to various metrics.
- Ray Tune log files: files containing performance info about each model configuration.
Performance information is stored in a separate file for each model configuration
as this enables Ray to run hyperparameter tuning in a distributed manner accross nodes. 
- TensorBoard files: Files containing information about both training and performance
of each model configuration. 

A directory structure will be created as follows for the output files. A argument
out_dir for the interfaces provided in model_selection_utils will specify the
base directory of the output directory structure. This directory and all
subdirectories will be created if they do not exist. 

The output structure:

    e-emagin-pet/output
        3_vessel
            norm_abn
                mlp
                    logs
                    hp_specs
                    results
                svm
                    logs
                    hp_specs
                    results
                rf
                    logs
                    hp_specs
                    results
        17_segment
            norm_abn
                mlp
                    logs
                    hp_specs
                    results
                svm
                    logs
                    hp_specs
                    results
                rf
                    logs
                    hp_specs
                    results
            localization
                mlp
                    ...
                svm 
                    ...
                rf
                    ...
        polar_plot
            ...


# Data 

    e-emagin-pet/data/raw
        formatted_raw_17s_3v_data.csv
        polar_plot
            - contains csv files representing cartesian mappings of full polar plot
              image data for rest and stress scans for each study
            - naming convention STUDYNO_SCANTYPE.csv where SCANTYPE is either 
              'rest' or 'stress'
    e-emagin-pet/data/splits
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
        

Tha above directory structure and csv files contianing data splits can be
generated by following the instructions in scripts/make_stratified_datasets.py.
The csv files contain tabular data for the relevant flow
measurements, the appropriate outcome columns, and columns for patient_id and
study_no. The files contianing training sets will have two additional columns,
‘cv_splits’ and ‘nn_val_split’, which indicate different groups within the
training set to be used during hyperparameter tuning for the classical ML 
models and the neural networks respectively. The ‘cv_splits’ column indicates
membership to 1 of 10 different groups which are the different folds of the data
used in 10 fold cross validation for the SVM and random forest. ‘nn_val_split’ indicates
membership to one of two groups – a training set (0) and a validation set for
hyperparameter tuning (1) for the neural networks. Helpful functions for
loading the datasets can be found in codebase/data_utils.py.

# Environment

### Singularity

The Singularity directory contians both a Singularity .def and .sif file. While
all of the code in this repo will run in the provided singularity
environment, the model selection job launcher cannot communicate with SLURM if
it is run from a within a singularity envoronment, as it needs access to the 
local machine's installation of SLURM. This is not possible with Singularity,
as the system's installations are kept separate from any process operating in
the Singularity image. Model selection initiated by the launcer tool can be
run from within the provided Singularity environment only when local mode is
enabled. Below are instructions for setting up a conda environment from which
jobs can be launhed on SLURM using this launcher tool.

### Conda

In order for these scripts to run, create a virtual environment 'e-emagin-pet'.
To set up this environment, execute the following commands.

1. conda create -n e-emagin-pet tensorflow pip scikit-learn pandas seaborn jupyter xlrd openpyxl
2. conda activate e-emagin-pet
3. pip install -U hyperopt


# Evaluation

A command line script, codebase/evaluate.py, in codebase can be used to compare the
performance of the various models. This evaluation scripts takes objects of
class ModelEvalWrapper. The ModelEvalWrapper class gives the same interface
to scikitlearn and keras models for functions necessary for evaluation. They
also record the input features used by each model as the logistic regression
may use fewer features than the other models on the tabular dataset. 


# Example

Follow the steps below to leverage the code from this repository for your
own analysis. Run run_analysis.sh to run all steps below (besides the iterative
hyperparameter tuning step) using fake data, formatted as expected by the
scripts in this repository, but with randomly generated values. The purpose of
run_analysis.sh is to illustrate how to run the included scripts, not to train
meaningful models. To train with data from your own institution, format data
as described below and as shown in the example random data, and run the
hyperparameter tuning for each model type with the example jupyter notebook
provided in Analysis_Notebooks.

## Step 1: Format tabular data

Save a csv containing features columns for 17 segmant and 3 vessel data
and label columns for detection and localization/classification of
abnormality type as described below. The file should also contian a unique
identifier for each study, 'study_no', and a unique identifier for each
patient 'pt_id'. File should be saved in ./data/raw/
    
### Feature columns:

- 3 vessel data:
    - 12 feature columns with names ${MEASUREMENT}_${VESSEL_TERRITORY} for
      measurements: ['rest', 'stress', 'reserve', 'difference'] and vessel territories:
      ['lad', 'rca', 'lcx']


- 17 segment data:
    - 68 feature columns with names ${MEASUREMENT}_${REGION} for measurements:
      ['rest', 'stress', 'reserve', 'difference'] and regions ['basal_anterior',
      'basal_anteroseptal', 'basal_inferoseptal', 'basal_inferior',
      'basal_inferolateral', 'basal_anterolateral', 'mid_anterior',
      'mid_anteroseptal', 'mid_inferoseptal', 'mid_inferior',
      'mid_inferolateral', 'mid_anterolateral', 'apical_anterior',
      'apical_septal', 'apical_inferior', 'apical_lateral', 'apex']

- polar plot data:
    - A directory data/raw/polar_plot with csvs of each rest and stress study
      in the format of 48 x 48 feature matrix with cartesian representation
      of polar plot and the naming convention STUDYNO_SCANTYPE.csv where
      SCANTYPE is either 'rest' or 'stress'

### Label columns:

- Per-patient detection:
    - 1 binary label column 'abnormal'. 0 represents normal, 1 represents abnormal'.

- Per-vessel localization and classification of abnormality type:
    - 6 binary columns ['scar_lad', 'scar_rca', 'scar_lcx', 'ischemia_lad',
      'ischemia_rca', 'ischemia_lcx'], where 0 indicates that the abnormality
      of the given type is not present in the given vessel territory, and 1
      indicates that the abnormality of the given type is present in the given
      vessel territory.

## Step 2: Split the data

- Use the script /codebase/make_stratified_datasets.py to split and store
  the formatted data file.
- Once stratified datasets are saved by the above script and image data is
  saved in the specified format, data can be loaded with
  data_utils.load_dataset, which is used widely through out the code.  
- If script does not converge, try increasing max_class_dev. See argument
  description with --help. 

Example:

    singularity exec --bind /home:/home ./Singularity/ray_nvidia.sif ./scripts/make_stratified_datasets.py 
        --outdir ./data/splits
        --in_datapath ./data/raw/data_random_values.csv
        --max_class_dev 0.02

## Step 3: Hyperparameter tuning

- Identify optimal hyperparameters for each model, for each predictive
  problem (detection, localizaiton).
- See example of iterative hyperparameter tuning process and hyperparameter
  tuning job launcher in Analysis_Notebooks/17_segment/model_selection_17_segment_localiation.ipynb 
- If your institution does not run a cluster managed by SLURM, it is still
  possible to use the python interface to the hyperparameter tuning job
  launcher by passing the argument local=True. If you use the local option,
  launch the jupyter notebook within the singlularity environment to ensure
  access to all necessary packages. If you want to launch jobs on your
  institution's SLURM cluster, do not use the local argument, and launch
  the jupyter notebook within the e-emagin-pet conda environment described
  above in the environment section.


## Step 4: Train final models

- Once the optimal hyperparameters have been identified for each model,
  record optimal hyperparameters for each model in a json file in /config_files. 

Example:

    singularity exec --bind /home:/home e-emagin-pet/Singularity/ray_nvidia.sif python3 train_models_17s_loc.py
        --datadir=./data
        --outdir=./logs
        --saved_models_path=./models/17_segment/localization
        --hyperparam_path=./config_files/17s_loc_hyperparams.json
      
## Step 4: Model comparison

- Use the script /codebase/evaluate.py to compare final models and identify
  highest performing model for each predictive task and representation of
  perfusion data.
- Run evaluation on validation set for model comparison

Example:

    singularity exec --bind /home:/home e-emagin-pet/Singularity/ray_nvidia.sif python3 evaluate.py
        MODEL1.joblib
        MODEL2.joblib 
        MODEL3.joblib
        --outdir ./results/norm_abn
        --datadir ./data
        --dataset 17_segment
        --problem norm_abn
        --all
        

## Step 5: Generalization performance

- Use /codebase/evaluate.py to calculate generalization performance for the
  highest performing model from the comparison on the held out test set for
  generalizabilty. 

Example:

    singularity exec --bind /home:/home e-emagin-pet/Singularity/ray_nvidia.sif python3 evaluate.py
        FINAL_MODEL.joblib
        --outdir ./results/norm_abn/generalization
        --datadir ./data
        --dataset 17_segment
        --problem norm_abn
        --final

# Public Release
©2022 The MITRE Corporation 
Approved for Public Release; Distribution Unlimited. 
Public Release Case Number 22-1848

# License

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

`http://www.apache.org/licenses/LICENSE-2.0`

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

