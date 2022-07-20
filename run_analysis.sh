#!/bin/bash
#SBATCH --job-name="test"
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=0
#SBATCH --output=./logs/%u-%x-job%j.out
#SBATCH --export=ALL
#SBATCH --mem=4GB


# Split data
singularity exec --bind /q:/q /q/PET-MBF/dberman/e-emagin-pet/Singularity/ray_nvidia.sif python ./scripts/make_stratified_datasets.py \
    --outdir ./data/splits \
    --in_datapath ./data/raw/data_random_values.csv \
    --max_class_dev 0.03


# BEFORE TRAINING FINAL MODELS WITH YOUR INSTITUTION'S DATA
# IDENTIFY OPTIMAL HYPERPARAMETERS THROUGH ITERATIVE PROCESS OUTLINED IN 
# EXAMPLE JUPYTER NOTEBOK IN ./Analysis_Notebooks


# Train final models for detection with optimal hyperparameters
singularity exec --bind /q:/q /q/PET-MBF/dberman/e-emagin-pet/Singularity/ray_nvidia.sif python3 ./scripts/train_models_17s_norm_abn.py \
    --datadir=./data \
    --outdir=./logs \
    --saved_models_path=./models/17_segment/norm_abn \
    --hyperparam_path=./config_files/17s_norm_abn_hyperparams.json


# Train final models for localization with optimal hyperparameters
singularity exec --bind /q:/q /q/PET-MBF/dberman/e-emagin-pet/Singularity/ray_nvidia.sif python3 ./scripts/train_models_17s_loc.py \
    --datadir=./data \
    --outdir=./logs \
    --saved_models_path=./models/17_segment/localization \
    --hyperparam_path=./config_files/17s_loc_hyperparams.json


# Run model comparison for detection models 
MODELS_PATH=./models/17_segment/norm_abn
OUTDIR=./results/norm_abn
singularity exec --bind /q:/q /q/PET-MBF/dberman/e-emagin-pet/Singularity/ray_nvidia.sif python3 ./scripts/evaluate.py \
    $MODELS_PATH/models0/svm.joblib \
    $MODELS_PATH/models0/rf.joblib \
    $MODELS_PATH/ensembles/unet_e.joblib \
    $MODELS_PATH/ensembles/mlp_e.joblib \
    --outdir $OUTDIR \
    --datadir ./data \
    --dataset 17_segment \
    --problem norm_abn \
    --all


# Run model comparison for localization models 
MODELS_PATH=./models/17_segment/localization
OUTDIR=./results/localization
singularity exec --bind /q:/q /q/PET-MBF/dberman/e-emagin-pet/Singularity/ray_nvidia.sif python3 ./scripts/evaluate.py \
    $MODELS_PATH/models0/rf.joblib \
    $MODELS_PATH/ensembles/unet_e.joblib \
    $MODELS_PATH/ensembles/mlp_e.joblib \
    --outdir $OUTDIR \
    --datadir ./data \
    --dataset 17_segment \
    --problem localization \
    --all


# Run evaluation of generalization performance for top performing detection model 
MODELS_PATH=./models/17_segment/norm_abn
OUTDIR=./results/norm_abn/generalization
singularity exec --bind /q:/q /q/PET-MBF/dberman/e-emagin-pet/Singularity/ray_nvidia.sif python3 ./scripts/evaluate.py \
    $MODELS_PATH/models0/rf.joblib \
    --outdir=$OUTDIR \
    --datadir=./data \
    --dataset=17_segment \
    --problem=norm_abn \
    --final


# Run evaluation of generalization performance for top performing localization model 
MODELS_PATH=./models/17_segment/localization
OUTDIR=./results/localization/generalization
singularity exec --bind /q:/q /q/PET-MBF/dberman/e-emagin-pet/Singularity/ray_nvidia.sif python3 ./scripts/evaluate.py \
    $MODELS_PATH/models0/rf.joblib \
    --outdir=$OUTDIR \
    --datadir=./data \
    --dataset=17_segment \
    --problem=localization \
    --final
