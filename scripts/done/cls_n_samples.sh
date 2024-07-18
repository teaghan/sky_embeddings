#!/bin/bash

# Module loads
module load python/3.11.5
source /home/obriaint/project/obriaint/torchnet/bin/activate
module load hdf5/1.10.6

# Copy files to slurm directory
cp /home/obriaint/scratch/sky_embeddings/cc/../data/HSC_dud_classes_calexp_GIRYZ7610_64_train.h5 $SLURM_TMPDIR
cp /home/obriaint/scratch/sky_embeddings/cc/../data/HSC_dud_classes_calexp_GIRYZ7610_64_val.h5 $SLURM_TMPDIR
cp /home/obriaint/scratch/sky_embeddings/cc/../data/HSC_dud_classes_GIRYZ7610_64_train.h5 $SLURM_TMPDIR
cp /home/obriaint/scratch/sky_embeddings/cc/../data/HSC_dud_classes_GIRYZ7610_64_val.h5 $SLURM_TMPDIR

# Log file to track completed training runs
LOG_FILE="/home/obriaint/scratch/sky_embeddings/cc/training_log.txt"

# Function to check if a training run has already been completed
function is_completed {
    grep -q "$1" "$LOG_FILE"
}

# Function to log a completed training run
function log_completed {
    echo "$1" >> "$LOG_FILE"
}

# List of training commands
declare -a TRAIN_CMDS=(
    "cls_ft_012k -v 1"
    "cls_ft_025k -v 1"
    "cls_ft_05k -v 2"
    "cls_ft_1k -v 4"
    "cls_ft_2k -v 8"
    "cls_ft_4k -v 16"
    "cls_ft_8k -v 32"
    "cls_ft_16k -v 64"
    "cls_ap_012k -v 1"
    "cls_ap_025k -v 1"
    "cls_ap_05k -v 2"
    "cls_ap_1k -v 4"
    "cls_ap_2k -v 8"
    "cls_ap_4k -v 16"
    "cls_ap_8k -v 32"
    "cls_ap_16k -v 64"
    "cls_fs_012k -v 1"
    "cls_fs_025k -v 1"
    "cls_fs_05k -v 2"
    "cls_fs_1k -v 4"
    "cls_fs_2k -v 8"
    "cls_fs_4k -v 16"
    "cls_fs_8k -v 32"
    "cls_fs_16k -v 64"
    "cls_ft_012k_wide -v 1"
    "cls_ft_025k_wide -v 1"
    "cls_ft_05k_wide -v 2"
    "cls_ft_1k_wide -v 4"
    "cls_ft_2k_wide -v 8"
    "cls_ft_4k_wide -v 16"
    "cls_ft_8k_wide -v 32"
    "cls_ft_16k_wide -v 64"
    "cls_ft_012k_large -v 1"
    "cls_ft_025k_large -v 1"
    "cls_ft_05k_large -v 2"
    "cls_ft_1k_large -v 4"
    "cls_ft_2k_large -v 8"
    "cls_ft_4k_large -v 16"
    "cls_ft_8k_large -v 32"
    "cls_ft_16k_large -v 64"
)

# Run each training command if not already completed
for CMD in "${TRAIN_CMDS[@]}"; do
    if ! is_completed "$CMD"; then
        python /home/obriaint/scratch/sky_embeddings/cc/../train_predictor.py $CMD -ct 10.00 -dd $SLURM_TMPDIR/
        log_completed "$CMD"
    fi
done

# Compare all
python /home/obriaint/scratch/sky_embeddings/cc/../compare_predictors.py _ -dd $SLURM_TMPDIR/
