#!/bin/bash

# Module loads
module load python/3.11.5
source/project/ def-sfabbro/a4ferrei/envs/mae_env/bin/activate
module load hdf5/1.10.6

# Copy files to slurm directory
cp /project/rrg-kyi/astro/hsc/HSC_galaxies_GRIZY_64_val_new.h5 $SLURM_TMPDIR

# Run MAE training
python/project/ rrg-kyi/a4ferrei/july_2024/sky_embeddings/cc/../pretrain_mim.py mim_1 -v 5 -ct 1.00 -dd $SLURM_TMPDIR/

# test up to this point with: 
# sbatch --time=01:00:00 --cpus-per-task=6 --account=def-sfabbro --mem=100000M  --gres=gpu:1 ./scripts/done/mim_1_troubleshoot.sh