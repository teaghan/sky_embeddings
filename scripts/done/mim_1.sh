#!/bin/bash

# Module loads
module load python/3.11.5
source /projects/def-sfabbro/a4ferrei/envs
/mae_env/bin/activate
module load hdf5/1.10.6

# Copy files to slurm directory
cp /project/rrg-kyi/astro/hsc/HSC_galaxies_GRIZY_64_val_new.h5 $SLURM_TMPDIR

# Run MAE training
python /home/a4ferrei/projects/rrg-kyi/a4ferrei/july_2024/sky_embeddingscc/../pretrain_mim.py mim_1 -v 5000 -ct 10.00 -dd $SLURM_TMPDIR/
