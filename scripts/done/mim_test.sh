#!/bin/bash

# Module loads
module load python/3.11.5
source/project/ def-sfabbro/a4ferrei/envs/mae_env/bin/activate
module load hdf5/1.10.6

# Copy validation files to slurm directory
cp /project/rrg-kyi/astro/hsc/HSC_dud_galaxy_GIRYZ7610_64_new.h5 $SLURM_TMPDIR
cp /project/rrg-kyi/astro/hsc/HSC_dud_simple_classifier_data_GIRYZ7610_64.h5 $SLURM_TMPDIR
cp /project/rrg-kyi/astro/hsc/HSC_dud_simple_regressor_data_GIRYZ7610_64.h5 $SLURM_TMPDIR

# Run MIM training
python/project/ rrg-kyi/a4ferrei/july_2024/sky_embeddings/cc/../pretrain_mim.py mim_test -v 5 -ct 10.00 -dd $SLURM_TMPDIR/

