#!/bin/bash

# Module loads
module load python/3.11.5
# source /project/def-sfabbro/a4ferrei/envs/mae_env/bin/activate
# module load StdEnv/2020
# module load intel/2021.2.0
module load hdf5/1.10.6

# Copy validation files to slurm directory
# cp /lustre07/scratch/obriaint/sky_embeddings/cc/../data/HSC_dud_galaxy_GIRYZ7610_64.h5 $SLURM_TMPDIR
# cp /lustre07/scratch/obriaint/sky_embeddings/cc/../data/HSC_dud_simple_classifier_data_GIRYZ7610_64.h5 $SLURM_TMPDIR
# cp /lustre07/scratch/obriaint/sky_embeddings/cc/../data/HSC_dud_simple_regressor_data_GIRYZ7610_64.h5 $SLURM_TMPDIR

# cp /project/rrg-kyi/astro/hsc/*.h5 /home/samjav/sky_embeddings/data/ $SLURM_TMPDIR

# # Run MIM training
python /home/samjav/sky_embeddings/pretrain_mim.py mim_MOCO_32.ini -v 5000 -ct 10.00 -dd $SLURM_TMPDIR/