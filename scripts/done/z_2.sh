#!/bin/bash

# Module loads
module load python/3.11.5
source /home/obriaint/project/obriaint/torchnet/bin/activate
module load hdf5/1.10.6

# Copy files to slurm directory
cp /home/obriaint/scratch/sky_embeddings/cc/../data/HSC_dud_galaxy_zspec_GIRYZ7610_64_train.h5  $SLURM_TMPDIR
cp /home/obriaint/scratch/sky_embeddings/cc/../data/HSC_dud_galaxy_zspec_GIRYZ7610_64_val.h5  $SLURM_TMPDIR

# Run predictor training
python /home/obriaint/scratch/sky_embeddings/cc/../train_predictor.py z_ft_2 -v 1000 -ct 10.00 -dd $SLURM_TMPDIR/

python /home/obriaint/scratch/sky_embeddings/cc/../test_predictor.py z_ft_2 -dd $SLURM_TMPDIR/

