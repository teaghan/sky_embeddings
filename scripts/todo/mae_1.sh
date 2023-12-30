#!/bin/bash

# Module loads
module load python/3.11.5
source /home/obriaint/project/obriaint/torchnet/bin/activate
module load hdf5/1.10.6

# Copy files to slurm directory
cp /home/obriaint/scratch/sky_embeddings/data/HSC_grid_GRIZY_64.h5 $SLURM_TMPDIR
cp /home/obriaint/scratch/sky_embeddings/data/HSC_galaxies_GRIZY_64.h5 $SLURM_TMPDIR

# Run MAE training
python /home/obriaint/scratch/sky_embeddings/train_mae.py mae_1 -v 5000 -ct 10.00 -dd $SLURM_TMPDIR/
