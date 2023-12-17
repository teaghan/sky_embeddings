#!/bin/bash

# Module loads
module load python/3.9.6
source /home/obriaint/project/obriaint/torchnet/bin/activate
module load hdf5/1.10.6

# Copy files to slurm directory
cp /home/obriaint/scratch/sky_embeddings/data/objects_GIR_64.h5 $SLURM_TMPDIR

# Run MAE training
python /home/obriaint/scratch/sky_embeddings/train_mae.py mae_1 -v 1000 -ct 10.00 -dd $SLURM_TMPDIR/
