#!/bin/bash

# Module loads
module load python/3.11.5
source /home/obriaint/project/obriaint/torchnet/bin/activate
module load hdf5/1.10.6

# Copy files to slurm directory
cp /lustre04/scratch/obriaint/sky_embeddings/cc/../data/HSC_galaxies_GRIZY_64_val_new.h5 $SLURM_TMPDIR

# Run MAE training
python /lustre04/scratch/obriaint/sky_embeddings/cc/../pretrain_mim.py mim_1 -v 5000 -ct 10.00 -dd $SLURM_TMPDIR/
