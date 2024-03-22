#!/bin/bash

# Module loads
module load python/3.11.5
source /home/a4ferrei/mae_env/bin/activate
module load hdf5/1.10.6

# Run MAE training
python /home/a4ferrei/scratch/github/sky_embeddings/pretrain_mim.py mim_21_unions -v 1000 -ct 10.00 -dd $SLURM_TMPDIR/
