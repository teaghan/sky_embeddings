#!/bin/bash

module load StdEnv/2020
module load gcc/9.3.0 arrow/13.0.0 python/3.10
source /home/a4ferrei/mae_env/bin/activate
module load hdf5/1.10.6

# Run MAE training
python /home/a4ferrei/scratch/github/sky_embeddings/pretrain_mim.py mim_77_unions -v 1000 -ct 10.00
