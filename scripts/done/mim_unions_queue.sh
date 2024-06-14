#!/bin/bash

module load StdEnv/2020
module load gcc/9.3.0 arrow/13.0.0 python/3.10
source /home/heesters/projects/def-sfabbro/a4ferrei/envs/mae_env/bin/activate
module load hdf5/1.10.6

# Run MAE training
python /home/heesters/projects/def-sfabbro/a4ferrei/github/sky_embeddings/pretrain_mim.py mim_218_unions -v 1 -ct 10.00
