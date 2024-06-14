#!/bin/bash

module load StdEnv/2023
module load gcc/12.3 arrow/16.1.0 python/3.11.5
source /home/heesters/projects/def-sfabbro/heesters/envs/ssl_env/bin/activate
module load hdf5/1.14.2

# Run MAE training
python /home/heesters/projects/def-sfabbro/heesters/github/sky_embeddings/pretrain_mim.py mim_218_unions -v 1 -ct 1.00
