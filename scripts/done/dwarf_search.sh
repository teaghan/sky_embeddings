#!/bin/bash

# Module loads
module load StdEnv/2020
module load gcc/9.3.0 arrow/13.0.0 python/3.10
source /home/a4ferrei/mae_env/bin/activate
module load hdf5/1.10.6

# Run MAE training
python /home/a4ferrei/scratch/github/sky_embeddings/similarity_search.py -model_name mim_88_unions
