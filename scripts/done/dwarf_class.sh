#!/bin/bash

# Module loads
module load gcc/9.3.0 arrow/13.0.0 python/3.10
module load python/3.11.5
source /home/a4ferrei/mae_env/bin/activate
module load hdf5/1.10.6

# Run MAE training
python /home/a4ferrei/scratch/github/sky_embeddings/utils/generate_unions_eval_dwarfs.py
