#!/bin/bash

# Module loads
module load python/3.11.5
source /home/obriaint/project/obriaint/torchnet/bin/activate
module load hdf5/1.10.6

python /home/obriaint/scratch/sky_embeddings/data_processing/2_create_h5_files.py dwarf

