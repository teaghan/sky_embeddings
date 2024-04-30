#!/bin/bash

# Module loads
module load StdEnv/2020
module load gcc/9.3.0 arrow/13.0.0 python/3.10
source /home/a4ferrei/mae_env/bin/activate
module load hdf5/1.10.6

# Run MAE training
python /home/a4ferrei/scratch/github/sky_embeddings/similarity_search_alt.py -model_name mim_88_unions -tgt_i [1]
python /home/a4ferrei/scratch/github/sky_embeddings/similarity_search_alt.py -model_name mim_88_unions -tgt_i [2]
python /home/a4ferrei/scratch/github/sky_embeddings/similarity_search_alt.py -model_name mim_88_unions -tgt_i [3]
python /home/a4ferrei/scratch/github/sky_embeddings/similarity_search_alt.py -model_name mim_88_unions -tgt_i [4]
python /home/a4ferrei/scratch/github/sky_embeddings/similarity_search_alt.py -model_name mim_88_unions -tgt_i [5]
python /home/a4ferrei/scratch/github/sky_embeddings/similarity_search_alt.py -model_name mim_88_unions -tgt_i [6]
python /home/a4ferrei/scratch/github/sky_embeddings/similarity_search_alt.py -model_name mim_88_unions -tgt_i [7]
python /home/a4ferrei/scratch/github/sky_embeddings/similarity_search_alt.py -model_name mim_88_unions -tgt_i [8]
python /home/a4ferrei/scratch/github/sky_embeddings/similarity_search_alt.py -model_name mim_88_unions -tgt_i [9]
python /home/a4ferrei/scratch/github/sky_embeddings/similarity_search_alt.py -model_name mim_88_unions -tgt_i [10]
python /home/a4ferrei/scratch/github/sky_embeddings/similarity_search_alt.py -model_name mim_88_unions -tgt_i [11]
python /home/a4ferrei/scratch/github/sky_embeddings/similarity_search_alt.py -model_name mim_88_unions -tgt_i [12]
python /home/a4ferrei/scratch/github/sky_embeddings/similarity_search_alt.py -model_name mim_88_unions -tgt_i [13]