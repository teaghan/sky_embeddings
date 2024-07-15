#!/bin/bash

# Module loads
module load python/3.11.5
source /home/a4ferrei/projects/def-sfabbro/a4ferrei/envs
/mae_env/bin/activate
module load hdf5/1.10.6

# Copy files to slurm directory
cp /lustre07/scratch/obriaint/sky_embeddings/cc/../data/HSC_dwarf_galaxies_GRIZY_64_new.h5 $SLURM_TMPDIR
cp /lustre07/scratch/obriaint/sky_embeddings/cc/../data/HSC_unkown_GRIZY_64_new.h5 $SLURM_TMPDIR
cp /lustre07/scratch/obriaint/sky_embeddings/cc/../data/HSC_strong_lens_candidates_GRIZY_64.h5 $SLURM_TMPDIR

python /lustre07/scratch/obriaint/sky_embeddings/cc/../similarity_search.py sim_22 -dd $SLURM_TMPDIR/ -bs 256

python /lustre07/scratch/obriaint/sky_embeddings/cc/../similarity_search.py sim_22 -dd $SLURM_TMPDIR/ -tgt_fn HSC_strong_lens_candidates_GRIZY_64.h5 -tgt_i [1,5,13,15,16,33,49] -snr [5,1000] -bs 256
