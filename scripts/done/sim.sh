#!/bin/bash

# Module loads
module load python/3.11.5
source /home/obriaint/project/obriaint/torchnet/bin/activate
module load hdf5/1.10.6

# Copy files to slurm directory
cp /home/obriaint/scratch/sky_embeddings/data/HSC_dud_dwarf_galaxy_calexp_GIRYZ7610_64_new.h5 $SLURM_TMPDIR
cp /home/obriaint/scratch/sky_embeddings/data/HSC_dud_dwarf_galaxy_GIRYZ7610_64_new.h5 $SLURM_TMPDIR
cp /home/obriaint/scratch/sky_embeddings/cc/../data/HSC_dud_unknown_calexp_GIRYZ7610_64_new.h5 $SLURM_TMPDIR
cp /home/obriaint/scratch/sky_embeddings/cc/../data/HSC_dud_unknown_GIRYZ7610_64_new.h5 $SLURM_TMPDIR
#cp /home/obriaint/scratch/sky_embeddings/cc/../data/HSC_dud_strong_lens_calexp_GIRYZ7610_64_new.h5 $SLURM_TMPDIR

#python /home/obriaint/scratch/sky_embeddings/cc/../similarity_search.py mim_25 -dd $SLURM_TMPDIR/ -bs 256 -mp False -tgt_i [1,2,3,4,7] -snr [0,10] -tgt_fn HSC_dud_dwarf_galaxy_calexp_GIRYZ7610_64_new.h5 -tst_fn HSC_dud_unknown_calexp_GIRYZ7610_64_new.h5
#python /home/obriaint/scratch/sky_embeddings/cc/../similarity_search.py mim_30 -dd $SLURM_TMPDIR/ -bs 256 -mp False -tgt_i [1,2,3,4,7] -snr [0,10] -tgt_fn HSC_dud_dwarf_galaxy_GIRYZ7610_64_new.h5 -tst_fn HSC_dud_unknown_GIRYZ7610_64_new.h5
#python /home/obriaint/scratch/sky_embeddings/cc/../similarity_search.py mim_32 -dd $SLURM_TMPDIR/ -bs 256 -mp False -tgt_i [1,2,3,4,7] -snr [0,10] -tgt_fn HSC_dud_dwarf_galaxy_GIRYZ7610_64_new.h5 -tst_fn HSC_dud_unknown_GIRYZ7610_64_new.h5
python /home/obriaint/scratch/sky_embeddings/cc/../sky_sim_search.py mim_32 -dd $SLURM_TMPDIR/ -bs 512 -mp False -tgt_i [1,2,3,4,7] -snr [0,10] -tgt_fn HSC_dud_dwarf_galaxy_GIRYZ7610_64_new.h5 -tst_dir /project/rrg-kyi/astro/hsc/pdr3_dud/
