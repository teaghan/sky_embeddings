#!/bin/bash

# Module loads
module load python/3.11.5
source /home/obriaint/project/obriaint/torchnet/bin/activate
module load hdf5/1.10.6

# Copy files to slurm directory
cp /home/obriaint/scratch/sky_embeddings/cc/../data/HSC_dud_classes_calexp_GIRYZ7610_64_train.h5 $SLURM_TMPDIR
cp /home/obriaint/scratch/sky_embeddings/cc/../data/HSC_dud_classes_calexp_GIRYZ7610_64_val.h5 $SLURM_TMPDIR

# Run predictor training
python /home/obriaint/scratch/sky_embeddings/cc/../train_predictor.py cls_ap_025k -v 1 -ct 10.00 -dd $SLURM_TMPDIR/
python /home/obriaint/scratch/sky_embeddings/cc/../train_predictor.py cls_ap_05k -v 2 -ct 10.00 -dd $SLURM_TMPDIR/
python /home/obriaint/scratch/sky_embeddings/cc/../train_predictor.py cls_ap_1k -v 4 -ct 10.00 -dd $SLURM_TMPDIR/
python /home/obriaint/scratch/sky_embeddings/cc/../train_predictor.py cls_ap_2k -v 8 -ct 10.00 -dd $SLURM_TMPDIR/
python /home/obriaint/scratch/sky_embeddings/cc/../train_predictor.py cls_ap_4k -v 16 -ct 10.00 -dd $SLURM_TMPDIR/
python /home/obriaint/scratch/sky_embeddings/cc/../train_predictor.py cls_ap_8k -v 32 -ct 10.00 -dd $SLURM_TMPDIR/
python /home/obriaint/scratch/sky_embeddings/cc/../train_predictor.py cls_ap_16k -v 64 -ct 10.00 -dd $SLURM_TMPDIR/
python /home/obriaint/scratch/sky_embeddings/cc/../train_predictor.py cls_ft_025k -v 1 -ct 10.00 -dd $SLURM_TMPDIR/
python /home/obriaint/scratch/sky_embeddings/cc/../train_predictor.py cls_ft_05k -v 2 -ct 10.00 -dd $SLURM_TMPDIR/
python /home/obriaint/scratch/sky_embeddings/cc/../train_predictor.py cls_ft_1k -v 4 -ct 10.00 -dd $SLURM_TMPDIR/
python /home/obriaint/scratch/sky_embeddings/cc/../train_predictor.py cls_ft_2k -v 8 -ct 10.00 -dd $SLURM_TMPDIR/
python /home/obriaint/scratch/sky_embeddings/cc/../train_predictor.py cls_ft_4k -v 16 -ct 10.00 -dd $SLURM_TMPDIR/
python /home/obriaint/scratch/sky_embeddings/cc/../train_predictor.py cls_ft_8k -v 32 -ct 10.00 -dd $SLURM_TMPDIR/
python /home/obriaint/scratch/sky_embeddings/cc/../train_predictor.py cls_ft_16k -v 64 -ct 10.00 -dd $SLURM_TMPDIR/
python /home/obriaint/scratch/sky_embeddings/cc/../train_predictor.py cls_fs_025k -v 1 -ct 10.00 -dd $SLURM_TMPDIR/
python /home/obriaint/scratch/sky_embeddings/cc/../train_predictor.py cls_fs_05k -v 2 -ct 10.00 -dd $SLURM_TMPDIR/
python /home/obriaint/scratch/sky_embeddings/cc/../train_predictor.py cls_fs_1k -v 4 -ct 10.00 -dd $SLURM_TMPDIR/
python /home/obriaint/scratch/sky_embeddings/cc/../train_predictor.py cls_fs_2k -v 8 -ct 10.00 -dd $SLURM_TMPDIR/
python /home/obriaint/scratch/sky_embeddings/cc/../train_predictor.py cls_fs_4k -v 16 -ct 10.00 -dd $SLURM_TMPDIR/
python /home/obriaint/scratch/sky_embeddings/cc/../train_predictor.py cls_fs_8k -v 32 -ct 10.00 -dd $SLURM_TMPDIR/
python /home/obriaint/scratch/sky_embeddings/cc/../train_predictor.py cls_fs_16k -v 64 -ct 10.00 -dd $SLURM_TMPDIR/

# Compare all
python /home/obriaint/scratch/sky_embeddings/cc/../compare_predictors.py _ -dd $SLURM_TMPDIR/

