import random
import numpy as np
import torch
import h5py
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2
import os
import glob
from astropy.io import fits
from astropy.wcs import WCS

# TEMP
import sys
src = '/home/a4ferrei/scratch/' 
cc_dataloader_path = '/github/extra/TileSlicer/'
sys.path.insert(0, src+cc_dataloader_path)
from dataloader import dataset_wrapper

eval_dataset_path = '/home/a4ferrei/scratch/data/dr5_eval_set.h5'

found = 0
eval_tiles = [(285, 281)] # only doing this with one tile in mind now
dataset = dataset_wrapper()

while found < len(eval_tiles):
    cutouts, catalog, tile = dataset.__next__() 
    if tile in eval_tiles: 
        found +=1 

        ra = np.array(catalog['ra'])
        dec = np.array(catalog['dec'])
        zspec = np.array(catalog['zspec'])

        with h5py.File(eval_dataset_path, 'w') as f: 
            dset1 = f.create_dataset("cutouts", data = cutouts)
            dset2 = f.create_dataset("ra", data = ra)
            dset3 = f.create_dataset("dec", data = dec)
            dset4 = f.create_dataset("zspec", data = zspec)
