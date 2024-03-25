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

eval_dataset_path = '/home/a4ferrei/scratch/data/dr5_eval_set_redshift.h5'

found = 0
eval_tiles = range(20)
dataset = dataset_wrapper()

tiles = []
ra_lst, dec_lst, zspec_lst, cutout_lst = [], [], [], []
while found < len(eval_tiles):
    cutouts, catalog, tile = dataset.__next__() 
    if tile in eval_tiles: 
        tiles.append(tile)
        found +=1
        print('#######', found)

        ra = np.array(catalog['ra'])
        dec = np.array(catalog['dec'])
        zspec = np.array(catalog['zspec'])

        for i in range(len(zspec)):
            if np.isfinite(zspec[i]) and zspec[i]>0.02:
                ra_lst.append(ra[i])
                dec_lst.append(dec[i])
                zspec_lst.append(zspec[i])
                cutout_lst.append(cutouts[i])

        print(len(zspec_lst))

with h5py.File(eval_dataset_path, 'w') as f: 
    dset1 = f.create_dataset("cutouts", data = np.array(cutout_lst))
    dset2 = f.create_dataset("ra", data = np.array(ra_lst))
    dset3 = f.create_dataset("dec", data = np.array(dec_lst))
    dset4 = f.create_dataset("zspec", data = np.array(zspec_lst))

print(tiles) # these are then the off limit tiles