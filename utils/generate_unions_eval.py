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
src = '/home/a4ferrei/projects/def-sfabbro/a4ferrei/' 
cc_dataloader_path = '/github/TileSlicer/'
sys.path.insert(0, src+cc_dataloader_path)
from dataloader import dataset_wrapper

eval_dataset_path = '/home/a4ferrei/projects/def-sfabbro/a4ferrei/data/dr5_eval_set_2.h5'
# dr5_eval_set.h5 just has (285, 281)

found = 0
#eval_tiles = [(285, 281), (150, 322), (183, 270), (144, 278)] 
eval_tiles = [1,2,3,4,5,6,7,8,9]# gets killed beyond even with high ram - may want to write in batches,10,11,12,13,14,15]
dataset = dataset_wrapper()

tiles = []
ra_lst, dec_lst, zspec_lst, cutout_lst = [], [], [], []
while found < len(eval_tiles):
    cutouts, catalog, tile = dataset.__next__() 
    #if tile in eval_tiles: 
    if True:
        tiles.append(tile)
        found +=1
        print('#######', found)

        ra = np.array(catalog['ra'])
        dec = np.array(catalog['dec'])
        zspec = np.array(catalog['zspec'])

        for i in range(len(zspec)):
            #if np.isfinite(zspec[i]) and zspec[i]>0.1:
            if not np.isnan(zspec[i]) and zspec[i]>0.01:
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