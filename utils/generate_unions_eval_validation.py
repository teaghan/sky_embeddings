import numpy as np
import h5py

# TEMP
import sys
src = '/home/a4ferrei/scratch/' 
cc_dataloader_path = '/github/extra/TileSlicer/'
sys.path.insert(0, src+cc_dataloader_path)
from dataloader import dataset_wrapper

eval_dataset_path = '/home/a4ferrei/scratch/data/dr5_eval_set_validation.h5' 

found = 0
eval_tiles = [(285, 281)]#, (150, 322), (183, 270), (144, 278)] # these become off limit tiles in main code
dataset = dataset_wrapper()

tiles = []
ra_lst, dec_lst, zspec_lst, cutout_lst = [], [], [], []
while found < len(eval_tiles):
    cutouts, catalog, tile = dataset.__next__() 

    tiles.append(tile)
    found +=1
    print('#######', found)

    ra = np.array(catalog['ra'])
    dec = np.array(catalog['dec'])

with h5py.File(eval_dataset_path, 'w') as f: 
    dset1 = f.create_dataset("cutouts", data = np.array(cutouts))
    dset2 = f.create_dataset("ra", data = np.array(ra))
    dset3 = f.create_dataset("dec", data = np.array(dec))

print(tiles) # these are then the off limit tiles