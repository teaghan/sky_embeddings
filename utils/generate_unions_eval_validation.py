import numpy as np
import h5py

# TEMP
import sys
src = '/home/a4ferrei/projects/def-sfabbro/a4ferrei/' 
cc_dataloader_path = '/github/TileSlicer/'
sys.path.insert(0, src+cc_dataloader_path)
from dataloader import dataset_wrapper

eval_dataset_path = '/home/a4ferrei/projects/def-sfabbro/a4ferrei/data/dr5_eval_set_validation.h5' # just a 10k sample
eval_tile = (285, 281)  # specify the tile to evaluate

# Initialize dataset wrapper
dataset = dataset_wrapper()

# Open HDF5 file for writing
with h5py.File(eval_dataset_path, 'w') as f: 
    # Load cutouts, catalog, and tile
    cutouts, catalog, tile = dataset.__next__()

    if tile == eval_tile:
        # Create dataset for cutouts
        dset_cutouts = f.create_dataset("cutouts", data=cutouts.astype(np.float64))
        
        # Store RA and Dec values
        dset_ra = f.create_dataset("ra", data=catalog['ra'].values.astype('f'))
        dset_dec = f.create_dataset("dec", data=catalog['dec'].values.astype('f'))

        # Print the evaluated tile
        print("Evaluated Tile:", eval_tile)

    else:
        print(f'for some reason {eval_tile} is not the first tile but {tile} is')