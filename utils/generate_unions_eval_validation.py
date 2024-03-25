import numpy as np
import h5py

# TEMP
import sys
src = '/home/a4ferrei/scratch/' 
cc_dataloader_path = '/github/extra/TileSlicer/'
sys.path.insert(0, src+cc_dataloader_path)
from dataloader import dataset_wrapper

eval_dataset_path = '/home/a4ferrei/scratch/data/dr5_eval_set_validation.h5' 
eval_tile = (285, 281)  # specify the tile to evaluate

# Initialize dataset wrapper
dataset = dataset_wrapper()

# Open HDF5 file for writing
with h5py.File(eval_dataset_path, 'w') as f: 
    # Load cutouts, catalog, and tile
    cutouts, catalog, _ = dataset.__next__()
    
    # Create dataset for cutouts
    dset_cutouts = f.create_dataset("cutouts", data=cutouts.astype(np.float64))
    
    # Store RA and Dec values
    dset_ra = f.create_dataset("ra", data=catalog['ra'].values.astype('f'))
    dset_dec = f.create_dataset("dec", data=catalog['dec'].values.astype('f'))

# Print the evaluated tile
print("Evaluated Tile:", eval_tile)