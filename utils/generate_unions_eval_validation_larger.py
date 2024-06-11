####   FOR USE IN SIMILARITY SEACH   ###
import numpy as np
import h5py
import pandas as pd

# TEMP
import sys
src = '/home/a4ferrei/projects/def-sfabbro/a4ferrei/' 
cc_dataloader_path = '/github/TileSlicer/'
sys.path.insert(0, src+cc_dataloader_path)
from dataloader import dataset_wrapper

eval_dataset_path = '/home/a4ferrei/projects/def-sfabbro/a4ferrei/data/dr5_eval_set_validation_10kx5tiles.h5' # just a 10k sample
num_tiles_to_save = 5  # specify the number of tiles to save

# Initialize dataset wrapper
dataset = dataset_wrapper()

# Open HDF5 file for writing
with h5py.File(eval_dataset_path, 'w') as f: 
    cutouts_combined = []
    columns = []
    for _ in range(num_tiles_to_save):
        # Load cutouts, catalog, and tile
        cutouts, catalog, _ = dataset.__next__()
        
        # Collect cutouts from all tiles
        cutouts_combined.extend(cutouts)
        
        # Store column names for the first iteration
        if not columns:
            columns = catalog.columns.tolist()
        
    # Create dataset for cutouts
    dset_cutouts = f.create_dataset("cutouts", data=np.array(cutouts_combined).astype(np.float64))
    
    # Store catalog data
    for column in columns:
        dset_column = f.create_dataset(column, data=np.array(catalog[column]).astype(catalog[column].dtype))

# Print the number of evaluated tiles
print("Number of Evaluated Tiles:", num_tiles_to_save)
