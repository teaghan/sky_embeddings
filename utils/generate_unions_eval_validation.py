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

# Initialize dataset wrapper
dataset = dataset_wrapper()

# Open HDF5 file for writing
with h5py.File(eval_dataset_path, 'w') as f: 
    # Create datasets
    dset_cutouts = f.create_dataset("cutouts", (len(eval_tiles),), dtype=h5py.special_dtype(vlen=np.float64))
    dset_ra = f.create_dataset("ra", (len(eval_tiles),), dtype='f')
    dset_dec = f.create_dataset("dec", (len(eval_tiles),), dtype='f')

    # Iterate over tiles
    for i, tile in enumerate(eval_tiles):
        cutouts, catalog, _ = dataset.__next__()  # Load next tile

        # Store cutouts
        dset_cutouts[i] = cutouts.astype(np.float64)

        # Store RA and Dec
        dset_ra[i] = catalog['ra']
        dset_dec[i] = catalog['dec']

# Print tiles
print("Evaluated Tiles:", eval_tiles)