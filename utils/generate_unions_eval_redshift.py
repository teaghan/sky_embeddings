import numpy as np
import h5py

import sys
src = '/home/heesters/projects/def-sfabbro/a4ferrei/' 
cc_dataloader_path = '/github/TileSlicer/'
sys.path.insert(0, src+cc_dataloader_path)
from dataloader import dataset_wrapper

eval_dataset_path = '/home/heesters/projects/def-sfabbro/a4ferrei/data/dr5_eval_set_redshift_6k_may2024_under1.h5'
#eval_dataset_path = '~/projects/  / /data/dr5_eval_set_redshift_6k_may2024_under1.h5'
eval_tiles = set(range(1000))  # specify tiles to evaluate, only taking 6k of each

# Initialize dataset wrapper
dataset = dataset_wrapper()

# Open HDF5 file for writing
with h5py.File(eval_dataset_path, 'w') as f:
    # Create datasets
    max_length = 6000
    dset_cutouts = f.create_dataset("cutouts", (max_length, 5, 224, 224), dtype=np.float32)
    dset_ra = f.create_dataset("ra", (max_length,), dtype='f')
    dset_dec = f.create_dataset("dec", (max_length,), dtype='f')
    dset_zspec = f.create_dataset("zspec", (max_length,), dtype='f')

    # Initialize counters
    index = 1
    tiles_written = []

    # Iterate over tiles
    while len(tiles_written) < len(eval_tiles):
        cutouts, catalog, tile = dataset.__next__()

        # Check if tile is in eval_tiles
        #if tile in eval_tiles:
        # index 0 gave an error so skipping for now
        #if index > 0: # since not specifying specific tiles, just doing a range
        if tile != (285, 281):
            print('######', tile, index)
            tiles_written.append(tile)

            # Process catalog data and store directly in datasets
            for i in range(len(catalog)):
                zspec = catalog['zspec'].iloc[i]
                if np.isfinite(zspec) and zspec > 0.002 and zspec < 1:
                    dset_cutouts[index] = cutouts[i] 
                    dset_ra[index] = catalog['ra'].iloc[i]
                    dset_dec[index] = catalog['dec'].iloc[i]
                    dset_zspec[index] = zspec

                    # Increment index
                    index += 1

                    # Check if index exceeds max_length
                    if index >= max_length:
                        print('HIT MAX LEN, ran successfully')
                        break

        print('Processed', len(tiles_written), 'out of', len(eval_tiles), 'tiles')

print("Tiles written:", tiles_written)