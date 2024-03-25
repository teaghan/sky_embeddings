import numpy as np
import h5py

import sys
src = '/home/a4ferrei/scratch/' 
cc_dataloader_path = '/github/extra/TileSlicer/'
sys.path.insert(0, src+cc_dataloader_path)
from dataloader import dataset_wrapper

eval_dataset_path = '/home/a4ferrei/scratch/data/dr5_eval_set_redshift.h5'
eval_tiles = set(range(30))  # specify tiles to evaluate, only taking 10k of each

# Initialize dataset wrapper
dataset = dataset_wrapper()

# Open HDF5 file for writing
with h5py.File(eval_dataset_path, 'w') as f:
    # Create datasets
    max_length = 6000
    dset_cutouts = f.create_dataset("cutouts", (max_length,), dtype=h5py.special_dtype(vlen=np.uint8))
    dset_ra = f.create_dataset("ra", (max_length,), dtype='f')
    dset_dec = f.create_dataset("dec", (max_length,), dtype='f')
    dset_zspec = f.create_dataset("zspec", (max_length,), dtype='f')

    # Initialize counters
    index = 0
    tiles_written = []

    # Iterate over tiles
    while len(tiles_written) < len(eval_tiles):
        cutouts, catalog, tile = dataset.__next__()

        # Check if tile is in eval_tiles
        if tile in eval_tiles:
            print('######', tile, index)
            tiles_written.append(tile)

            # Process catalog data and store directly in datasets
            for i in range(len(catalog)):
                zspec = catalog['zspec'].iloc[i]
                if np.isfinite(zspec) and zspec > 0.002:
                    dset_cutouts[index] = cutouts[i].astype(np.uint8)
                    dset_ra[index] = catalog['ra'].iloc[i]
                    dset_dec[index] = catalog['dec'].iloc[i]
                    dset_zspec[index] = zspec

                    # Increment index
                    index += 1

                    # Check if index exceeds max_length
                    if index >= max_length:
                        break

        print('Processed', len(tiles_written), 'out of', len(eval_tiles), 'tiles')

    # Resize datasets to actual length
    dset_cutouts.resize(index)
    dset_ra.resize(index)
    dset_dec.resize(index)
    dset_zspec.resize(index)

print("Tiles written:", tiles_written)