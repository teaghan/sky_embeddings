# pretty inefficient but only needs to run once
import numpy as np
import h5py

import sys
src = '/home/a4ferrei/scratch/' 
cc_dataloader_path = '/github/extra/TileSlicer/'
sys.path.insert(0, src+cc_dataloader_path)
from dataloader import dataset_wrapper

eval_dataset_path = '/home/a4ferrei/scratch/data/dr5_eval_set_dwarfs_1k.h5' # lets see if we can fill 1k
eval_tiles = set(range(50000))  

# Initialize dataset wrapper
dataset = dataset_wrapper()

# Open HDF5 file for writing
with h5py.File(eval_dataset_path, 'w') as f:
    # Create datasets
    max_length = 1000
    dset_cutouts = f.create_dataset("cutouts", (max_length, 5, 224, 224), dtype=np.float32)
    dset_ra = f.create_dataset("ra", (max_length,), dtype='f')
    dset_dec = f.create_dataset("dec", (max_length,), dtype='f')
    dset_dwarf = f.create_dataset("dwarf", (max_length,), dtype='f')

    # Initialize counters
    index = 1
    tiles_written = []

    # Iterate over tiles
    while len(tiles_written) < len(eval_tiles):
        cutouts, catalog, tile = dataset.__next__()

        # Check if tile is in eval_tiles
        if tile != (285, 281):
            print('######', tile, index)
            tiles_written.append(tile)

            # Process catalog data and store directly in datasets
            entries = len(catalog)
            for i in range(entries):
                lsb = catalog['lsb'].iloc[i]
                gal = catalog['class'].iloc[i]
                if lsb==1 and gal==2:
                    dset_cutouts[index] = cutouts[i] 
                    dset_ra[index] = catalog['ra'].iloc[i]
                    dset_dec[index] = catalog['dec'].iloc[i]
                    dset_dwarf[index] = 1
                    # Increment index
                    index += 1

                    lsb = catalog['lsb'].iloc[i+1]
                    gal = catalog['class'].iloc[i+1]
                    if i+1 < entries and (lsb!=1 or gal!=2):
                        dset_cutouts[index] = cutouts[i+1] 
                        dset_ra[index] = catalog['ra'].iloc[i+1]
                        dset_dec[index] = catalog['dec'].iloc[i+1]
                        dset_dwarf[index] = 0
                        # Increment index
                        index += 1
                    else:
                        print('dwarf without a partner')


                    # ADD NON-LSB after each one of these thinking they are similar...? 
                    # check next one is not one or it is not last one --> maybe set up 30ks
                    dset_cutouts[index+1] = cutouts[i+1] 
                    dset_ra[index+1] = catalog['ra'].iloc[i+1]
                    dset_dec[index+1] = catalog['dec'].iloc[i+1]

                    # Increment index
                    index += 1

                    # Check if index exceeds max_length
                    if index >= max_length:
                        print('HIT MAX LEN, ran successfully')
                        break

        print('Processed', len(tiles_written), 'out of', len(eval_tiles), 'tiles')

print("Tiles written:", tiles_written)