import numpy as np
import os
import h5py
from utils import getPatches, cutouts_to_hdf5, get_cutouts_from_patch
import pandas as pd
import glob
import shutil
import gc
import sys

def create_h5_dataset(fits_paths, bands, use_calexp, img_size, out_dir, out_filename, scratch_dir='/scratch',
                      labels_path=None, create_grid=False, patch_strategy='all', 
                      n_patches=-1, patch_low_bound=0, patch_high_bound=-1):

    # Temporary storage location
    tmp_dir = os.path.join(out_dir,'tmp_'+''.join(np.random.randint(0,9,size=(6,)).astype(str)))
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.mkdir(tmp_dir)

    # Find fits files and get ready for loading
    kwargs = {'data_paths':fits_paths,
              'bands':bands,
              'use_calexp':use_calexp,
              'patch_strategy':patch_strategy,
              'n_patches':n_patches, 
              'patch_low_bound': patch_low_bound,
              'patch_high_bound': patch_high_bound}

    if (labels_path is not None) & (not create_grid):
        # Load csv file with ra, dec, and zspec info 
        labels = pd.read_csv(labels_path)
    # Loop through fits patches
    for i, p in enumerate(getPatches(**kwargs)):

        # Load patch
        p.load_header()
        # Find ra and decs that are within the path
        p.match_catalogue(labels)
        if len(p.matches)>0:
            p.load()
            # Grab cutouts
            cutouts = get_cutouts_from_patch(p, bands=bands)        
            # Save to h5 file
            cutouts_to_hdf5(tmp_dir, cutouts, bands=bands)
            del cutouts

        if (i+1)%10==0:
            print(f'{i+1} patches complete...', end='\r')
        # Close fits files
        p.__del__()
        del p
        
        gc.collect()

    print('\nFinished creating cutouts.')

    tmp_h5_paths = glob.glob(os.path.join(tmp_dir, '*.h5'))
    if len(tmp_h5_paths)>0:
        print(f'Creating {out_filename} data file from {len(tmp_h5_paths)} small files...')
        # Combine individual files into one large .h5 file
        with h5py.File(os.path.join(out_dir, out_filename), "w") as f_out:
            for i, fn in enumerate(tmp_h5_paths):
                try:
                    with h5py.File(fn, "r") as f_in: 
                        for k in f_in.keys():
    
                            if i==0:
                                # Create the same datasets
                                shape = list(f_in[k].shape)
                                shape[0] = 0
                                maxshape = shape.copy()
                                maxshape[0] = None
                                f_out.create_dataset(k, shape=tuple(shape), maxshape=maxshape)
    
                            ds_out = f_out[k]
                            num = len(f_in[k])
                            # Resize the dataset to accommodate the new data
                            ds_out.resize(ds_out.shape[0] + num, axis=0)
                            # Add the new data
                            ds_out[-num:] = f_in[k][:]
                except OSError:
                    continue
                if (i+1)%1000==0:
                    print(f'{i+1} files complete...')
            print(f'Finished creating {out_filename} with {len(ds_out)} samples.')
    else:
        print('No matches found.')
    
    # Remove temp files
    shutil.rmtree(tmp_dir)

def create_h5_subsets(fits_paths, out_name, labels_path, out_dir, patch_start, patch_end, batch_size,
                      bands, use_calexp, img_size):

    for i in range(patch_start, patch_end, batch_size):
        low = i
        high = i+batch_size
        if high>patch_end:
            high = patch_end-1
        out_filename = f'{out_name}_{low}_to_{high}.h5'
        if os.path.exists(os.path.join(out_dir, out_filename)):
            print(f'{out_filename} already exists. Moving on...')
            continue
        else:
            create_h5_dataset(fits_paths, bands, use_calexp, img_size, out_dir, out_filename, labels_path=labels_path,
                              patch_strategy='n_to_n', patch_low_bound=low, patch_high_bound=high)

img_size = 64

#fits_paths = ['/project/rrg-kyi/astro/hsc/pdr3_wide','/project/rrg-kyi/astro/hsc/pdr3_dud']
fits_paths = ['/home/obriaint/scratch/sky_embeddings/data/pdr3_dud']

out_dir = '/home/obriaint/scratch/sky_embeddings/data/'

bands = ['G', 'I', 'R', 'Y', 'Z', 'NB0387', 'NB0816', 'NB0921', 'NB1010']

batch_size = 100

# Command line argument
file_type = sys.argv[1]

if 'gal' in file_type:
    # Galaxies
    for out_name, use_calexp in zip(['HSC_dud_galaxy_GIRYZ7610_64'], #'HSC_dud_galaxy_calexp_GIRYZ7610_64',
                                    [False]): # True
        labels_path = os.path.join(out_dir,'HSC_galaxy_dud.csv')
        patch_start = 0
        patch_end = 1515
        create_h5_subsets(fits_paths, out_name, labels_path, out_dir, patch_start, patch_end, batch_size,
                              bands, use_calexp, img_size)

if 'qso' in file_type:
    # QSOs
    for out_name, use_calexp in zip(['HSC_dud_qso_calexp_GIRYZ7610_64', 'HSC_dud_qso_GIRYZ7610_64'],
                                    [True, False]):
        labels_path = os.path.join(out_dir,'HSC_qso_dud.csv')
        patch_start = 0
        patch_end = 1515
        create_h5_subsets(fits_paths, out_name, labels_path, out_dir, patch_start, patch_end, batch_size,
                              bands, use_calexp, img_size)

if 'star' in file_type:
    # Stars
    for out_name, use_calexp in zip(['HSC_dud_star_calexp_GIRYZ7610_64', 'HSC_dud_star_GIRYZ7610_64'],
                                    [True, False]):
        labels_path = os.path.join(out_dir,'HSC_star_dud.csv')
        patch_start = 0
        patch_end = 1515
        create_h5_subsets(fits_paths, out_name, labels_path, out_dir, patch_start, patch_end, batch_size,
                              bands, use_calexp, img_size)

if 'zspec' in file_type:
    # Redshift files
    for out_name, use_calexp in zip(['HSC_dud_galaxy_zspec_calexp_GIRYZ7610_64_train', 'HSC_dud_galaxy_zspec_GIRYZ7610_64_train'],
                                    [True, False]):
        labels_path = os.path.join(out_dir,'HSC_galaxy_dud_zspec_train.csv')
        patch_start = 0
        patch_end = 1515
        create_h5_subsets(fits_paths, out_name, labels_path, out_dir, patch_start, patch_end, batch_size,
                              bands, use_calexp, img_size)
    for out_name, use_calexp in zip(['HSC_dud_galaxy_zspec_calexp_GIRYZ7610_64_val', 'HSC_dud_galaxy_zspec_GIRYZ7610_64_val'],
                                    [True, False]):
        labels_path = os.path.join(out_dir,'HSC_galaxy_dud_zspec_val.csv')
        patch_start = 0
        patch_end = 1515
        create_h5_subsets(fits_paths, out_name, labels_path, out_dir, patch_start, patch_end, batch_size,
                              bands, use_calexp, img_size)
    for out_name, use_calexp in zip(['HSC_dud_galaxy_zspec_calexp_GIRYZ7610_64_test', 'HSC_dud_galaxy_zspec_GIRYZ7610_64_test'],
                                    [True, False]):
        labels_path = os.path.join(out_dir,'HSC_galaxy_dud_zspec_test.csv')
        patch_start = 0
        patch_end = 1515
        create_h5_subsets(fits_paths, out_name, labels_path, out_dir, patch_start, patch_end, batch_size,
                              bands, use_calexp, img_size)

if 'dwarf' in file_type:
    # Dwarf Galaxies
    for out_name, use_calexp in zip(['HSC_dud_dwarf_galaxy_calexp_GIRYZ7610_64', 'HSC_dud_dwarf_galaxy_GIRYZ7610_64'],
                                    [True, False]):
        labels_path = os.path.join(out_dir,'dwarf_galaxies.csv')

        out_filename = f'{out_name}.h5'
        if os.path.exists(os.path.join(out_dir, out_filename)):
            print(f'{out_filename} already exists. Moving on...')
            continue
        else:
            create_h5_dataset(fits_paths, bands, use_calexp, img_size, out_dir, out_filename, labels_path=labels_path,
                              patch_strategy='all', n_patches=-1)

if 'lense' in file_type:
    # Strong Lensing events
    for out_name, use_calexp in zip(['HSC_dud_strong_lens_calexp_GIRYZ7610_64', 'HSC_dud_strong_lens_GIRYZ7610_64'],
                                    [True, False]):
        labels_path = os.path.join(out_dir,'strong_lens_candidates.csv')

        out_filename = f'{out_name}.h5'
        if os.path.exists(os.path.join(out_dir, out_filename)):
            print(f'{out_filename} already exists. Moving on...')
            continue
        else:
            create_h5_dataset(fits_paths, bands, use_calexp, img_size, out_dir, out_filename, labels_path=labels_path,
                              patch_strategy='all', n_patches=-1)


if 'unknown' in file_type:
    # Stars
    for out_name, use_calexp in zip(['HSC_dud_unknown_calexp_GIRYZ7610_64', 'HSC_dud_unknown_GIRYZ7610_64'],
                                    [True, False]):
        labels_path = os.path.join(out_dir,'HSC_unknown_dud.csv')
        patch_start = 0
        patch_end = 1515
        create_h5_subsets(fits_paths, out_name, labels_path, out_dir, patch_start, patch_end, batch_size,
                              bands, use_calexp, img_size)

'''
if 'dwarf' in file_type:
    # Dwarf Galaxies
    fits_paths = ['/project/rrg-kyi/astro/hsc/pdr3_wide','/project/rrg-kyi/astro/hsc/pdr3_dud']
    batch_size = 5000
    for out_name, use_calexp in zip(['HSC_all_dwarf_galaxy_calexp_GIRYZ7610_64', 'HSC_dud_dwarf_galaxy_GIRYZ7610_64'],
                                    [True, False]):
        labels_path = os.path.join(out_dir,'dwarf_galaxies.csv')
        patch_start = 0
        patch_end = 52882
        create_h5_subsets(fits_paths, out_name, labels_path, out_dir, patch_start, patch_end, batch_size,
                              bands, use_calexp, img_size)
'''

'''

out_filename = 'HSC_strong_lens_candidates_GRIZY_64.h5'
labels_path = '../data/strong_lens_candidates.csv'
create_h5_dataset(fits_path, bands, img_size, out_dir, out_filename, labels_path=labels_path,
                  patch_strategy='all', n_patches=-1)

out_filename = 'HSC_unkown_GRIZY_64.h5'
labels_path = '../data/unkown.csv'
create_h5_dataset(fits_path, bands, img_size, out_dir, out_filename, labels_path=labels_path,
                  patch_strategy='all', n_patches=-1)

'''
