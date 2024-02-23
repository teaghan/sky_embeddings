import numpy as np
import os
import h5py
from utils import getPatches, cutouts_to_hdf5, get_cutouts_from_patch
import pandas as pd
import glob
import shutil
import gc

def create_h5_dataset(fits_path, bands, img_size, out_dir, out_filename, scratch_dir='/scratch',
                      labels_path=None, create_grid=False, patch_strategy='all', 
                      n_patches=-1, patch_low_bound=0, patch_high_bound=-1):

    # Temporary storage location
    tmp_dir = os.path.join(scratch_dir,'tmp_'+''.join(np.random.randint(0,9,size=(6,)).astype(str)))
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.mkdir(tmp_dir)

    # Find fits files and get ready for loading
    kwargs = {'data_path':fits_path,
              'bands':bands,
              'patch_strategy':patch_strategy,
              'n_patches':n_patches, 
              'patch_low_bound': patch_low_bound,
              'patch_high_bound': patch_high_bound,
              'cache_dir':tmp_dir}

    if (labels_path is not None) & (not create_grid):
        # Load csv file with ra, dec, and zspec info 
        labels = pd.read_csv(labels_path)
    
    # Loop through fits patches
    #print(f'Looking through {len(patches)} patches...')
    for i, p in enumerate(getPatches(**kwargs)):

        #print(i, p.files)
        # Load patch
        p.load()
        
        if create_grid:
            # Base grid size on size of patch and img_size
            p.generate_grid(img_size)
        else:
            # Find ra and decs that are within the path
            p.match_catalogue(labels)
        # Grab cutouts
        cutouts = get_cutouts_from_patch(p, collect_var=True)
        if len(cutouts)>0:
            cutouts_to_hdf5(tmp_dir, cutouts, collect_var=True)

        if (i+1)%100==0:
            print(f'{i+1} patches complete...')
        # Close fits files
        p.__del__()
        del p
        del cutouts
        gc.collect()

    print('Finished creating cutouts.')
    '''
    tmp_dir = '../data/tmp_734431/'
    '''
    tmp_h5_paths = glob.glob(os.path.join(tmp_dir, '*.h5'))
    if len(tmp_h5_paths)>0:
        print(f'Creating {out_filename} data file from {len(tmp_h5_paths)} small files...')
        # Combine individual files into one large .h5 file
        with h5py.File(os.path.join(scratch_dir, out_filename), "w") as f_out:
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

    print('Copying file to final destination')
    shutil.copyfile(os.path.join(scratch_dir, out_filename), 
                    os.path.join(out_dir, out_filename))
    # Remove temp files
    #shutil.rmtree(tmp_dir)

img_size = 64
fits_path = '/arc/projects/ots/pdr3_dud'
out_dir = '/arc/projects/unions/HSC_h5/'
bands = ['G','R','I','Z','Y']
'''
out_filename = 'HSC_galaxies_GRIZY_64.h5'
labels_path = '../data/galaxies.csv'
create_h5_dataset(fits_path, bands, img_size, out_dir, out_filename, labels_path=labels_path,
                  patch_strategy='all', n_patches=-1)

out_filename = 'HSC_qso_GRIZY_64.h5'
labels_path = '../data/qso.csv'
create_h5_dataset(fits_path, bands, img_size, out_dir, out_filename, labels_path=labels_path,
                  patch_strategy='all', n_patches=-1)

out_filename = 'HSC_dwarf_galaxies_GRIZY_64.h5'
labels_path = '../data/dwarf_galaxies.csv'
create_h5_dataset(fits_path, bands, img_size, out_dir, out_filename, labels_path=labels_path,
                  patch_strategy='all', n_patches=-1)

out_filename = 'HSC_strong_lens_candidates_GRIZY_64.h5'
labels_path = '../data/strong_lens_candidates.csv'
create_h5_dataset(fits_path, bands, img_size, out_dir, out_filename, labels_path=labels_path,
                  patch_strategy='all', n_patches=-1)

out_filename = 'HSC_unkown_GRIZY_64.h5'
labels_path = '../data/unkown.csv'
create_h5_dataset(fits_path, bands, img_size, out_dir, out_filename, labels_path=labels_path,
                  patch_strategy='all', n_patches=-1)


batch_size = 40
labels_path = '../data/stars.csv'
for i in range(0, 1477, batch_size):
    low = i
    high = i+batch_size
    out_filename = f'HSC_stars_GRIZY_64_{low}_to_{high}.h5'
    create_h5_dataset(fits_path, bands, img_size, out_dir, out_filename, labels_path=labels_path,
                      patch_strategy='n_to_n', patch_low_bound=low, patch_high_bound=high)
'''
batch_size = 40
labels_path = '../data/zspec.csv'
for i in range(0, 1477, batch_size):
    low = i
    high = i+batch_size
    out_filename = f'HSC_zspec_GRIZY_64_{low}_to_{high}.h5'
    create_h5_dataset(fits_path, bands, img_size, out_dir, out_filename, labels_path=labels_path,
                      patch_strategy='n_to_n', patch_low_bound=low, patch_high_bound=high)
'''
batch_size = 40
for i in range(1240, 1477, batch_size):
    low = i
    high = i+batch_size
    out_filename = f'HSC_grid_GRIZY_64_{low}_to_{high}.h5'
    create_h5_dataset(fits_path, bands, img_size, out_dir, out_filename, create_grid=True,
                      patch_strategy='n_to_n', patch_low_bound=low, patch_high_bound=high)
'''