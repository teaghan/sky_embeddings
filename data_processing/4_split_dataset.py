import os
import glob
import numpy as np
import h5py

data_dir = '/home/obriaint/scratch/sky_embeddings/data/'
in_path = os.path.join(data_dir, 'HSC_dud_galaxy_GIRYZ7610_64.h5')
subsets = ['train', 'val', 'test']
subset_frac = [0.8,0.1,0.1]

# Determine output filenames
in_path_split = in_path.split('.')
out_paths = ['.'.join([in_path_split[0]+f'_{s}', in_path_split[1]]) for s in subsets]


with h5py.File(in_path, 'r') as f_in:
    keys = list(f_in.keys())
    num_samples = len(f_in[keys[0]])

    # Spit randomly
    random_indices = np.arange(0,num_samples)
    np.random.shuffle(random_indices)
    
    start = 0
    for out_path, frac in zip(out_paths, subset_frac):
        end = int(start + num_samples*frac)        
        with h5py.File(out_path, 'w') as f_out:
            for k in keys:
                # Create the same datasets as in file
                shape = list(f_in[k].shape)
                shape[0] = end-start
                f_out.create_dataset(k, shape=tuple(shape))

                # Copy data subset one by one...
                ds_out = f_out[k]
                for i, j in enumerate(random_indices[start:end]):
                    
                    # Add the new data
                    ds_out[i] = f_in[k][j]
        start = end
        print(f'Finished creating {out_path} with {int(num_samples*frac)} samples.')
