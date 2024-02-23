import os
import glob
import numpy as np
import h5py

data_dir = '/arc/projects/unions/HSC_h5/'
out_path = os.path.join(data_dir, 'HSC_zspec_GRIZY_64.h5')
h5_paths = glob.glob(os.path.join(data_dir, 'HSC_zspec_GRIZY_64_*_to_*.h5'))

batch_size = 2048

order = np.argsort([int(p.split('_')[5]) for p in h5_paths])
h5_paths = [h5_paths[i] for i in order]
print('Combining the following files:')
for p in h5_paths:
    print(p)

# Combine individual files into one large .h5 file
with h5py.File(out_path, 'w') as f_out:
    for i, fn in enumerate(h5_paths):
        with h5py.File(fn, 'r') as f_in: 
            for k in f_in.keys():
                #if 'zspec' in k:
                #    continue
                
                if i==0:
                    # Create the same datasets as in file
                    shape = list(f_in[k].shape)
                    shape[0] = 0
                    maxshape = shape.copy()
                    maxshape[0] = None
                    f_out.create_dataset(k, shape=tuple(shape), maxshape=maxshape)

                ds_out = f_out[k]
                num_in = len(f_in[k])
                for j in range(0, num_in, batch_size):
                    
                    # Collect data from in file
                    data_in = f_in[k][j:j+batch_size]
                    num_data = data_in.shape[0]
                    
                    # Resize the dataset to accommodate the new data
                    ds_out.resize(ds_out.shape[0] + num_data, axis=0)
                    
                    # Add the new data
                    ds_out[-num_data:] = data_in

        print(f'Finished combining {i+1} files...')
                    
    print(f'Finished creating {out_path} with {len(ds_out)} samples.')