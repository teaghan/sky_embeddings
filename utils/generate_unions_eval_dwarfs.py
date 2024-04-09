import h5py
import random

root_dir = '/home/a4ferrei/scratch/data/'
dwarf_file = root_dir + 'dr5_eval_set_dwarfs_only.h5' # exists
validation_file = root_dir + 'dr5_eval_set_validation.h5' # exist
dwarf_class_file = root_dir + 'dr5_eval_set_dwarfs_class.h5' # to create

# Open dwarf_file and validation_file
with h5py.File(dwarf_file, 'r') as dwarf_f, h5py.File(validation_file, 'r') as validation_f:
    print('dwarfs')
    # Read data from dwarf_file
    dwarf_data = {}
    for key in dwarf_f.keys():
        print(key)
        if key == 'images':
            dwarf_data['cutouts'] = dwarf_f[key][:]
        else:
            dwarf_data[key] = dwarf_f[key][:]

    print('\nvalidation')
    # Read data from validation_file
    validation_data = {}
    for key in validation_f.keys():
        print(key)
        validation_data[key] = validation_f[key][:]

    key_0 = 'ra'

    # Randomly sample the same amount of data from the validation set
    validation_data_sampled = {}
    sample_indices = random.sample(range(len(validation_data[key_0])), len(dwarf_data[key_0]))
    for key in validation_data.keys():
        validation_data_sampled[key] = [validation_data[key][i] for i in sample_indices]

    # Combine data from both files
    combined_data = {}
    for key in validation_f.keys():
        combined_data[key] = dwarf_data[key] + validation_data_sampled[key]

    # Add is_dwarf key to indicate the source of the data
    combined_data['is_dwarf'] = [1] * len(dwarf_data[key_0]) + [0] * len(validation_data_sampled[key_0])

# Save combined data to dwarf_class_file
with h5py.File(dwarf_class_file, 'w') as dwarf_class_f:
    for key, value in combined_data.items():
        dwarf_class_f.create_dataset(key, data=value)

print(f"Combined data has been saved to '{dwarf_class_file}'.")
