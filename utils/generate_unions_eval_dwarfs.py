import h5py
import random

root_dir = '/home/a4ferrei/scratch/data/'
dwarf_file = root_dir + 'dr5_eval_set_dwarfs_only.h5' # exists
validation_file = root_dir + 'dr5_eval_set_validation.h5' # exist
dwarf_class_file = root_dir + 'dr5_eval_set_dwarfs_class.h5' # to create

# Open dwarf_file and validation_file
with h5py.File(dwarf_file, 'r') as dwarf_f, h5py.File(validation_file, 'r') as validation_f:
    # Read data from dwarf_file
    dwarf_data = {}
    for key in dwarf_f.keys():
        dwarf_data[key] = dwarf_f[key][:]

    # Read data from validation_file
    validation_data = {}
    for key in validation_f.keys():
        validation_data[key] = validation_f[key][:]

    # Randomly sample the same amount of data from the validation set
    validation_data_sampled = {}
    sample_indices = random.sample(range(len(validation_data['zspec'])), len(dwarf_data['zspec']))
    for key in validation_data.keys():
        validation_data_sampled[key] = [validation_data[key][i] for i in sample_indices]

    # Combine data from both files
    combined_data = {}
    for key in dwarf_data.keys():
        combined_data[key] = dwarf_data[key] + validation_data_sampled[key]

    # Add is_dwarf key to indicate the source of the data
    combined_data['is_dwarf'] = [1] * len(dwarf_data['zspec']) + [0] * len(validation_data_sampled['zspec'])

# Save combined data to dwarf_class_file
with h5py.File(dwarf_class_file, 'w') as dwarf_class_f:
    for key, value in combined_data.items():
        dwarf_class_f.create_dataset(key, data=value)

print(f"Combined data has been saved to '{dwarf_class_file}'.")
