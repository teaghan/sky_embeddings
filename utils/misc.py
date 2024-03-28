import numpy as np
import h5py
import argparse
import torch

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser('Training for Masked Image Modelling', add_help=False)

    # Positional mandatory arguments
    parser.add_argument("model_name", help="Name of model.", type=str)

    # Optional arguments
    
    # How often to display the losses
    parser.add_argument("-v", "--verbose_iters", 
                        help="Number of batch  iters after which to evaluate val set and display output.", 
                        type=int, default=10000)
    
    # How often to display save the model
    parser.add_argument("-ct", "--cp_time", 
                        help="Number of minutes after which to save a checkpoint.", 
                        type=float, default=15)
    
    # Alternate data directory than sky_embeddings/data/
    parser.add_argument("-dd", "--data_dir", 
                        help="Data directory if different from StarNet_SS/data/", 
                        type=str, default=None)
    
    return parser

def calculate_n_samples_per_class(class_counts, num_train, balanced=False):
    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)
    
    if balanced:
        # Determine the maximum number of samples per class without exceeding num_train total
        n_samples = min(num_train // num_classes, min(class_counts.values()))
        n_samples_per_class = {cls: n_samples for cls in class_counts}
    else:
        n_samples_per_class = {cls: int((count / total_samples) * num_train) for cls, count in class_counts.items()}
    
    return n_samples_per_class

def select_training_indices(data_file_path, num_train, balanced=False):
    # Load the 'class' dataset from the HDF5 file
    with h5py.File(data_file_path, 'r') as f:
        class_data = np.array(f['class'])
    
    # Determine the total number of samples for each class
    unique, counts = np.unique(class_data, return_counts=True)
    class_counts = dict(zip(unique, counts))
    
    # Calculate `n_samples` for each class based on the balanced parameter
    n_samples_per_class = calculate_n_samples_per_class(class_counts, num_train, balanced)
    
    # Collect the first `n_samples` indices for each class
    training_indices = []
    for cls, n_samples in n_samples_per_class.items():
        indices_of_class = np.where(class_data == cls)[0][:n_samples]
        training_indices.extend(indices_of_class.tolist())
    
    return training_indices

def central_indices(tensor_2d, n):
    """
    Returns the indices of the central 'n' pixels of a 2D tensor.
    
    Parameters:
    tensor_2d (Tensor): A 2D tensor from which to find central indices.
    n (int): Number of central pixels to find.

    Returns:
    Tensor: Indices of the central 'n' pixels.
    """
    # Ensure 'n' is a square number to form a square patch of pixels
    side_length = int(n ** 0.5)
    if side_length ** 2 != n:
        raise ValueError("n must be a perfect square to form a square patch of pixels.")
    
    # Calculate the center of the tensor
    center_y, center_x = tensor_2d.shape[0] // 2, tensor_2d.shape[1] // 2
    
    # Calculate start and end indices for the slice
    start_y = center_y - side_length // 2
    end_y = start_y + side_length
    start_x = center_x - side_length // 2
    end_x = start_x + side_length
    
    # Create meshgrid of indices and then flatten
    yy, xx = torch.meshgrid(torch.arange(start_y, end_y), torch.arange(start_x, end_x), indexing="ij")
    indices = torch.stack((yy.flatten(), xx.flatten()), dim=1)
    
    return indices

def select_centre(latent, n_patches):
    '''
    Grab the central n_patches from a set of latent features.
    
    Parameters:
    latent (Tensor): A 3D tensor from which to select the centre from (b, number of original patches, n_features)
    n_patches (int): Number of central patches to find.

    Returns:
    Tensor: A 3D tensor of the central latent features (b, n_patches, n_features)
    '''
    
    total_n_patches = latent.shape[1]
    n_patches_per_side = int(total_n_patches**0.5)
    patch_indices = torch.arange(total_n_patches).reshape((n_patches_per_side, n_patches_per_side))
    
    indices = central_indices(patch_indices, n=n_patches)
    patch_indices = patch_indices[indices[:,0],indices[:,1]]
    return latent[:,patch_indices]

def calculate_snr(images, n_central_pix):
    """
    Calculate the channel-wise Signal-to-Noise Ratio (SNR) for a batch of images.

    This function computes the SNR by measuring the ratio of the mean of the central 
    `n_central_pix` x `n_central_pix` pixels to the standard deviation of the 
    surrounding pixels for each channel in each image.

    Parameters:
    images (numpy.ndarray): A 4D NumPy array with shape (batch_size, n_channels, img_size, img_size).
                            This array represents the batch of images.
    n_central_pix (int): The size of the square central region from which the mean will be calculated.

    Returns:
    numpy.ndarray: A 2D NumPy array with shape (batch_size, n_channels) containing the SNR values for
                   each channel of each image in the batch.

    Note:
    - The function assumes that `n_central_pix` is smaller than `img_size`.
    - A small value (1e-8) is added to the standard deviation to avoid division by zero.
    """
    batch_size, n_channels, img_size, _ = images.shape

    # Calculate the start and end indices for the central region
    start = (img_size - n_central_pix) // 2
    end = start + n_central_pix

    # Extracting the central region for all images and channels
    central_region = images[:, :, start:end, start:end]

    # Create a mask for the surrounding region
    mask = np.ones((img_size, img_size), dtype=bool)
    mask[start:end, start:end] = False

    # Apply the mask to all images to extract the surrounding region
    surrounding_region = images[:, :, mask].reshape(batch_size, n_channels, -1)

    # Calculate the mean of the central region and the standard deviation of the surrounding region
    mean_central = np.mean(central_region, axis=(2, 3))
    std_surrounding = np.std(surrounding_region, axis=2)

    # Calculate the SNR
    snr = mean_central / (std_surrounding + 1e-8)  # Added a small value to avoid division by zero

    return snr

def h5_snr(h5_path, n_central_pix=8, batch_size=5000, num_samples=None):
        
    snr_vals = []
    with h5py.File(h5_path, "r") as f:   
        if num_samples is None:
            num_samples = len(f['cutouts'])
        if num_samples<batch_size:
            cutouts = f['cutouts'][:num_samples]
            snr_vals.append(calculate_snr(cutouts, n_central_pix))
        else:
            for i in range(0, num_samples, batch_size):
                end = min([num_samples, i+batch_size])
                cutouts = f['cutouts'][i:end]
                snr_vals.append(calculate_snr(cutouts, n_central_pix))
    
    return np.concatenate(snr_vals)