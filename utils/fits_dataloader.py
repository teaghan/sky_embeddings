import numpy as np
import torch
import h5py
from torchvision.transforms import v2
import glob
from astropy.io import fits

def build_fits_dataloader(fits_paths, bands, norm_type, batch_size, num_workers,
                     img_size=64, cutouts_per_tile=1024, pix_mean=None, pix_std=None, 
                     augment=False, shuffle=True):
    '''Return a dataloader to be used during training.'''

    if augment:
        # Define augmentations
        transforms = v2.Compose([v2.GaussianBlur(kernel_size=5, sigma=(0.1,1.5)),
                                 v2.RandomResizedCrop(size=(img_size, img_size), scale=(0.8,1), antialias=True),
                                 v2.RandomHorizontalFlip(p=0.5),
                                 v2.RandomVerticalFlip(p=0.5)])
    else:
        transforms = None
    
    # Build dataset
    dataset = FitsDataset(fits_paths, bands=bands, img_size=img_size, 
                          cutouts_per_tile=cutouts_per_tile,
                          batch_size=batch_size, shuffle=shuffle,
                          norm=norm_type, transform=transforms,
                          global_mean=pix_mean, global_std=pix_std)

    # Build dataloader
    return torch.utils.data.DataLoader(dataset, batch_size=1, 
                                       shuffle=shuffle, num_workers=num_workers,
                                       pin_memory=True)

def find_HSC_bands(fits_paths, bands):
    '''An HSC specific function that returns a list of file paths with all of the requested bands.'''

    filenames = []
    for fits_path in fits_paths:
        # Look for fits files
        fits_files = sorted(glob.glob(f"{fits_path}/calexp-HSC-*.fits"))
        
        # Convert '/arc/projects/ots/pdr3_dud/calexp-HSC-I-9707-4%2C0.fits' to 9707-4%2C0.fits
        unique_patches = list(set(['-'.join(x.split('-')[-2:]) for x in fits_files]))
        unique_patches = sorted(unique_patches)
        
        # Make it hashable
        set_fits_files = set(fits_files)
    
        # Sort file names
        for t in unique_patches:
            potential_files = [f'{fits_path}/calexp-HSC-{b}-{t}' for b in bands]
    
            # `f in set_fits_files` is O(n) if fits_files is a list,
            # ~O(1) if fits_files is a hash table
            if (all([f in set_fits_files for f in potential_files])):
                filenames.append(potential_files)
    print(f"Found {len(filenames)} patches with the {bands} bands.")

    return filenames

def random_cutouts(input_array, img_size, n_cutouts):
    """
    Generate random cutouts from a larger 3D numpy array.

    Args:
    - input_array: The larger 3D numpy array of shape (C, H, W).
    - img_size: The desired size of each cutout in both height and width.
    - n_cutouts: The number of random cutouts to generate.

    Returns:
    - A numpy array of shape (n_cutouts, C, img_size, img_size).
    """
    C, H, W = input_array.shape
    # Pre-allocate numpy array for efficiency
    cutouts = np.zeros((n_cutouts, C, img_size, img_size), dtype=input_array.dtype)

    # Generate random coordinates for all cutouts in a batch
    h_starts = np.random.randint(0, H - img_size + 1, size=n_cutouts)
    w_starts = np.random.randint(0, W - img_size + 1, size=n_cutouts)

    for i, (h_start, w_start) in enumerate(zip(h_starts, w_starts)):
        # Fill the pre-allocated array directly
        cutouts[i] = input_array[:, h_start:h_start+img_size, w_start:w_start+img_size]

    return cutouts

class FitsDataset(torch.utils.data.Dataset):
    
    """
    Dataset loader for the cutout datasets.
    """

    def __init__(self, fits_paths, bands=['G','R','I','Z','Y'], img_size=64, cutouts_per_tile=1024,
                 batch_size=64, shuffle=True, norm=None, transform=None, 
                 global_mean=0.1, global_std=2., pixel_min=None, pixel_max=None):
        
        self.fits_paths = fits_paths
        self.img_size = img_size
        self.cutouts_per_tile = cutouts_per_tile
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.norm = norm
        self.transform = transform
        self.global_mean = global_mean
        self.global_std = global_std 
        self.pixel_min = pixel_min
        self.pixel_max = pixel_max

        # Find names of patch fits files
        self.band_filenames = find_HSC_bands(fits_paths, bands)
                        
    def __len__(self):
        # The number of fits patches with all of the requested bands
        return len(self.band_filenames)
    
    def __getitem__(self, idx):

        # Grab fits filenames
        patch_filenames = self.band_filenames[idx]
        
        # Load all channels of the patch of sky
        cutouts = []
        for fn in patch_filenames:
            cutouts.append(fits.open(fn, mode='readonly', ignore_missing_simple=True)[1].data)
        # Organize into (C, H, W)
        cutouts = np.array(cutouts)

        # Split into a grid of cutouts based on img_size and overlap
        cutouts = random_cutouts(cutouts, self.img_size, self.cutouts_per_tile)

        # Shuffle images
        if self.shuffle:
            permutation = np.random.permutation(cutouts.shape[0])
            cutouts = cutouts[permutation]

        # Remove any NaN pixel values
        cutouts[np.isnan(cutouts)] = 0.

        # Clip pixel values
        if self.pixel_min is not None:
            cutouts[cutouts<self.pixel_min] = self.pixel_min
        if self.pixel_max is not None:
            cutouts[cutouts>self.pixel_max] = self.pixel_max

        # Apply any augmentations
        cutouts = torch.from_numpy(cutouts)
        if self.transform is not None:
            cutouts = self.transform(cutouts)
            
        if self.norm=='minmax':
            # Normalize each sample between 0 and 1
            sample_min = torch.amin(cutouts, dim=(1,2,3), keepdim=True)
            sample_max = torch.amax(cutouts, dim=(1,2,3), keepdim=True)
            cutouts = (cutouts - sample_min) / (sample_max - sample_min + 1e-6)
        elif self.norm=='zscore':
            # Normalize each sample to have zero mean and unit variance
            sample_mean = torch.mean(cutouts, dim=(1,2,3), keepdim=True)
            sample_std = torch.std(cutouts, dim=(1,2,3), keepdim=True)
            cutouts = (cutouts - sample_mean) / (sample_std + 1e-6)
        elif self.norm=='global':
            # Normalize dataset to have zero mean and unit variance
            cutouts = (cutouts - self.global_mean) / self.global_std

        # Sort into M batches of batch_size
        M = cutouts.shape[0] // self.batch_size
        C = cutouts.shape[1]
        cutouts = cutouts[:M * self.batch_size].reshape((M, self.batch_size, C, self.img_size, self.img_size))

        return cutouts