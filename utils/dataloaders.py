import random
import numpy as np
import torch
import h5py
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2
import os
import glob
from astropy.io import fits
from astropy.wcs import WCS

# Custom brightness adjustment for images
def adjust_brightness(img, brightness_factor):
    return img * brightness_factor

# Custom transform that applies brightness adjustment with a random factor
class RandomBrightnessAdjust:
    def __init__(self, brightness_range=(0.8, 1.2)):
        self.brightness_range = brightness_range

    def __call__(self, img):
        brightness_factor = random.uniform(*self.brightness_range)
        return adjust_brightness(img, brightness_factor)

# Custom brightness adjustment for images
def add_noise(img, noise_factor):
    return img + torch.randn_like(img) * noise_factor

# Custom transform that applies random noise
class RandomNoise:
    def __init__(self, noise_range=(0., 0.1)):
        self.noise_range = noise_range

    def __call__(self, img):
        noise_factor = random.uniform(*self.noise_range)
        return add_noise(img, noise_factor)

class RandomChannelNaN:
    def __init__(self, max_channels=1):
        """
        Initializes the RandomChannelNaN transformation with a maximum number of channels to replace.

        Args:
            max_channels (int): The maximum number of channels that can be replaced with NaN values.
        """
        self.max_channels = max_channels

    def __call__(self, img):
        """
        Applies the RandomChannelNaN transformation to an input image tensor, randomly replacing up to `max_channels` channels with NaN values.

        Args:
            img (torch.Tensor): The input image tensor. Expected shape: (C, H, W) or (B, C, H, W) where
                C is the number of channels, H is the height, and W is the width, and B is the batch size.

        Returns:
            torch.Tensor: The augmented image tensor with a random number of channels replaced by NaN values, up to `max_channels`.
        """
        # Ensure the input is a tensor
        if not isinstance(img, torch.Tensor):
            raise TypeError("img must be a torch.Tensor")

        # Check if the tensor has a batch dimension
        if img.dim() == 3:
            img = img.unsqueeze(0)  # Add a batch dimension if it's not there

        B, C, H, W = img.shape

        # Ensure max_channels is not greater than the number of channels in the image
        if self.max_channels > C:
            raise ValueError(f"max_channels must be less than or equal to the number of channels in the image. Got {self.max_channels} for an image with {C} channels.")

        for b in range(B):
            # Randomly decide the number of channels to replace, up to max_channels
            n_channels_to_replace = random.randint(0, self.max_channels)

            # Randomly select n_channels_to_replace to replace with NaN
            channels_to_replace = random.sample(range(C), n_channels_to_replace)

            for c in channels_to_replace:
                img[b, c, :, :] = torch.nan

        if img.shape[0] == 1:
            img = img.squeeze(0)  # Remove the batch dimension if it was originally not there

        return img

# Define the augmentation pipeline
def get_augmentations(img_size=64, flip=True, crop=True, brightness=0.8, noise=0.01, nan_channels=2):
    transforms = []
    if flip:
        transforms.append(v2.RandomHorizontalFlip())
        transforms.append(v2.RandomVerticalFlip())
    if crop:
        transforms.append(v2.RandomResizedCrop(size=(img_size, img_size), 
                                               scale=(0.8, 1.0), 
                                               ratio=(0.9, 1.1), antialias=True))
    if brightness is not None:
        transforms.append(RandomBrightnessAdjust(brightness_range=(brightness, 1/brightness)))
    if noise is not None:
        transforms.append(RandomNoise(noise_range=(0., noise)))
    if nan_channels is not None:
        transforms.append(RandomChannelNaN(max_channels=nan_channels))
        
    return v2.Compose(transforms)

def build_fits_dataloader(fits_paths, bands, min_bands, batch_size, num_workers,
                          patch_size=8, max_mask_ratio=None, 
                          img_size=64, cutouts_per_tile=1024, use_calexp=True,
                          augment=False, brightness=0.8, noise=0.01, nan_channels=2, 
                          shuffle=True, ra_dec=True, transforms=None,
                          use_overlap=False, overlap=0.5):
    '''Return a dataloader to be used during training.'''

    if (transforms is None) and augment:
        transforms = get_augmentations(img_size=img_size, flip=True, crop=True, 
                                       brightness=brightness, noise=noise, nan_channels=nan_channels)
    
    # Build dataset
    dataset = FitsDataset(fits_paths, patch_size=patch_size, 
                          max_mask_ratio=max_mask_ratio,
                          bands=bands, min_bands=min_bands, img_size=img_size, 
                          cutouts_per_tile=cutouts_per_tile,
                          batch_size=batch_size, ra_dec=ra_dec,
                          transform=transforms, use_calexp=use_calexp,
                          use_overlap=use_overlap, overlap=overlap)

    # Build dataloader
    return torch.utils.data.DataLoader(dataset, batch_size=1, 
                                       shuffle=shuffle, num_workers=num_workers,
                                       drop_last=True, pin_memory=True)

def build_h5_dataloader(filename, batch_size, num_workers, patch_size=8, num_channels=5, 
                        max_mask_ratio=None, label_keys=None, img_size=64, num_patches=None, 
                        augment=False, brightness=0.8, noise=0.01, nan_channels=2, 
                        shuffle=True, indices=None, transforms=None):

    if (transforms is None) and augment:
        transforms = get_augmentations(img_size=img_size, flip=True, crop=True, 
                                       brightness=brightness, noise=noise, nan_channels=nan_channels)

    
    # Build dataset
    dataset = H5Dataset(filename, img_size=img_size, patch_size=patch_size, 
                        num_channels=num_channels, max_mask_ratio=max_mask_ratio,
                        num_patches=num_patches, 
                        label_keys=label_keys, transform=transforms, indices=indices)

    # Build dataloader
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                       shuffle=shuffle, num_workers=num_workers,
                                       pin_memory=True)

class MaskGenerator:
    """
    A class for generating channel-wise masks for image patches.

    This class generates binary masks to be applied to images for tasks such as image inpainting or data augmentation.
    Masks are generated based on an input size, patch size, maximum mask ratio, and number of mask channels specified.
    The generated masks can be used to selectively hide or reveal parts of an image by masking out certain patches.
    Each channel will have a unique set of patches masked; however, the same number of patches will be masked in each channel.

    Attributes:
        input_size (int): The size of the input image (assumed square) in pixels.
        patch_size (int): The size of each square patch in pixels.
        max_mask_ratio (float): The maximum ratio of the image that can be masked. 
                                Values should be between 0 and 1, where 1 means 100% of the image can be masked.
        num_mask_chans (int): The number of mask channels to generate. For grayscale masks, this should be 1.
        n_patches (int): The number of patches per dimension, calculated as input_size / patch_size.
        token_count (int): The total number of patches in the image, calculated as n_patches^2.

    Methods:
        __call__():
            Generates and returns a mask based on the initialized parameters.
            The method randomly decides the masking ratio for each call within the limits of the max_mask_ratio attribute.
            Masks are generated separately for each channel specified by num_mask_chans and then combined.
            The mask is scaled up to the original image size by repeating the mask values for each patch.

    Returns:
        torch.Tensor: A tensor representing the generated mask. If num_mask_chans is 1, the channel dimension is removed.
                      The shape of the tensor is either (num_mask_chans, input_size, input_size) for multiple channels 
                      or (input_size, input_size) for a single channel.
    """
    def __init__(self, input_size=192, patch_size=4, max_mask_ratio=0.9, num_mask_chans=1):
        self.input_size = input_size
        self.patch_size = patch_size
        self.max_mask_ratio = max_mask_ratio
        self.num_mask_chans = num_mask_chans
        
        # Number of patches per dimension
        self.n_patches = self.input_size // self.patch_size

        # Total number of patches
        self.token_count = self.n_patches ** 2
        
    def __call__(self):

        # Randomly set the mask ratio for this sample
        mask_ratio = torch.rand(1).item() * self.max_mask_ratio
        mask_count = int(torch.ceil(torch.tensor(self.token_count * mask_ratio)).item())

        # Iterate over channels
        masks = torch.zeros((self.num_mask_chans, self.token_count), dtype=torch.int)
        for i in range(self.num_mask_chans):
            # Indices of randomly masked patches for this channel
            mask_idx = torch.randperm(self.token_count)[:mask_count]

            # Patches with mask=1 will be masked 
            masks[i, mask_idx] = 1

        # Scale to image size
        masks = masks.view(self.num_mask_chans, self.n_patches, self.n_patches)
        masks = masks.repeat_interleave(self.patch_size, dim=1).repeat_interleave(self.patch_size, dim=2)

        if self.num_mask_chans == 1:
            # Remove channel dimension if only one channel mask is generated
            return masks.squeeze(0)
        return masks

class H5Dataset(torch.utils.data.Dataset):
    
    """
    A PyTorch dataset class for loading and transforming data from H5 files, specifically designed for astronomical
    cutouts or similar types of image datasets. This dataset loader supports dynamic masking, pixel value clipping,
    optional positional channels, and custom transformations.

    The class can handle datasets where each sample consists of an image (cutout) and optionally additional labels,
    such as RA (Right Ascension) and Dec (Declination) or other specified metadata. It also supports generating masks
    for images dynamically using a `MaskGenerator` instance based on specified criteria like image size, patch size,
    and maximum mask ratio.

    Parameters:
        data_file (str): Path to the H5 file containing the dataset.
        img_size (int): The size to which images will be resized or cropped.
        patch_size (int): The size of each patch for the purpose of mask generation.
        num_channels (int): The number of channels in the images.
        max_mask_ratio (float, optional): The maximum ratio of the image that can be masked by the `MaskGenerator`.
        num_patches (int, optional): The number of patches into which the positional channel is divided.
        label_keys (list of str, optional): Keys to retrieve additional labels from the H5 file.
        transform (callable, optional): A function/transform that takes in an image and returns a transformed version.
        pixel_min (float, optional): The minimum pixel value for clipping.
        pixel_max (float, optional): The maximum pixel value for clipping.
        indices (list of int, optional): A list of indices specifying which samples to include in the dataset.
                                         If None, all samples in the file are included.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Returns the sample at index `idx` along with its mask and labels. The sample consists of an
                          image (cutout), a dynamically generated mask if `max_mask_ratio` is specified, and labels
                          (either RA and Dec or those specified by `label_keys`).
    """

    def __init__(self, data_file, img_size, patch_size, num_channels, max_mask_ratio, 
                 num_patches=None, label_keys=None, 
                 transform=None, pixel_min=-3., pixel_max=None, indices=None):
        
        self.data_file = data_file
        self.transform = transform
        self.img_size = img_size
        self.num_patches = num_patches
        self.label_keys = label_keys
        self.pixel_min = pixel_min
        self.pixel_max = pixel_max
        self.indices = indices
        self.max_mask_ratio = max_mask_ratio

        if max_mask_ratio is not None:
            self.mask_generator = MaskGenerator(input_size=img_size,
                                                patch_size=patch_size,
                                                max_mask_ratio=max_mask_ratio,
                                                num_mask_chans=num_channels)
        else:
            self.mask_generator = None
                        
    def __len__(self):
        if self.indices is not None:
            # Custom set of indices
            return len(self.indices)
        else:
            with h5py.File(self.data_file, "r") as f:    
                num_samples = len(f['cutouts'])
            return num_samples
    
    def __getitem__(self, idx):
        if self.indices is not None:
            # Use custom set of indices
            idx = self.indices[idx]
        with h5py.File(self.data_file, "r") as f: 
            # Load cutout
            cutout = f['cutouts'][idx]

            # Clip pixel values
            if self.pixel_min is not None:
                cutout[cutout<self.pixel_min] = self.pixel_min
            if self.pixel_max is not None:
                cutout[cutout>self.pixel_max] = self.pixel_max

            if (np.array(cutout.shape[1:])>self.img_size).any():
                # Select central cutout
                cutout = extract_center(cutout, self.img_size)

            # Load RA and Dec
            ra_dec = torch.from_numpy(np.asarray([f['ra'][idx], f['dec'][idx]]).astype(np.float32))

            # Load labels
            if self.label_keys is not None:
                labels = [f[k][idx] for k in self.label_keys]
                if 'class' in self.label_keys:
                    labels = torch.from_numpy(np.asarray(labels).astype(np.int64)).long()
                else:
                    labels = torch.from_numpy(np.asarray(labels).astype(np.float32))

        cutout = torch.from_numpy(cutout).to(torch.float32)
        # Apply any augmentations, etc.
        if self.transform is not None:
            cutout = self.transform(cutout)

        if self.mask_generator is not None:
            # Generate random mask
            mask = self.mask_generator()
        else:
            mask = torch.zeros_like(cutout)

        if self.label_keys is None:
            return cutout, mask, ra_dec
        else:
            return cutout, mask, ra_dec, labels
            

def find_HSC_bands(fits_paths, bands, min_bands=2, verbose=1, use_calexp=True):
    '''
    Searches for HSC (Hyper Suprime-Cam) survey FITS files across specified paths and returns a nested list of filenames 
    that contain at least a minimum number of color bands per sky patch. Optimized to minimize filesystem operations and
    efficiently organize files by patch and band.

    Parameters:
    - fits_paths (list of str): Paths to search for HSC FITS files.
    - bands (list of str): The color bands to search for (e.g., ['G', 'R', 'I', 'Z', 'Y']).
    - min_bands (int, optional): The minimum number of color bands required for a patch to be included. Defaults to 2.
    - use_calexp (bool, optional): Determines whether to include files with 'calexp-' prefix. Defaults to True.

    Returns:
    - list of lists: A nested list where each sublist contains the file paths for the bands found for a patch. 
      If a particular color band doesn't exist for a given patch, it is replaced by 'None'. The order of the filenames 
      in each sublist matches the order of the bands provided.
    '''
    
    patch_files = {}  # Dictionary to store available bands for each patch

    for fits_path in fits_paths:
        fits_files = glob.glob(f"{fits_path}/*.fits")

        for file_path in fits_files:
            file_name = file_path.split('/')[-1]  # Extract just the filename
            # Determine if file matches the calexp condition
            if (use_calexp and file_name.startswith('calexp-')) or (not use_calexp and not file_name.startswith('calexp-')):
                # Extract band and patch identifier from the filename
                parts = file_name.split('-')
                if len(parts)<3:
                    continue
                band = parts[-3]
                patch = '-'.join(parts[-2:])
                
                if band in bands:
                    if patch not in patch_files:
                        patch_files[patch] = {b: 'None' for b in bands}
                    patch_files[patch][band] = file_path

    # Filter patches by the minimum number of available bands and organize the filenames
    filenames = []
    for patch, available_bands in patch_files.items():
        current_patch_files = [available_bands[band] for band in bands]
        if len([f for f in current_patch_files if f != 'None']) >= min_bands:
            filenames.append(current_patch_files)

    if verbose:
        print(f"Found {len(filenames)} patches with at least {min_bands} of the {bands} bands.")

    return filenames

def load_fits_bands(patch_filenames, return_wc=False):
    """
    Load FITS files from a list of filenames representing different bands of astronomical images.
    If a file cannot be loaded or is specified as 'None', it is replaced with an array of np.nan values.
    The function ensures all arrays, whether loaded from files or filled with np.nan, have the same shape,
    allowing for consistent handling of multi-band astronomical data. The first valid file encountered
    determines the reference shape for the np.nan arrays.

    Parameters:
    - patch_filenames (list of str): A list containing the filenames of the FITS files to be loaded.
      Filenames should be full paths. A filename can be 'None' to indicate a missing file for a band,
      in which case it will be replaced with an array of np.nan values of the same shape as other bands.

    Returns:
    - numpy.ndarray: A 3D numpy array where the first dimension corresponds to the different bands
      (channels), and the remaining dimensions correspond to the spatial dimensions of the images.
      The array is organized as (C, H, W), where C is the number of channels (bands), H is the height,
      and W is the width of the images. If any band is missing, its corresponding array will be filled
      with np.nan values.

    Raises:
    - Exception: If there are issues opening a file, an error message is printed, and the process continues,
      replacing the problematic file with an array of np.nan values. The function aims to complete loading
      as much data as possible, even in the presence of errors.
    """
    
    imgs = []
    reference_shape = None  # Initially unknown
    wc_collected = False
    for fn in patch_filenames:
        if fn == 'None':
            # For now, just append a placeholder (None) for missing files
            imgs.append(None)
        else:
            try:
                # Attempt to open the FITS file
                with fits.open(fn, mode='readonly', ignore_missing_simple=True) as hdul:
                    data = hdul[1].data
                    if reference_shape is None:
                        reference_shape = data.shape  # Found our reference shape
                    imgs.append(data)

                    # Collect pixel to world coord
                    if not wc_collected:
                        if return_wc:
                            wcs = WCS(hdul[1].header)
                            # Return function for determining RA and Dec from pixel coords
                            def pix_to_radec(x, y):
                                # The ordering of the axes in the fits files is a bit 
                                # confusing to me, but I'm pretty sure this is right...
                                return wcs.all_pix2world(x, y, 0)
                        else:
                            pix_to_radec = None
                        wc_collected = True
                        
            except Exception as e:
                # Handle the case where the FITS file cannot be opened
                print(f"Error opening {fn}: {e}")
                imgs.append(None)

    # Now, ensure all placeholders are replaced with np.nan arrays of the correct shape
    for i, item in enumerate(imgs):
        if item is None:
            imgs[i] = np.full(reference_shape, np.nan)

    # Organize into (C, H, W) and convert to a single NumPy array
    return np.stack(imgs), pix_to_radec
    
def random_cutouts(input_array, img_size, n_cutouts, pix_to_radec=None):
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

    if pix_to_radec is not None:
        # Collect RA and Dec at centre of each cutout
        ra, dec = pix_to_radec(h_starts+img_size//2, w_starts+img_size//2)
        return cutouts, np.vstack((ra, dec)).T

    return cutouts

def generate_overlap_coords(img_shape, cutout_size, overlap):
    """
    Generate the top-left coordinates for overlapping cutouts.

    Parameters:
    - img_shape: Tuple (H, W) of the input image dimensions.
    - cutout_size: Size of the cutouts.
    - overlap: Overlap percentage (0 to 1).

    Returns:
    - List of (x, y) coordinates for the top-left corner of each cutout.
    """
    H, W = img_shape
    step_size = int(cutout_size * (1 - overlap))
    coords = [(i, j) for i in range(0, H - cutout_size + 1, step_size)
                     for j in range(0, W - cutout_size + 1, step_size)]

    # Ensure we cover the right and bottom edges
    if H % step_size != 0:
        for j in range(0, W - cutout_size + 1, step_size):
            coords.append((H - cutout_size, j))
    if W % step_size != 0:
        for i in range(0, H - cutout_size + 1, step_size):
            coords.append((i, W - cutout_size))
    if (H % step_size != 0) and (W % step_size != 0):
        coords.append((H - cutout_size, W - cutout_size))

    return coords

def overlapping_cutouts(input_array, img_size, overlap, pix_to_radec=None):
    """
    Generate overlapping cutouts from a larger 3D numpy array.

    Args:
    - input_array: The larger 3D numpy array of shape (C, H, W).
    - img_size: The desired size of each cutout in both height and width.
    - overlap: The percentage overlap between cutouts.

    Returns:
    - A numpy array of shape (n_cutouts, C, img_size, img_size).
    """
    C, H, W = input_array.shape
    coords = generate_overlap_coords((H, W), img_size, overlap)
    cutouts = np.zeros((len(coords), C, img_size, img_size), dtype=input_array.dtype)

    for i, (h_start, w_start) in enumerate(coords):
        cutouts[i] = input_array[:, h_start:h_start+img_size, w_start:w_start+img_size]

    if pix_to_radec is not None:
        # Collect RA and Dec at centre of each cutout
        h_centers = [h + img_size // 2 for h, w in coords]
        w_centers = [w + img_size // 2 for h, w in coords]
        ra, dec = pix_to_radec(h_centers, w_centers)
        return cutouts, np.vstack((ra, dec)).T

    return cutouts

class FitsDataset(torch.utils.data.Dataset):
    
    """
    A PyTorch dataset class for loading astronomical image data from FITS files, designed to handle multi-band 
    astronomical images and generate cutouts of a specified size. This dataset supports dynamic mask generation, 
    pixel value clipping, and custom transformations. It's particularly suited for tasks involving astronomical 
    image analysis where inputs are large sky surveys in FITS format.

    Parameters:
        fits_paths (list of str): A list of directories containing the FITS files.
        patch_size (int, optional): The size of each square patch for the purpose of mask generation. Defaults to 8.
        max_mask_ratio (float, optional): The maximum ratio of the cutout that can be masked. If None, masking is disabled.
        bands (list of str, optional): The list of colour bands to include in the dataset. Defaults to ['G','R','I','Z','Y'].
        img_size (int, optional): The size of the square cutouts to generate from the FITS tiles, in pixels. Defaults to 64.
        cutouts_per_tile (int, optional): The number of cutouts to generate from each FITS tile. Defaults to 1024.
        batch_size (int, optional): The number of cutouts per batch. Defaults to 64.
        transform (callable, optional): A function/transform that takes in a PyTorch tensor and returns a transformed version.
        pixel_min (float, optional): The minimum pixel value for clipping. Defaults to -3.
        pixel_max (float, optional): The maximum pixel value for clipping. If None, no upper clipping is applied.

    Attributes:
        band_filenames (list of lists): A nested list where each sublist contains the paths to the FITS files of the requested bands for a single tile.
        mask_generator (MaskGenerator or None): An instance of MaskGenerator used to create masks for the cutouts if max_mask_ratio is specified; otherwise None.

    Methods:
        __len__(): Returns the total number of FITS tiles with all requested bands available.
        __getitem__(idx): Returns a batch of cutouts and their corresponding masks (if mask generation is enabled) from the FITS tile at the specified index.

    Note:
        The actual loading of FITS files and generation of cutouts involve significant preprocessing, including 
        handling of NaN values, pixel value clipping, optional data augmentation, and dynamic mask generation. The 
        dataset is designed to work with batches of cutouts, facilitating efficient loading and processing of 
        large-scale astronomical data for deep learning models.
    """

    def __init__(self, fits_paths, patch_size=8, max_mask_ratio=None, bands=['G','R','I','Z','Y'], min_bands=5,
                 img_size=64, cutouts_per_tile=1024, batch_size=64, ra_dec=False,
                 transform=None, pixel_min=-3., pixel_max=None, use_calexp=True, use_overlap=False, overlap=0.5):
        
        self.fits_paths = fits_paths
        self.img_size = img_size
        self.cutouts_per_tile = cutouts_per_tile
        self.batch_size = batch_size
        self.ra_dec = ra_dec
        self.transform = transform
        self.pixel_min = pixel_min
        self.pixel_max = pixel_max
        self.use_calexp = use_calexp
        self.use_overlap = use_overlap
        self.overlap = overlap

        # Find names of patch fits files
        self.band_filenames = find_HSC_bands(fits_paths, bands, min_bands, use_calexp=use_calexp)

        if max_mask_ratio is not None:
            num_channels = len(bands)
            self.mask_generator = MaskGenerator(input_size=img_size,
                                                patch_size=patch_size,
                                                max_mask_ratio=max_mask_ratio,
                                                num_mask_chans=num_channels)
        else:
            self.mask_generator = None
                        
    def __len__(self):
        # The number of fits patches with all of the requested bands
        return len(self.band_filenames)
    
    def __getitem__(self, idx):
        # Grab fits filenames
        patch_filenames = self.band_filenames[idx]
        
        # Load all channels of the patch of sky
        # Any missing channels will be filled with np.nan
        cutouts, pix_to_radec = load_fits_bands(patch_filenames, return_wc=self.ra_dec)

        if self.use_overlap:
            if self.ra_dec:
                cutouts, ra_dec = overlapping_cutouts(cutouts, self.img_size, self.overlap, pix_to_radec)
                ra_dec = torch.from_numpy(ra_dec.astype(np.float32))
            else:
                cutouts = overlapping_cutouts(cutouts, self.img_size, self.overlap, pix_to_radec)
        else:
            if self.ra_dec:
                cutouts, ra_dec = random_cutouts(cutouts, self.img_size, self.cutouts_per_tile, pix_to_radec)
                ra_dec = torch.from_numpy(ra_dec.astype(np.float32))
            else:
                cutouts = random_cutouts(cutouts, self.img_size, self.cutouts_per_tile, pix_to_radec)

        # Clip pixel values
        if self.pixel_min is not None:
            cutouts[cutouts<self.pixel_min] = self.pixel_min
        if self.pixel_max is not None:
            cutouts[cutouts>self.pixel_max] = self.pixel_max

        # Apply any augmentations
        cutouts = torch.from_numpy(cutouts).to(torch.float32)
        if self.transform is not None:
            cutouts = self.transform(cutouts)

        if self.mask_generator is not None:
            # Generate random mask
            masks = torch.stack([self.mask_generator() for i in range(len(cutouts))])
        
        # Sort into M batches of batch_size
        M = cutouts.shape[0] // self.batch_size
        C = cutouts.shape[1]
        cutouts = cutouts[:M * self.batch_size].reshape((M, self.batch_size, C, self.img_size, self.img_size))
        if self.mask_generator is not None:
            masks = masks[:M * self.batch_size].reshape((M, self.batch_size, C, self.img_size, self.img_size))
        else:
            masks = torch.zeros((M, self.batch_size))

        if self.ra_dec:
            ra_dec = ra_dec[:M * self.batch_size].reshape((M, self.batch_size, 2))
            return cutouts, masks, ra_dec
        else:
            return cutouts, masks

def extract_center(array, n):
    """
    Extracts the central nxn chunk from a numpy array.

    :param array: Input 3D numpy array (C, H, W)
    :param n: Size of the square chunk to extract.
    :return: nxn central chunk of the array.
    """
    # Dimensions of the input array
    rows, cols = array.shape[1:]

    # Calculate the starting indices
    start_row = rows // 2 - n // 2
    start_col = cols // 2 - n // 2

    # Extract and return the nxn center
    return array[:,start_row:start_row + n, start_col:start_col + n]