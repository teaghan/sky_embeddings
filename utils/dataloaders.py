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
            img (torch.Tensor): The input image tensor. Expected shape: (C, H, W) where
                C is the number of channels, H is the height, and W is the width.

        Returns:
            torch.Tensor: The augmented image tensor with a random number of channels replaced by NaN values, up to `max_channels`.
        """
        # Ensure the input is a tensor
        if not isinstance(img, torch.Tensor):
            raise TypeError("img must be a torch.Tensor")

        # Ensure max_channels is not greater than the number of channels in the image
        C, _, _ = img.shape
        if self.max_channels > C:
            raise ValueError(f"max_channels must be less than or equal to the number of channels in the image. Got {self.max_channels} for an image with {C} channels.")

        # Randomly decide the number of channels to replace, up to max_channels
        n_channels_to_replace = random.randint(0, self.max_channels)

        # Randomly select n_channels_to_replace to replace with NaN
        channels_to_replace = random.sample(range(C), n_channels_to_replace)

        for c in channels_to_replace:
            img[c, :, :] = torch.nan

        return img

# Define the augmentation pipeline
def get_augmentations(img_size=64, flip=True, crop=True, brightness=True, noise=True, nan_channels=True):
    transforms = []
    if flip:
        transforms.append(v2.RandomHorizontalFlip())
        transforms.append(v2.RandomVerticalFlip())
    if crop:
        transforms.append(v2.RandomResizedCrop(size=(img_size, img_size), 
                                               scale=(0.8, 1.0), 
                                               ratio=(0.9, 1.1), antialias=True))
    if brightness:
        transforms.append(RandomBrightnessAdjust(brightness_range=(0.2, 5)))
    if noise:
        transforms.append(RandomNoise(noise_range=(0., 0.1)))
    if nan_channels:
        transforms.append(RandomChannelNaN(max_channels=2))
        
    return v2.Compose(transforms)

def build_fits_dataloader(fits_paths, bands, min_bands, batch_size, num_workers,
                          patch_size=8, max_mask_ratio=None, 
                          img_size=64, cutouts_per_tile=1024, use_calexp=True,
                          augment=False, shuffle=True):
    '''Return a dataloader to be used during training.'''

    if augment:
        transforms = get_augmentations(img_size=img_size)
    else:
        transforms = None
    
    # Build dataset
    dataset = FitsDataset(fits_paths, patch_size=patch_size, 
                          max_mask_ratio=max_mask_ratio,
                          bands=bands, min_bands=min_bands, img_size=img_size, 
                          cutouts_per_tile=cutouts_per_tile,
                          batch_size=batch_size, shuffle=shuffle,
                          transform=transforms, use_calexp=use_calexp)

    # Build dataloader
    return torch.utils.data.DataLoader(dataset, batch_size=1, 
                                       shuffle=shuffle, num_workers=num_workers,
                                       pin_memory=True)

def build_h5_dataloader(filename, batch_size, num_workers, patch_size=8, num_channels=5, 
                        max_mask_ratio=None, label_keys=None, img_size=64, pos_channel=False, 
                        num_patches=None, augment=False, shuffle=True, indices=None):

    if augment:
        transforms = transforms = get_augmentations(img_size=img_size)
    else:
        transforms = None
    
    # Build dataset
    dataset = H5Dataset(filename, img_size=img_size, patch_size=patch_size, 
                        num_channels=num_channels, max_mask_ratio=max_mask_ratio,
                        pos_channel=pos_channel, num_patches=num_patches, 
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
        pos_channel (bool, optional): If True, adds a positional channel to the image based on RA and Dec.
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
                 pos_channel=False, num_patches=None, label_keys=None, 
                 transform=None, pixel_min=-3., pixel_max=None, indices=None):
        
        self.data_file = data_file
        self.transform = transform
        self.img_size = img_size
        self.pos_channel = pos_channel
        self.num_patches = num_patches
        self.label_keys = label_keys
        self.pixel_min = pixel_min
        self.pixel_max = pixel_max
        self.indices = indices

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

            # Remove any NaN value
            #cutout[np.isnan(cutout)] = 0.

            # Clip pixel values
            if self.pixel_min is not None:
                cutout[cutout<self.pixel_min] = self.pixel_min
            if self.pixel_max is not None:
                cutout[cutout>self.pixel_max] = self.pixel_max

            if (np.array(cutout.shape[1:])>self.img_size).any():
                # Select central cutout
                cutout = extract_center(cutout, self.img_size)

            if self.label_keys is None:
                # Load RA and Dec
                labels = torch.from_numpy(np.asarray([f['ra'][idx], f['dec'][idx]]).astype(np.float32))
            else:
                # or other labels
                labels = [f[k][idx] for k in self.label_keys]
                labels = torch.from_numpy(np.asarray(labels).astype(np.float32))

            if self.pos_channel:
                # RA and Dec for positional channel
                central_ra = torch.tensor([f['ra'][idx]])
                central_dec = torch.tensor([f['dec'][idx]])
                # Determine the resolution at these spots
                ra_res, dec_res = hsc_dud_res(central_ra, central_dec)

        cutout = torch.from_numpy(cutout).to(torch.float32)
        # Apply any augmentations, etc.
        if self.transform is not None:
            cutout = self.transform(cutout)

        # Add position as additional channel
        if self.pos_channel:
            pos_channel = celestial_image_channel(central_ra, central_dec, ra_res, dec_res, self.img_size, self.num_patches,
                                                  ra_range=[0, 360], dec_range=[-90, 90])
            cutout = torch.cat((cutout, pos_channel), dim=0)

        if self.mask_generator is not None:
            # Generate random mask
            mask = self.mask_generator()
        else:
            mask = torch.zeros_like(cutout)

        return cutout, mask, labels

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

def load_fits_bands(patch_filenames):
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
            except Exception as e:
                # Handle the case where the FITS file cannot be opened
                print(f"Error opening {fn}: {e}")
                imgs.append(None)

    # Now, ensure all placeholders are replaced with np.nan arrays of the correct shape
    for i, item in enumerate(imgs):
        if item is None:
            imgs[i] = np.full(reference_shape, np.nan)

    # Organize into (C, H, W) and convert to a single NumPy array
    return np.stack(imgs)

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
        shuffle (bool, optional): Whether to shuffle the cutouts before batching. Defaults to True.
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

    def __init__(self, fits_paths,  patch_size=8, max_mask_ratio=None, bands=['G','R','I','Z','Y'], min_bands=5,
                 img_size=64, cutouts_per_tile=1024, batch_size=64, shuffle=True, 
                 transform=None, pixel_min=-3., pixel_max=None, use_calexp=True):
        
        self.fits_paths = fits_paths
        self.img_size = img_size
        self.cutouts_per_tile = cutouts_per_tile
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transform = transform
        self.pixel_min = pixel_min
        self.pixel_max = pixel_max
        self.use_calexp = use_calexp

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
        cutouts = load_fits_bands(patch_filenames)

        # Split into a grid of cutouts based on img_size and overlap
        cutouts = random_cutouts(cutouts, self.img_size, self.cutouts_per_tile)

        # Shuffle images
        if self.shuffle:
            permutation = np.random.permutation(cutouts.shape[0])
            cutouts = cutouts[permutation]

        # Remove any NaN pixel values
        #cutouts[np.isnan(cutouts)] = 0.

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

def celestial_image_channel(central_ras, central_decs, ra_res, dec_res, img_size, num_patches,
                                  ra_range=[0, 360], dec_range=[-90, 90]):
    """
    Generate an additional image channel encoding RA and Dec information for each patch.

    Args:
    central_ras (torch.Tensor): Tensor of central Right Ascensions for the batch, shape: [batch_size].
    central_decs (torch.Tensor): Tensor of central Declinations for the batch, shape: [batch_size].
    ra_res (torch.Tensor): Tensor of RA resolutions for the batch, shape: [batch_size].
    dec_res (torch.Tensor): Tensor of Dec resolutions for the batch, shape: [batch_size].
    img_size (int): Size of the images in pixels (assumed to be square).
    num_patches (int): Total number of patches in each image.
    ra_range (list): The range of Right Ascension values, default is [0, 360].
    dec_range (list): The range of Declination values, default is [-90, 90].

    Returns:
    torch.Tensor: Tensor of shape [batch_size, img_size, img_size] containing the additional image channel.
    """

    batch_size = central_ras.size(0)
    # Normalize RA and Dec values to [0, 1] range for the whole batch
    norm_ras = (central_ras - ra_range[0]) / (ra_range[1] - ra_range[0])
    norm_decs = (central_decs - dec_range[0]) / (dec_range[1] - dec_range[0])

    # Calculate the size of each patch and the number of patches along one dimension
    patch_size = img_size // int(np.sqrt(num_patches))  # Assuming square patches
    num_patches_side = int(np.sqrt(num_patches))
    pixel_per_patch = patch_size ** 2

    # Calculate the offsets to center RA and Dec in the middle of each image
    center_offset_ra = (patch_size * ra_res) / 2
    center_offset_dec = (patch_size * dec_res) / 2

    # Create a linear space for trigonometric encoding, to be used for both RA and Dec
    sin_cos_range = torch.linspace(-np.pi, np.pi, pixel_per_patch // 4)

    # Compute the column and row indices for each patch
    # This indexing assumes the RA is constant for a given column in your image
    # if the RA is constant for the rows, use indexing='ij'
    cols, rows = torch.meshgrid(torch.arange(num_patches_side), torch.arange(num_patches_side), indexing='xy')
    cols = cols.flatten()
    rows = rows.flatten()

    # Initialize the channel tensor
    pos_channel_batch = torch.zeros((batch_size, img_size, img_size), device=central_ras.device)

    # Process each RA and Dec in the batch
    for i in range(batch_size):
        # Adjust for central RA/Dec and calculate offsets
        ra_offsets = ((cols - num_patches_side / 2) * ra_res[i] * patch_size + center_offset_ra[i]) / (ra_range[1] - ra_range[0])
        dec_offsets = ((rows - num_patches_side / 2) * dec_res[i] * patch_size + center_offset_dec[i]) / (dec_range[1] - dec_range[0])
        
        # Normalize offsets and wrap within [0, 1]
        patch_ras = (norm_ras[i] + ra_offsets) % 1
        patch_decs = (norm_decs[i] + dec_offsets) % 1

        # Convert to radians
        patch_ras = patch_ras * 2 * np.pi 
        patch_decs = patch_decs * 2 * np.pi 

        # Create trigonometric embeddings for RA and Dec
        ra_embeddings = torch.cat([torch.sin(sin_cos_range + patch_ras[:, None]),
                                   torch.cos(sin_cos_range + patch_ras[:, None])], dim=1)
        dec_embeddings = torch.cat([torch.sin(sin_cos_range + patch_decs[:, None]),
                                    torch.cos(sin_cos_range + patch_decs[:, None])], dim=1)
        

        # Concatenate the RA and Dec embeddings for each patch
        pos_channel = torch.cat([ra_embeddings, dec_embeddings], dim=1)

        # Reshape and permute to align the patches correctly
        pos_channel = pos_channel.view(num_patches_side, num_patches_side, patch_size, patch_size)
        pos_channel = pos_channel.permute(0, 2, 1, 3).contiguous()
        
        # Reshape to the final image size
        pos_channel = pos_channel.view(img_size, img_size)

        pos_channel_batch[i] = pos_channel

    return pos_channel_batch

def evaluate_polynomial(coefficients, x):
    """ Evaluate a polynomial with given coefficients at x (similar to np.poly1d).
    
    Args:
    coefficients (Tensor): A tensor of coefficients, [a_n, ..., a_1, a_0]
    x (Tensor): Points at which to evaluate, can be a scalar or a tensor
    
    Returns:
    Tensor: The evaluated polynomial
    """
    degree = len(coefficients) - 1
    y = torch.zeros_like(x)
    for i, coeff in enumerate(coefficients):
        y += coeff * x ** (degree - i)
    return y

def hsc_dud_res(ra, dec):
    '''Return the resolution of the images.'''
    # The Dec resolution is pretty constant in the frames that I looked at
    dec_res = 4.7e-05

    # The RA resolution was dependent on the Dec.
    # In the Dec range [-10,60], this cubic function returns a pretty good estimate
    coefficients = torch.tensor([-2.4e-10, 1.02e-09, 2.869e-08, -4.672178e-05])
    ra_res = evaluate_polynomial(coefficients, dec)
    return ra_res, torch.ones_like(ra_res)*dec_res
