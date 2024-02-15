import numpy as np
import torch
import h5py
from torchvision.transforms import v2
import glob
from astropy.io import fits
import torchvision
torchvision.disable_beta_transforms_warning()

def build_fits_dataloader(fits_paths, bands, norm_type, batch_size, num_workers,
                          patch_size=8, max_mask_ratio=None, 
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
                          patch_size=patch_size, max_mask_ratio=max_mask_ratio,
                          cutouts_per_tile=cutouts_per_tile,
                          batch_size=batch_size, shuffle=shuffle,
                          norm=norm_type, transform=transforms,
                          global_mean=pix_mean, global_std=pix_std)

    # Build dataloader
    return torch.utils.data.DataLoader(dataset, batch_size=1, 
                                       shuffle=shuffle, num_workers=num_workers,
                                       pin_memory=True)

def build_dataloader(filename, norm_type, batch_size, num_workers, 
                     patch_size=8, num_channels=5, 
                     max_mask_ratio=None, label_keys=None, 
                     img_size=64, pos_channel=False, pix_mean=None, pix_std=None, num_patches=None, 
                     augment=False, shuffle=True, indices=None):

    if augment:
        transforms = v2.Compose([v2.GaussianBlur(kernel_size=5, sigma=(0.1,1.5)),
                                 v2.RandomResizedCrop(size=(img_size, img_size), scale=(0.8,1), antialias=True),
                                 v2.RandomHorizontalFlip(p=0.5),
                                 v2.RandomVerticalFlip(p=0.5)])
    else:
        transforms = None
    
    # Data loaders
    dataset = CutoutDataset(filename, img_size=img_size, 
                            patch_size=patch_size, num_channels=num_channels, max_mask_ratio=max_mask_ratio,
                            pos_channel=pos_channel, 
                            num_patches=num_patches, label_keys=label_keys, norm=norm_type,
                            transform=transforms, global_mean=pix_mean, global_std=pix_std,
                           indices=indices)

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                       shuffle=shuffle, num_workers=num_workers,
                                       pin_memory=True)

class MaskGenerator:
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

class CutoutDataset(torch.utils.data.Dataset):
    
    """
    Dataset loader for the cutout datasets.
    """

    def __init__(self, data_file, img_size, patch_size, num_channels, max_mask_ratio, 
                 pos_channel=False, num_patches=None, label_keys=None, 
                 norm=None, transform=None, global_mean=0.1, global_std=2., pixel_min=-3., pixel_max=None,
                indices=None):
        
        self.data_file = data_file
        self.transform = transform
        self.norm = norm
        self.img_size = img_size
        self.pos_channel = pos_channel
        self.num_patches = num_patches
        self.label_keys = label_keys
        self.global_mean = global_mean
        self.global_std = global_std 
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
            # Custom set of indices
            idx = self.indices[idx]
        with h5py.File(self.data_file, "r") as f: 
            # Load cutout
            cutout = f['cutouts'][idx]

            # Remove any NaN value
            cutout[np.isnan(cutout)] = 0.

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

        cutout = torch.from_numpy(cutout)
        # Apply any augmentations, etc.
        if self.transform is not None:
            cutout = self.transform(cutout)

        # Add position as additional channel
        if self.pos_channel:
            pos_channel = celestial_image_channel(central_ra, central_dec, ra_res, dec_res, self.img_size, self.num_patches,
                                                  ra_range=[0, 360], dec_range=[-90, 90])
            cutout = torch.cat((cutout, pos_channel), dim=0)
            
        if self.norm=='minmax':
            # Normalize sample between 0 and 1
            sample_min = torch.min(cutout)
            sample_max = torch.max(cutout)
            cutout = (cutout - sample_min) / (sample_max - sample_min)
        elif self.norm=='zscore':
            # Normalize sample to have zero mean and unit variance
            sample_mean = torch.mean(cutout)
            sample_std = torch.std(cutout)
            cutout = (cutout - sample_mean) / (sample_std +1e-6)
        elif self.norm=='global':
            # Normalize dataset to have zero mean and unit variance
            cutout = (cutout - self.global_mean) / self.global_std

        if self.mask_generator is not None:
            # Generate random mask
            mask = self.mask_generator()
        else:
            mask = 0

        return cutout, mask, labels

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

    def __init__(self, fits_paths,  patch_size=8, max_mask_ratio=None, bands=['G','R','I','Z','Y'], img_size=64, cutouts_per_tile=1024,
                 batch_size=64, shuffle=True, norm=None, transform=None, 
                 global_mean=0.1, global_std=2., pixel_min=-3., pixel_max=None):
        
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
