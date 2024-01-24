import numpy as np
import torch
import h5py

def build_dataloader(filename, norm_type, batch_size, num_workers, label_keys=None, 
                     img_size=64, pos_channel=False, num_patches=None, shuffle=True):
    
    # Data loaders
    dataset = CutoutDataset(filename, img_size=img_size, pos_channel=pos_channel, 
                            num_patches=num_patches, label_keys=label_keys, norm=norm_type)

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                       shuffle=shuffle, num_workers=num_workers,
                                       pin_memory=True)

class CutoutDataset(torch.utils.data.Dataset):
    
    """
    Dataset loader for the cutout datasets.
    """

    def __init__(self, data_file, img_size, pos_channel=False, num_patches=None, label_keys=None, 
                 norm=None, transform=None, global_mean=0.1, global_std=2., pixel_min=-3, pixel_max=50):
        
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
                        
    def __len__(self):
        with h5py.File(self.data_file, "r") as f:    
            num_samples = len(f['cutouts'])
        return num_samples
    
    def __getitem__(self, idx):
        
        with h5py.File(self.data_file, "r") as f: 
            # Load cutout
            cutout = f['cutouts'][idx].transpose(1,2,0)

            dummy = np.copy(cutout)
            
            cutout[np.isnan(cutout)] = 0.
            cutout[cutout<self.pixel_min] = self.pixel_min
            cutout[cutout>self.pixel_max] = self.pixel_max

            if (np.array(cutout.shape[:2])>self.img_size).any():
                # Select central cutout
                cutout = extract_center(cutout, self.img_size)

            if self.label_keys is None:
                # Load RA and Dec
                labels = torch.from_numpy(np.asarray([f['ra'][idx], f['dec'][idx]]).astype(np.float32))
            else:
                labels = [f[k][idx] for k in self.label_keys]
                labels = torch.from_numpy(np.asarray(labels).astype(np.float32))

            if self.pos_channel:
                # RA and Dec for positional channel
                central_ra = torch.tensor([f['ra'][idx]])
                central_dec = torch.tensor([f['dec'][idx]])
                # Determine the resolution at these spots
                ra_res, dec_res = hsc_dud_res(central_ra, central_dec)

        isnan = np.where(np.isnan(cutout))[0]
        if len(isnan)>0:
            print('A', idx, isnan)
            
        if self.transform:
            cutout = self.transform(cutout)
        else:
            cutout = torch.from_numpy(cutout)
            cutout = cutout.permute(2,0,1)

        if self.pos_channel:
            pos_channel = celestial_image_channel(central_ra, central_dec, ra_res, dec_res, self.img_size, self.num_patches,
                                                  ra_range=[0, 360], dec_range=[-90, 90])
            cutout = torch.cat((cutout, pos_channel), dim=0)

        isnan = np.where(np.isnan(cutout))[0]
        if len(isnan)>0:
            print('B', idx, isnan)
            
        if self.norm=='minmax':
            # Normalize sample between 0 and 1
            sample_min = torch.min(cutout)
            sample_max = torch.max(cutout)
            cutout = (cutout - sample_min) / (sample_max - sample_min)
        elif self.norm=='zscore':
            # Normalize to have zero mean and unit variance
            sample_mean = torch.min(cutout)
            sample_std = torch.std(cutout)
            cutout = (cutout - sample_mean) / sample_std
        elif self.norm=='global':
            cutout = (cutout - self.global_mean) / self.global_std

        isnan = np.where(np.isnan(cutout))[0]
        if len(isnan)>0:
            print('C', idx, dummy[isnan])

        return cutout, labels

def extract_center(array, n):
    """
    Extracts the central nxn chunk from a 2D numpy array.

    :param array: Input 2D numpy array.
    :param n: Size of the square chunk to extract.
    :return: nxn central chunk of the array.
    """
    # Dimensions of the input array
    rows, cols = array.shape[:2]

    # Calculate the starting indices
    start_row = rows // 2 - n // 2
    start_col = cols // 2 - n // 2

    # Extract and return the nxn center
    return array[start_row:start_row + n, start_col:start_col + n]

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