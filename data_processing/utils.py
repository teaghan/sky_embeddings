import os
import functools
import re
import glob
import random
import h5py
import sys
import uuid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
import astropy.wcs
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy.visualization import ZScaleInterval
from astropy.table import Table

import logging as log
root = log.getLogger()
root.setLevel(log.DEBUG)

handler = log.StreamHandler(sys.stdout)
handler.setLevel(log.DEBUG)
formatter = log.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)


def load_fits_bands(patch_filenames, return_header=False):
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
                    header = hdul[1].header  
            except Exception as e:
                # Handle the case where the FITS file cannot be opened
                print(f"Error opening {fn}: {e}")
                imgs.append(None)

    # Now, ensure all placeholders are replaced with np.nan arrays of the correct shape
    for i, item in enumerate(imgs):
        if item is None:
            imgs[i] = np.full(reference_shape, np.nan)

    
    # Organize into (C, H, W) and convert to a single NumPy array
    if return_header:
        return np.stack(imgs), header
    else:
        return np.stack(imgs)

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

#@functools.total_ordering
class Patch:
    def __init__(self, files, bands):
        self.files = files
        self.bands = bands
        # Iterable stuff
        self.current = 0
        self.high = len(self.files)
        self.matches = None
        self.data = None

    def load(self):
        # Load the HSC data into memory
        self.data = load_fits_bands(self.files)

    def load_header(self):
        for fn in self.files:
            if fn!='None':
                try:
                    # Attempt to open the FITS header
                    with fits.open(fn, mode='readonly', ignore_missing_simple=True) as hdul:
                        self.header = hdul[1].header  
                        break
                except Exception as e:
                    continue
    def close(self):
        if self.matches is not None:
            del self.matches
        if self.data is not None:
            del self.data

    # To handle closing fits files
    def __del__(self):
        try:
            self.close()
        except Exception as e:
            pass

    # Plot
    def plot(self):
        fig, ax = plt.subplots(1, 5)
        for i, (bandstr, hdulist) in enumerate(self.loaded_files.items()):
            interval = ZScaleInterval()
            ax[i].imshow(interval(hdulist[1].data))
        plt.show()

    # MatchCat
    def match_catalogue(self, labels_dataframe, size=64):
        # wcs from random band, extension 1
        self.matches = match_cat(self.header, df=labels_dataframe, size=size)

    # Generate uniform grid of ra, dec
    def generate_grid(self, grid_spacing):#nx, ny):
        # get wcs and bounds
        wcs = WCS(self.header)
        NAXIS1 = self.header['NAXIS1']
        NAXIS2 = self.header['NAXIS2']

        #x_spacings = np.linspace(0, NAXIS1, nx+2, dtype='i')[1:-1]
        #y_spacings = np.linspace(0, NAXIS2, ny+2, dtype='i')[1:-1]
        x_spacings = np.arange(grid_spacing/2 + 1 , NAXIS1 - grid_spacing/2 - 1,grid_spacing).astype(int)
        y_spacings = np.arange(grid_spacing/2 + 1 , NAXIS2 - grid_spacing/2 - 1,grid_spacing).astype(int)

        xs = []
        ys = []
        for x in x_spacings:
            for y in y_spacings:
                xs.append(x)
                ys.append(y)
                
        ras, decs = wcs.all_pix2world(xs, ys, 0)
        
        # Null zspec column for now
        zspecs = np.zeros(len(ras), dtype='i')
        zspec_err = np.zeros(len(ras), dtype='i')

        df = pd.DataFrame({'ra':ras, 'dec': decs, 'zspec': zspecs, 'zspec_err':zspec_err})

        # self.matches isn't quite an appropriate name anymore
        self.matches = df

    def write_vo_table(self):
        # So we can check sources in ds9.
        t = Table.from_pandas(self.matches)
        t.write(f'{str(self)}_matches.xml', table_id=f'{str(self)}', format='votable')

    def get_wcs(self):
        return WCS(self.header)

    def __len__(self):
        return len(self.files)

    # So we can iterate over patches
    def __iter__(self):
        return self

    def __next__(self):
        if self.current < self.high:
            result = self.files[self.current]
            self.current += 1
            return result
        raise StopIteration

    # So we can sort patches
    def __eq__(self, other):
        return str(self) == str(other)

    # So we can hash and use patches as dictionary keys
    def __hash__(self):
        return hash(self.__repr__())

def getPatches(**kwargs):
    data_dirs = kwargs['data_paths']
    bands = kwargs['bands']
    use_calexp = kwargs['use_calexp']

    filenames = find_HSC_bands(data_dirs, bands, min_bands=2, verbose=1, use_calexp=use_calexp)

    patches = []
    for patch_filenames in filenames:
        patches.append(Patch(patch_filenames, bands))

    number_to_return = kwargs['n_patches']
    if kwargs['patch_strategy'] == 'random':
        log.info(f"Choosing {number_to_return} random patches to use.")
        patches_to_return = random.sample(patches, number_to_return)
    elif kwargs['patch_strategy'] == 'first_n':
        log.info(f"Choosing first {number_to_return} patches to use.")
        patches_to_return = patches[:number_to_return]
    elif kwargs['patch_strategy'] =='n_to_n':
        low = kwargs['patch_low_bound']
        high = kwargs['patch_high_bound']
        log.info(f"Returning patches {low} to {high}.")
        patches_to_return = patches[low:high]
    elif kwargs['patch_strategy'] == 'all':
        patches_to_return = patches

    return patches_to_return

def match_cat(header, df=None, size=64):
    # Load the FITS hdulist using astropy.io.fits
    nx = header['NAXIS1']
    ny = header['NAXIS2']
    wcs = WCS(header)
    x,y = wcs.all_world2pix(df['ra'], df['dec'], 0)
    return df[(x<nx-size//2) & (y<ny-size//2) & (x>size//2) & (y>size//2)]

class Cutout:
    def __init__(self):
        pass

def get_cutouts_from_patch(patch, bands, size=64):
    # Array to hold cutouts for frame
    cutouts = []

    w = patch.get_wcs()
    
    # Fill missing labels with NaN
    if 'zspec' in patch.matches.columns:
        zspec_vals = patch.matches.zspec
    else:
        zspec_vals = np.full((len(patch.matches.ra),), np.nan)
    if 'zspec_err' in patch.matches.columns:
        zspec_err_vals = patch.matches.zspec_err
    else:
        zspec_err_vals = np.full((len(patch.matches.ra),), np.nan)
        
    # For each coordinate match falling within the frame
    for ra, dec, zspec, zspec_err, j in zip(patch.matches.ra, patch.matches.dec, 
                                            zspec_vals, zspec_err_vals, range(len(patch.matches))):
        c = SkyCoord(ra*u.deg, dec*u.deg)

        cutout = Cutout()
        cutout.data = np.zeros((len(bands), size, size))
        # Loop through bands
        for i, (band, data) in enumerate(zip(bands, patch.data)):#loaded_files.items():

            # Select cutout based on RA and Dec
            x, y = astropy.wcs.utils.skycoord_to_pixel(c, w)
            c_data = np.copy(Cutout2D(data,(x,y), (size, size)).data)
            if c_data.shape != (size, size):
                c_data = np.copy(Cutout2D(data,(int(x),int(y)), (size, size)).data)
            # Save to correct channel
            cutout.data[i] = c_data

        cutout.meta = {'ra': ra, 'dec': dec, 'zspec': zspec, 'zspec_err':zspec_err}
        cutouts.append(cutout)
        
    return cutouts

def cutouts_to_hdf5(hdf5_dir, cutout_list, bands):

    if not os.path.exists(hdf5_dir):
        os.mkdir(hdf5_dir)

    # random filename
    filename = str(uuid.uuid1())+'.h5'

    f = h5py.File(os.path.join(hdf5_dir, filename), "w")

    num_bands = len(bands)#cutout_list[0].data.keys())
    size = cutout_list[0].data.shape[1]  # -> 64

    c_dset = f.create_dataset("cutouts", (len(cutout_list),num_bands,size,size), dtype='f')
    ra_dset = f.create_dataset("ra", (len(cutout_list),), dtype='f')
    dec_dset = f.create_dataset("dec", (len(cutout_list),), dtype='f')
    zspec_dset = f.create_dataset("zspec", (len(cutout_list),), dtype='f')
    zspecerr_dset = f.create_dataset("zspec_err", (len(cutout_list),), dtype='f')

    for i, c in enumerate(cutout_list):
        # Add data
        c_dset[i] = c.data

        # Add meta
        ra_dset[i] = c.meta['ra']
        dec_dset[i] = c.meta['dec']
        zspec_dset[i] = c.meta['zspec']
        zspecerr_dset[i] = c.meta['zspec_err']

    f.close()