import os
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import time

from utils import getPatches

fits_path = '/arc/projects/ots/pdr3_dud'
bands = ['G','R','I','Z','Y']

# Find fits files and get ready for loading
kwargs = {'data_path':fits_path,
          'bands':bands,
          'patch_strategy':'all',
          'n_patches':-1, 
          'patch_low_bound': 0,
          'patch_high_bound': -1,
          'cache_dir':'/scratch'}

ra_res = []
dec_res = []
ra_min = []
ra_max = []
dec_min = []
dec_max = []

for i, p in enumerate(getPatches(**kwargs)):
    with fits.open(p.files[0]) as hdul:
        # Load fits header
        wcs = WCS(hdul[1].header)

        # Find RAs of pixels
        xs = np.linspace(0, hdul[1].header['NAXIS2'], hdul[1].header['NAXIS2']+2, dtype='i')[1:-1]
        ys = np.zeros_like(xs)
        ras, _ = wcs.all_pix2world(xs, ys, 0)

        # Find Decs of pixels
        ys = np.linspace(0, hdul[1].header['NAXIS1'], hdul[1].header['NAXIS1']+2, dtype='i')[1:-1]
        xs = np.zeros_like(ys)
        _, decs = wcs.all_pix2world(xs, ys, 0)
        
        ra_res.append(np.mean(np.diff(ras)))
        dec_res.append(np.mean(np.diff(decs)))
        ra_min.append(np.min(ras))
        ra_max.append(np.max(ras))
        dec_min.append(np.min(decs))
        dec_max.append(np.max(decs))

    if (i+1)%500==0:
        print(f'{i+1} files complete.')

print(f'RA ranges from {np.min(ra_min)} to {np.max(ra_max)}.')
print(f'Dec ranges from {np.min(dec_min)} to {np.max(dec_max)}.')

# For RA resolution
# Fit a cubic polynomial (degree 3) to the data
coefficients = np.polyfit(dec_min, ra_res, 3)
print(f'RA resolution as a function of Dec coeffs: {np.round(coefficients,12)}')

# For Dec resolution
print(f'Dec resolutions found: {np.unique(np.round(dec_res,6))}')