import os
import math
import functools
import re
import glob
import pickle
import random
import h5py
from abc import ABC, abstractmethod
import re
import uuid
import sys

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


@functools.total_ordering
class Patch:
    def __init__(self, files):
        self.files = sorted(files)
        # Iterable stuff
        self.current = 0
        self.high = len(self.files)
        # with fits.open(self.files[0]) as hdul:
        #     self.wcs = WCS(hdul[1].header)
        self.matches = None
        


    def load(self):
        # Load the HSC data into memory
        self.loaded_files = {}
        for f in self.files:
            band = re.search(r'calexp-HSC-(.)-', f).group()[11]
            self.loaded_files[band] = fits.open(f, mode='readonly')

    def close(self):
        for band, hdulist in self.loaded_files.items():
            hdulist.close()
        if self.matches is not None:
            del self.matches

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
        keys = list(self.loaded_files.keys())
        header = self.loaded_files[keys[0]][1].header  
        self.matches = match_cat(header, df=labels_dataframe, size=size)

    # Generate uniform grid of ra, dec
    def generate_grid(self, grid_spacing):#nx, ny):
        # get wcs and bounds
        keys = list(self.loaded_files.keys())
        header = self.loaded_files[keys[0]][1].header
        wcs = WCS(header)
        NAXIS1 = header['NAXIS1']
        NAXIS2 = header['NAXIS2']

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
        keys = list(self.loaded_files.keys())
        header = self.loaded_files[keys[0]][1].header
        return WCS(header)

    def __str__(self):
        """
        /arc/projects/ots/pdr3_dud/calexp-HSC-G-9707-4%2C0.fits'
        /arc/projects/ots/pdr3_dud/calexp-HSC-I-9707-4%2C0.fits'
        /arc/projects/ots/pdr3_dud/calexp-HSC-R-9707-4%2C0.fits'
                      --> G,I,R-9707-4,0
        """
        bands = [f.split('/')[-1].split('-')[2] for f in self.files]
        re_string = fr'(?<=-[{"".join(bands)}]-).*(?=.fits)'
        patch_id = re.search(fr'(?<=-[{"".join(bands)}]-).*(?=.fits)', self.files[0]).group()

        # If loaded data available, append a rough WCS estimation
        try:
            keys = list(self.loaded_files.keys())
            header = self.loaded_files[keys[0]][1].header
            wcs = WCS(header)
            sky = wcs.pixel_to_world(0, 0)
            ra = f"{sky.ra.deg:.2f}"
            dec = f"{sky.dec.deg:.2f}"
        except:
            ra = ''
            dec = ''
        return ','.join(bands).rstrip(',') + '-' + patch_id.replace('%2C', ',') + f" {ra} {dec}"

    def __repr__(self):
        return f"{str(self)}"

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

    def __lt__(self, other):
        return str(self) < str(other)

    # So we can hash and use patches as dictionary keys
    def __hash__(self):
        return hash(self.__repr__())


def getPatches(**kwargs):
    """
    Controls the sample of patches (stacks of 3/5-band aligned fits files) used.

    Control number of bands with kwargs['bands'], data path with 
    kwargs['data_path'].

    Can also return less than all patches found for testing purposes;
    kwargs['n_patches'], kwargs['patch_strategy'] = 'random' | 'first_n' | 'all'.

    Saves names of all patches used to kwargs['cache_dir'] + '/used_patches.txt'

    kwargs['bands']: ['G', 'I', 'R', ...]

    Use for example with 
    
    ```
    patches = dataloaders.getPatches(**{'data_path':'/arc/projects/ots/pdr3_dud', 'bands':['G','I','R'], 
            'patch_strategy':'all', 'n_patches':-1, 'cache_dir':'/scratch'})
    ```
    
    Or with
    
    ```
    patches = getPatches(**{'data_path':'/arc/projects/ots/pdr3_dud', 'bands':['G','I','R'], 'n_patches':30, 'patch_strategy':'first_n', 'cache_dir':'/scratch'})
    ```
    
    """
    data_dir = kwargs['data_path']
    bands = kwargs['bands']
    fits_files = sorted(glob.glob(f"{data_dir}/calexp-HSC-*.fits"))
    # Basically take '/arc/projects/ots/pdr3_dud/calexp-HSC-I-9707-4%2C0.fits'
    # and put it to 9707-4%2C0.fits
    unique_patches = list(set(['-'.join(x.split('-')[-2:]) for x in fits_files]))
    unique_patches = sorted(unique_patches)
    
    # Sets are hashable, see explanation below
    set_fits_files = set(fits_files)

    patches = []
    for t in unique_patches:
        potential_files = [f'{data_dir}/calexp-HSC-{b}-{t}' for b in bands]

        # `f in set_fits_files` is O(n) if fits_files is a list,
        # ~O(1) if fits_files is a hash table
        if (all([f in set_fits_files for f in potential_files])):
            patches.append(Patch(potential_files))

    log.info(f"Found {len(patches)} {len(bands)}-band patches.")

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
    
    patch_save_file = os.path.join(kwargs['cache_dir'], "used_patches.txt")
    with open(patch_save_file, 'w') as f:
        for p in sorted(patches_to_return):
            f.write(f'{str(p)}\n')
    log.info(f"Full list of patches used can be found in {patch_save_file}")
    return patches_to_return

class Buffer:
    def __init__(self, empty_func=print, max_length=10000):
        self.storage = []
        self.max_length = max_length
        self.empty_func = empty_func

        self.count_written = 0

    def add(self, item):
        if len(self.storage) == self.max_length:
            self.flush()
        self.storage.append(item)
        self.count_written += 1

    def flush(self):
        #log.info(f"Flushing {len(self.storage)} items from buffer.")
        self.empty_func(self.storage)
        self.storage = []

    def __del__(self):
        # Handle any remaining objects
        self.flush()



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

def get_cutouts_from_patch(patch, size=64, collect_var=False):
    # Array to hold cutouts for frame
    cutouts = []

    w = patch.get_wcs()
    
    # With self-supervision we might not have labels
    if 'zspec' in patch.matches.columns:
        # For each coordinate match falling within the frame
        for ra, dec, zspec, zspec_err, j in zip(patch.matches.ra, patch.matches.dec, patch.matches.zspec, patch.matches.zspec_err, range(len(patch.matches))):
            c = SkyCoord(ra*u.deg, dec*u.deg)

            cutout = Cutout()
            cutout.data = {}
            cutout.var_data = {}                
            for band, hdulist in patch.loaded_files.items():
                x, y = astropy.wcs.utils.skycoord_to_pixel(c, w)

                data = hdulist[1].data
                c_data = np.copy(Cutout2D(data,(x,y), (size, size)).data)
                if c_data.shape != (size, size):
                    c_data = np.copy(Cutout2D(data,(int(x),int(y)), (size, size)).data)
                cutout.data[band] = c_data

                if collect_var:
                    var_data = hdulist[3].data
                    v_data = np.copy(Cutout2D(var_data,(x,y), (size, size)).data)
                    if v_data.shape != (size, size):
                        v_data = np.copy(Cutout2D(var_data,(int(x),int(y)), (size, size)).data)
                    cutout.var_data[band] = v_data

            cutout.meta = {'ra': ra, 'dec': dec, 'zspec': zspec, 'zspec_err':zspec_err,}
            cutouts.append(cutout)
    # else:
    #     for ra, dec, j in zip(matches.ra, matches.dec, range(len(matches))):
    #         c = SkyCoord(ra*u.deg, dec*u.deg)
    #         x, y = astropy.wcs.utils.skycoord_to_pixel(c, w)
    #         # print(ra,dec,x,y)
    #         img_cut[j] = Cutout2D(data,(x,y), (size, size)).data

    #         met_cut.append({'ra': ra, 'dec': dec})
        
    return cutouts


def get_cutouts_from_file(labels, fits_file, size=64):
    # Get image data
    with fits.open(fits_file) as hdul:
        header = hdul[1].header
        data = hdul[1].data
        
        # First find redshifts within frame
        matches = match_cat(header, df=labels)
        
        # Array to hold cutouts for frame
        img_cut = np.zeros((len(matches), size, size))
        
        # List of cutout metadata dictionaries
        met_cut = []
        
        # Now get cutouts for each of those ra, dec pairs
        w = WCS(hdul[1].header)
        
        # With self-supervision we might not have labels
        if 'zspec' in matches.columns:
            for ra, dec, zspec, zspec_err, j in zip(matches.ra, matches.dec, matches.zspec, matches.zspec_err, range(len(matches))):
                c = SkyCoord(ra*u.deg, dec*u.deg)
                x, y = astropy.wcs.utils.skycoord_to_pixel(c, w)
                # print(ra,dec,x,y)
                img_cut[j] = Cutout2D(data,(x,y), (size, size)).data

                met_cut.append({'ra': ra, 'dec': dec, 'zspec': zspec, 'zspec_err':zspec_err, 'first_parent_file':fits_file})
        else:
            for ra, dec, j in zip(matches.ra, matches.dec, range(len(matches))):
                c = SkyCoord(ra*u.deg, dec*u.deg)
                x, y = astropy.wcs.utils.skycoord_to_pixel(c, w)
                # print(ra,dec,x,y)
                img_cut[j] = Cutout2D(data,(x,y), (size, size)).data

                met_cut.append({'ra': ra, 'dec': dec, 'first_parent_file':fits_file})
            
        return img_cut, met_cut
    
    
def multiband_cutout_getter(labels, fits_files, size=64):
    """
    >>> labels = "/arc/projects/ots/dark3d/labels/hsc_ssp.parquet"
    >>> fits_files = \
            ["/arc/projects/ots/pdr3_dud/calexp-HSC-G-10054-0%2C0.fits",
             "/arc/projects/ots/pdr3_dud/calexp-HSC-I-10054-0%2C0.fits",
             "/arc/projects/ots/pdr3_dud/calexp-HSC-R-10054-0%2C0.fits"]
    >>> cutouts, meta = multiband_cutout_getter(labels, fits_files)
    >>> print(cutouts[0].shape, meta[0])
    
    (3, 64, 64) {'ra': 149.55274, 'dec': 2.9686445, 'zspec': 0.34407418966293335, 'first_parent_file': '/arc/projects/ots/pdr3_dud/calexp-HSC-G-10054-0%2C0.fits'}
    
    """
    # Get cutouts from all files of patch
    cutout_data = []
    cutout_meta = []
    for i, f in enumerate(fits_files):
        data, meta = get_cutouts_from_file(labels, f)
        cutout_data.append(data)
        # Only return metadata from first cutout of each band
        if i == 0:
            cutout_meta.extend(meta)
    
    # Go from eg. (3,17,32,32) to (17,3,32,32)
    mb_cutouts = np.array(cutout_data).swapaxes(0,1)
    
    return mb_cutouts, cutout_meta


def singleband_cutout_getter(labels, fits_file, size=64):
    """
    >>> labels = "/arc/projects/ots/dark3d/labels/hsc_ssp.parquet"
    >>> fits_file = "/arc/projects/ots/pdr3_dud/calexp-HSC-G-10054-0%2C0.fits"
    >>> cutouts, meta = singleband_cutout_getter(labels, fits_file)
    >>> print(cutouts[0].shape, meta[0])
    
    (1, 64, 64) {'ra': 149.55274, 'dec': 2.9686445, 'zspec': 0.34407418966293335, 'first_parent_file': '/arc/projects/ots/pdr3_dud/calexp-HSC-G-10054-0%2C0.fits'}
    
    """
    return multiband_cutout_getter(labels, [fits_file], size=size)


def filenameToPath(filename, rootPath='/scratch/test'):
    # To store millions of cutouts, need many sub-directories
    first_level = 64   # n directories in root path
    second_level = 64  # n directories in each of those
    third_level = 4      # ...
    
    n_bits_1 = 6      # First n bits of idx give address of first directory
    n_bits_2 = 6
    n_bits_3 = 2
    
    # Make directories
    if not os.path.exists(rootPath):
        for i in range(first_level):
            for j in range(second_level):
                for k in range(third_level):
                    os.makedirs(os.path.join(rootPath, str(i), str(j), str(k)))
    
    # cutouts have a prefix of idx_...
    idx = int(filename.split('_')[0])
    
    if idx > first_level*second_level*third_level-1:
        raise ValueError("Storage Exceeded")
    
    # Encode directories as binary bits
    # Eg.
    #    0000 | 0000 | 00
    # 
    # First four bits of i encode one of 16 directories, next four bits encode
    # which of 16 sub-directories, last two bits encode which of four sub-sub-directories. 
    
    first_dir = idx >> (n_bits_2 + n_bits_3)
    second_dir = (idx >> n_bits_3) & ((2**(n_bits_2))-1) # Right shift to eliminate third level, bit mask to select second
    third_dir = idx & ((2**(n_bits_3))-1)
    
    return os.path.join(rootPath, str(first_dir), str(second_dir), str(third_dir), filename)

def prebake_cutouts_to_disk(labels, patches, output_path='/scratch/cutouts', size=64, multiband=True, n_cutouts=100):
    # Labels is provided as a dataframe
    
    cutouts = []
    
    # Go through all patches, and return once we have found enough cutouts
    i = 0
    for patch in patches:
        if multiband:
            cutouts, metas = multiband_cutout_getter(labels, patch.files, size=size)
        else:
            cutouts, metas = singleband_cutout_getter(labels, patch.files[0], size=size)
        # cutouts is an array like (17, 3, 64, 64)
        #log.info(f"{os.path.split(patch.files[0])[1]}, {cutouts.shape}")
        for cutout, meta in zip(cutouts, metas):
            filename = filenameToPath(str(i)+'_.pkl', rootPath=output_path)
            with open(filename, 'wb') as f:
                pickle.dump((cutout, meta), f)
            i += 1
            if i % 100 == 0:
                log.info(f"Cutout {i}, {filename}...")
            if i == n_cutouts:
                return
            
    raise RuntimeError(f'Only found {i} cutouts in {len(patches)} fields but expected {n_cutouts}')


def get_cutout_from_disk(idx, output_path='/scratch/cutouts'):
    with open(filenameToPath(str(idx)+'_.pkl', rootPath=output_path), 'rb') as f:
        cutout, meta = pickle.load(f)
    return cutout, meta


######## Other helper functions   ############

def coordinateToCutout(ra, dec, size=64, bands=['G', 'R', 'I'],
        fits_file_directories=['/arc/projects/ots/pdr3_dud/calexp-HSC*.fits'],
        crval_cache_file='crval_catalogue.pkl',
        rough_frame_size=0.6*u.deg
    ):
    """
    Note: if a coordinate falls to multiple frames, may return multiple cutouts.
    """
    start = time.time()
    coordinate = SkyCoord(ra*u.deg, dec*u.deg)
    # Build catalogue of RA/DEC positions of all files
    if not os.path.exists(crval_cache_file):
        files = []
        for path in fits_file_directories:
            files.extend(glob.glob(path))

        files = np.array(sorted(files))

        ras = np.zeros(len(files), dtype='f')
        decs = np.zeros(len(files), dtype='f')

        print(f"Building ra/dec catalogue, iterating over fits headers...")
        for i, f in enumerate(files):
            with fits.open(f) as hdul:
                #ras[i] = float(hdul[1].header['CRVAL1'])
                #decs[i] = float(hdul[1].header['CRVAL2'])
                w = WCS(hdul[1].header)
                sky = w.pixel_to_world(2000, 2000)
                ras[i] = float(sky.ra.deg)
                decs[i] = float(sky.dec.deg)
            if i % 100 == 0:
                print(f"Finished {i}...")
        with open(crval_cache_file, 'wb') as f:
            pickle.dump((ras, decs, files), f)
        print(f"Done.")
    else:
        # print(f"Loaded crval1, crval2, and file list from {crval_cache_file}")
        #start = time.time()
        with open(crval_cache_file, 'rb') as f:
            ras, decs, files = pickle.load(f)
        #end = time.time()
        #print(f'{end-start} seconds')

    end = time.time()
    #print(f"{end-start} setup")
    start = time.time()
    # Now build the catalogue and find closest members
    catalog = SkyCoord(ra=ras*u.degree, dec=decs*u.degree)
    #print(coordinate.separation(catalog) < 0.01*u.deg)
    #print(catalog[coordinate.separation(catalog) < 0.01*u.deg])
    close_files = files[coordinate.separation(catalog) < 0.01*u.deg]
    #print(len(close_files))
    end = time.time()
    #print(f"{end-start} close files")

    # Do a final check to make sure in footprint
    start = time.time()
    output = []
    for f in close_files:
        with fits.open(f) as hdul:
            w = WCS(hdul[1].header)
            if w.footprint_contains(coordinate):
                output.append(f)
    end = time.time()
    #print(f"{end-start} final check")

    start = time.time()
    # Filter for just three bands
    out_1 = []
    for b in bands:
        out_1.extend([x for x in output if f'HSC-{b}-' in x])

    out_1.sort(key=lambda x: (x.split('-HSC-')[1][1:], x.split('-HSC-')[1][:1]))

    # Just return the first time it appears.... For now
    if len(out_1) == 0:
        pass
    elif len(out_1) == 3:
        pass
    elif len(out_1) > 3:
        out_1 = out_1[:3]
    
    # Iterate over groups of bands, then individual files
    cutouts = []
    for i in range(0, len(out_1), len(bands)):
        for j in range(len(bands)):
            fits_file = out_1[i+j]
            with fits.open(fits_file) as f:
                w = WCS(f[1].header)
                s = time.time()
                cutout = Cutout2D(f[1].data, coordinate,(64,64), wcs=w)
                e = time.time()
                #print(f"    {e-s} for cutout2d")
                cutouts.append(cutout)

    end = time.time()
    #print(f"{end-start} getting cutout")

    return cutouts


######### (Mostly) HDF5 Functions ################


def next_power_of_2(x):
    return 1 if x == 0 else 2**math.ceil(math.log2(x))


def i_to_indices(i, mode='modulo', **kwargs):
    """
    mode := 'modulo' | 'bits'
    """
    if mode == 'bits':
        result = _i_to_indices_bits(i, **kwargs)
    elif mode == 'modulo':
        result = _i_to_indices_modulo(i, **kwargs)
    else:
        raise NotImplementedError(str(mode))
    
    return result

def _i_to_indices_bits(i, **kwargs):
    """
    Eg. 'n_storage_files' = 4
    
    11000001 = 97
    __             # First two bits specify storage file 3
      ______       # Last six bits specify index 1

    kwargs = {
        'n_cutouts': 2**11,
        'n_storage_files': 2**1
    }
    """
    power_2_ceiling_n = next_power_of_2(kwargs['n_cutouts'])
    
    if i > kwargs['n_cutouts']:
        raise ValueError(f"i {i} > n_cutouts {kwargs['n_cutouts']}")
    
    total_bits = power_2_ceiling_n.bit_length()-1
    storage_file_bits = kwargs['n_storage_files'].bit_length()-1
    within_file_bits = total_bits - storage_file_bits
    
    storage_mask = ((2**storage_file_bits)-1) << within_file_bits
    within_file_mask = (2**within_file_bits)-1
    
    s_index = (i & storage_mask) >> within_file_bits
    w_index = i & within_file_mask
    
    return s_index, w_index


def _i_to_indices_modulo(i, **kwargs):
    """
    kwargs = {
        'cutouts_per_file': 10000
    }
    """
    return i//kwargs['cutouts_per_file'], i % kwargs['cutouts_per_file']


def cutouts_to_hdf5(hdf5_dir, cutout_list, collect_var=False):
    # cutout.meta = {'ra': 352.85712, 'dec': -0.5041202, 'zspec': 0.6624044179916382}
    # cutout.data = {'G': array(shape=(64,64)), 'R':...}

    if not os.path.exists(hdf5_dir):
        os.mkdir(hdf5_dir)

    # random filename
    filename = str(uuid.uuid1())+'.h5'

    f = h5py.File(os.path.join(hdf5_dir, filename), "w")

    num_bands = len(cutout_list[0].data.keys())
    size = list(cutout_list[0].data.values())[0].shape[0]  # -> 64

    c_dset = f.create_dataset("cutouts", (len(cutout_list),num_bands,size,size), dtype='f')
    ra_dset = f.create_dataset("ra", (len(cutout_list),), dtype='f')
    dec_dset = f.create_dataset("dec", (len(cutout_list),), dtype='f')
    zspec_dset = f.create_dataset("zspec", (len(cutout_list),), dtype='f')
    zspecerr_dset = f.create_dataset("zspec_err", (len(cutout_list),), dtype='f')
    if collect_var:
        v_dset = f.create_dataset("cutout variance", (len(cutout_list),num_bands,size,size), dtype='f')

    for i, c in enumerate(cutout_list):
        # Add data
        data = c.data # {'G': array, 'R': array...}
        sorted_keys = sorted(list(data.keys()))#list(data.keys())#
        print(sorted_keys)
        for j, key in enumerate(sorted_keys):
            c_dset[i,j,:,:] = data[key]
            
        if collect_var:
            data = c.var_data
            for j, key in enumerate(sorted_keys):
                v_dset[i,j,:,:] = data[key]

        # Add meta
        ra_dset[i] = c.meta['ra']
        dec_dset[i] = c.meta['dec']
        zspec_dset[i] = c.meta['zspec']
        zspecerr_dset[i] = c.meta['zspec_err']

    f.close()



def initialize_hdf5(filename, **kwargs):
    """
    kwargs = {
        'n_cutouts': 2**11,
        'bands': ['G', 'I', 'R'],
        'cutout_size': 64,
        'cutouts_per_file': 10000
    }
    """
    f = h5py.File(filename, "w")

    c_dset = f.create_dataset("cutouts", (kwargs['cutouts_per_file'],len(kwargs['bands']),kwargs['cutout_size'],kwargs['cutout_size']), dtype='f')
    ra_dset = f.create_dataset("ra", (kwargs['cutouts_per_file'],), dtype='f')
    dec_dset = f.create_dataset("dec", (kwargs['cutouts_per_file'],), dtype='f')
    zspec_dset = f.create_dataset("zspec", (kwargs['cutouts_per_file'],), dtype='f')

    f.close()


from typing import Dict, Tuple, Sequence
class MultiHDF5Store(ABC):
    """
    Given a directory (possibly empty) of h5 files,
    knows how to read and write cutouts to/from it.


    m = MultiH5Interface()
    cutout, meta = m.read(i)
    m.write(cutout, meta, i)


    """
    def __init__(self, **kwargs):
        """
        - Should work if H5 files already present
        - Should create files if not present
        """
        self.kwargs = kwargs

        # Initialize on-disk files
        if not os.path.exists(kwargs['h5_directory']):
            os.mkdir(kwargs['h5_directory'])
        if not self._h5_files_exist():
            self._make_h5_files()
        self._get_h5_list()

    @abstractmethod
    def read(self, i) -> Tuple[np.array, dict]:
        raise NotImplementedError

    @abstractmethod
    def write(self, i, cutout: np.array, meta:dict):
        raise NotImplementedError

    @abstractmethod
    def _make_h5_files(self):
        raise NotImplementedError

    @abstractmethod
    def _get_h5_list(self):
        raise NotImplementedError

    def _h5_files_exist(self):
        # Super simple check
        return glob.glob(os.path.join(self.kwargs['h5_directory'], "*.h5")) or \
                glob.glob(os.path.join(self.kwargs['h5_directory'], "*.hdf5"))
            


class NatMultiHDF5(MultiHDF5Store):
    dsets = ['cutouts', 'ra', 'dec', 'zspec']
    cutout_dset = 'cutouts'
    ra_dset = 'ra'
    dec_dset = 'dec'
    zspec_dset = 'zspec'
    """
    kwargs = {
        'n_cutouts': 57000
        'bands': ['G','I','R'],
        'cutout_size': 64,
        'cutouts_per_file': 10000,
        'h5_directory': '/scratch/hdf5'
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        if False: # Silence linter warnings
            self.kwargs = {}

    def _make_h5_files(self):
        n_files = math.ceil(self.kwargs['n_cutouts'] / self.kwargs['cutouts_per_file'])

        for i in range(n_files):
            h5_path = os.path.join(self.kwargs['h5_directory'], f"{i}.h5")
            initialize_hdf5(h5_path, **self.kwargs)

    def _get_h5_list(self):
        self.h5_files = sorted(
                glob.glob(os.path.join(self.kwargs['h5_directory'], '*.h5')),
                key=lambda x: int(os.path.split(x)[1].split('.')[0]))

    def read(self, i):
        h5_file_index, cutout_index = i_to_indices(i, mode='modulo', **self.kwargs)

        f = h5py.File(self.h5_files[h5_file_index], "r")

        cutout = f[NatMultiHDF5.cutout_dset][cutout_index,:,:,:]
        meta = {
                'ra':f[NatMultiHDF5.ra_dset][cutout_index],
                'dec':f[NatMultiHDF5.dec_dset][cutout_index],
                'zspec':f[NatMultiHDF5.zspec_dset][cutout_index]
        }
        f.close()

        return cutout, meta

    def write(self, i, cutout, meta):
        """
        cutout: np.array, shape=(32,5,64,64)
        """
        h5_file_index, cutout_index = i_to_indices(i, mode='modulo', **self.kwargs)

        f = h5py.File(self.h5_files[h5_file_index], "a")
        f[NatMultiHDF5.cutout_dset][cutout_index,:,:,:] = cutout
        f[NatMultiHDF5.ra_dset][cutout_index] = meta['ra']
        f[NatMultiHDF5.dec_dset][cutout_index] = meta['dec']
        f[NatMultiHDF5.zspec_dset][cutout_index] = meta['zspec']
        f.close()


class SpencerMultiHDF5(MultiHDF5Store):
    dsets = ['cfis_id', 'dec', 'images', 'ra', 'tile']
    cutout_dset = 'images'
    ra_dset = 'ra'
    dec_dset = 'dec'
    zspec_dset = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        if False: # Silence linter warnings
            self.kwargs = {}

    def _make_h5_files(self):
        raise NotImplementedError

    def _get_h5_list(self):
        """Spencer's stacks have format of
                'cutout.stacks.ugriz.lsb.200x200.316.10000.h5'
        """
        self.h5_files = sorted(
                glob.glob(os.path.join(self.kwargs['h5_directory'], '*.h5')),
                key=lambda x: int(os.path.split(x)[1].split('.')[5]))

    def read(self, i):
        h5_file_index, cutout_index = i_to_indices(i, mode='modulo', **self.kwargs)

        f = h5py.File(self.h5_files[h5_file_index], "r")

        cutout = f[SpencerMultiHDF5.cutout_dset][cutout_index,:,:,:]
        meta = {
                'ra':f[SpencerMultiHDF5.ra_dset][cutout_index],
                'dec':f[SpencerMultiHDF5.dec_dset][cutout_index],
                'zspec': None
        }
        f.close()

        return cutout, meta

    def write(self):
        raise NotImplementedError


class CutoutIterator():
    def __init__(self, patches, labels, **kwargs):
        self.kwargs = kwargs
        self.labels = labels
        self.patches = patches
        self.cutouts = []
        self.metas = []
        
    def __iter__(self):
        return self
        
    def __next__(self):
        # quit when we run out of cutouts
        if (not self.patches) and (not self.cutouts):
            raise StopIteration
        
        # Get new batch of cutouts
        if not self.cutouts:
            patch = self.patches.pop()
            if len(self.kwargs['bands']) > 1:
                self.cutouts, self.metas = multiband_cutout_getter(self.labels, patch.files, size=self.kwargs['cutout_size'])
            else:
                self.cutouts, self.metas = singleband_cutout_getter(self.labels, patch.files[0], size=self.kwargs['cutout_size'])
            # hack
            self.cutouts = list(self.cutouts)
        
        return self.cutouts.pop(), self.metas.pop()


class DummyCutoutIterator():
    """
    Just yields randomized numpy arrays, hopefully faster for testing.
    kwargs = {
        'n_cutouts': 2**11,
        'bands': ['G', 'I', 'R'],
        'cutout_size': 64,
        'n_storage_files': 2*2
    }
    """
    def __init__(self, patches, labels, **kwargs):
        self.kwargs = kwargs
        self.labels = labels
        self.patches = patches
        self.i = 0
        
    def __iter__(self):
        return self
        
    def __next__(self):
        if self.i == self.kwargs['n_cutouts']:
            raise StopIteration
        else:
            self.i += 1

        data = np.random.uniform(size=(len(self.kwargs['bands']),self.kwargs['cutout_size'],self.kwargs['cutout_size']))
        meta = { 'ra': -1.0, 'dec': -1.0, 'zspec': -1.0, 'first_parent_file': 'null.fits'}
        
        return data, meta



def grid_ras_within_frames(fits_files, n_per_frame=100):
    """
    patches = getPatches(**{'data_path':'/arc/projects/ots/pdr3_dud', 'bands':['G','I','R'], 'n_patches':10, 'patch_strategy':'first_n', 'cache_dir':'/scratch'})

    # Get a grid of ras/decs within the patches field of view
    grid_df = grid_ras_within_frames([p.files[0] for p in patches], n_per_frame=9)
    print(grid_df)
    """
    
    # Get evenly spaced ras within frame
    ras = []
    decs = []
    all_xs = []
    all_ys = []
    
    for f in fits_files:
        with fits.open(f) as hdul:
            header = hdul[1].header
            wcs = WCS(header)
            nx = header['NAXIS1']
            ny = header['NAXIS2']
            
            x_spacings = np.linspace(0, nx, int(n_per_frame**0.5)+2, dtype='i')[1:-1]
            y_spacings = np.linspace(0, ny, int(n_per_frame**0.5)+2, dtype='i')[1:-1]
            
            xs = []
            ys = []
            for x in x_spacings:
                for y in y_spacings:
                    xs.append(x)
                    ys.append(y)
                    
            r, d = wcs.all_pix2world(xs, ys, 0)
            ras.extend(r)
            decs.extend(d)
            all_xs.extend(xs)
            all_ys.extend(ys)
            
    
    df = pd.DataFrame({'ra':ras, 'dec': decs, 'xs': all_xs, 'ys': all_ys})
    
    return df

