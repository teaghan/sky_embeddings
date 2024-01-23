import numpy as np
import torch
import h5py

def build_dataloader(filename, norm_type, batch_size, num_workers, 
                     label_keys=None, img_size=64, shuffle=True):
    
    # Data loaders
    dataset = CutoutDataset(filename, img_size, label_keys, norm=norm_type)

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                       shuffle=shuffle, num_workers=num_workers,
                                       pin_memory=True)

class CutoutDataset(torch.utils.data.Dataset):
    
    """
    Dataset loader for the cutout datasets.
    """

    def __init__(self, data_file, img_size, label_keys=None, norm=None, transform=None, global_mean=0.1, global_std=2.):
        
        self.data_file = data_file
        self.transform = transform
        self.norm = norm
        self.img_size = img_size
        self.label_keys = label_keys
        self.global_mean = global_mean
        self.global_std = global_std
                        
    def __len__(self):
        with h5py.File(self.data_file, "r") as f:    
            num_samples = len(f['cutouts'])
        return num_samples
    
    def __getitem__(self, idx):
        
        with h5py.File(self.data_file, "r") as f: 
            # Load cutout
            cutout = f['cutouts'][idx].transpose(1,2,0)
            cutout[np.isnan(cutout)] = 0.

            if (np.array(cutout.shape[:2])>self.img_size).any():
                # Select central cutout
                cutout = extract_center(cutout, self.img_size)

            if self.label_keys is None:
                # Load RA and Dec
                labels = torch.from_numpy(np.asarray([f['ra'][idx], f['dec'][idx]]).astype(np.float32))
            else:
                labels = [f[k][idx] for k in self.label_keys]
                labels = torch.from_numpy(np.asarray(labels).astype(np.float32))

        if self.transform:
            cutout = self.transform(cutout)
        else:
            cutout = torch.from_numpy(cutout)
            cutout = cutout.permute(2,0,1)
        
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