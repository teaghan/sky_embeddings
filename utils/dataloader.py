import numpy as np
import torch
import h5py

def build_dataloader(filename, norm_type, batch_size, num_workers):
    
    # Data loaders
    dataset = CutoutDataset(filename, norm=norm_type)

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                       shuffle=True, num_workers=num_workers,
                                       pin_memory=True)

class CutoutDataset(torch.utils.data.Dataset):
    
    """
    Dataset loader for the cutout datasets.
    """

    def __init__(self, data_file, norm=None, transform=None):
        
        self.data_file = data_file
        self.transform = transform
        self.norm = norm
                        
    def __len__(self):
        with h5py.File(self.data_file, "r") as f:    
            num_samples = len(f['cutouts'])
        return num_samples
    
    def __getitem__(self, idx):
        
        with h5py.File(self.data_file, "r") as f: 
            # Load cutout
            cutout = f['cutouts'][idx].transpose(1,2,0)
            labels = torch.from_numpy(np.asarray([f['ra'][idx], f['dec'][idx]]).astype(np.float32))

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

        return cutout, labels