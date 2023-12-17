import numpy as np
import torch
import h5py

class CutoutDataset(torch.utils.data.Dataset):
    
    """
    Dataset loader for the cutout datasets.
    """

    def __init__(self, data_file, transform=None):
        
        self.data_file = data_file
        self.transform = transform
                        
    def __len__(self):
        with h5py.File(self.data_file, "r") as f:    
            num_samples = len(f['cutouts'])
        return num_samples
    
    def __getitem__(self, idx):
        
        with h5py.File(self.data_file, "r") as f: 
            # Load cutout
            cutout = f['cutouts'][idx].transpose(1,2,0)
            labels = torch.from_numpy(np.asarray([f['ra'][idx], f['dec'][idx]]).astype(np.float32))
            #cutout = torch.from_numpy(f['cutouts'][idx])

        if self.transform:
            cutout = self.transform(cutout)
            
        #cutout = cutout.permute(2,0,1)

        return cutout, labels