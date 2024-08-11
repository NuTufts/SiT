import os,sys

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Example of defining a data loader class
class SimpleDetDataset(Dataset):
    NUM_COLUMNS = 5
    """
    When the data files are made, they contain the following types of data:
    
    hf.create_dataset(f'class_{i}', data=entry_data.class_index )
    hf.create_dataset(f'evis_{i}',  data=entry_data.Evis )
    hf.create_dataset(f'nparticles_{i}', data=entry_data.truthNparticles )
    hf.create_dataset(f'img_{i}',   data=entry_data.img, compression='gzip', compression_opts=9)
    hf.create_dataset(f'pdg_{i}',   data=entry_data.pdg, compression='gzip', compression_opts=9)
    
    """    
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.dataset_lengths = []
        self.cumulative_lengths = [0]
        # we have to scan the files to map out which file has which indices
        for file_path in file_paths:
            with h5py.File(file_path, 'r') as hf:
                nkeys = len(hf.keys())
                length = nkeys // SimpleDetDataset.NUM_COLUMNS  # Divide by number of columns in each entry
                print("length=",nkeys," for ",file_path)
                self.dataset_lengths.append(length)
                self.cumulative_lengths.append(self.cumulative_lengths[-1] + length)
        
    def __len__(self):
        return self.cumulative_lengths[-1]
    
    def __getitem__(self, idx):
        file_idx = np.searchsorted(self.cumulative_lengths, idx, side='right') - 1
        local_idx = idx - self.cumulative_lengths[file_idx]
        
        with h5py.File(self.file_paths[file_idx], 'r') as hf:
            img = np.array(hf[f'img_{local_idx}'])
            class_idx = np.array(hf[f'class_{local_idx}'])
        
        # Convert to torch tensor and normalize if needed
        img_tensor = torch.from_numpy(img).float()
        
        return img_tensor, torch.tensor(class_idx, dtype=torch.long)

# Usage example
def get_simpledet_dataloader(file_paths, batch_size=2, num_workers=1):
    
    finput_v = []
    if os.path.isdir( file_paths ):
        flist = os.listdir( file_paths )
        for f in flist:
            finput_v.append( file_paths + "/" + f.strip() )
    elif os.path.isfile( file_paths ):
        # single file
        finput_v = [file_paths]
    elif file_paths is list:
        for f in flist:
            if os.path.isfile( f ):
                finput_v.append(f)
            else:
                raise ValueError("One of the filepaths in the input list is not a valid file path.")

    dataset = SimpleDetDataset(finput_v)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

