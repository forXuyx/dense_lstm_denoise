import os
import numpy as np
import torch
from base import BaseDataLoader
from torch.utils.data import Dataset

class GWDataset(Dataset):
    def __init__(self, data_dir):
        # get the data list
        self.root = data_dir
        self.clean_list = os.listdir(data_dir + 'clean/')

    def __len__(self):
        return len(self.clean_list)
    
    def __getitem__(self, index):
        clean_path = self.root + 'clean/' + self.clean_list[index]
        noisy_path = clean_path.replace('clean', 'noisy')
        
        clean = np.load(clean_path)
        noisy = np.load(noisy_path)

        # normalize the data to [-1, 1]
        clean[clean>0] = clean[clean>0] / np.max(noisy)
        clean[clean<0] = clean[clean<0] / np.abs(np.min(noisy))
        noisy[noisy>0] = noisy[noisy>0] / np.max(noisy)
        noisy[noisy<0] = noisy[noisy<0] / np.abs(np.min(noisy))

        # to tensor
        clean = torch.from_numpy(clean).float()
        noisy = torch.from_numpy(noisy).float()

        return noisy, clean

class GWDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.2, num_workers=8, training=True):
        self.data_dir = data_dir
        self.dataset = GWDataset(self.data_dir)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
