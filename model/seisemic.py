import os
import numpy as np
import torch
from torch.utils.data import Dataset


class NpyDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir):
        self.clean_files = [os.path.join(clean_dir, f) for f in os.listdir(clean_dir) if f.endswith('.npy')]
        self.noisy_files = [os.path.join(noisy_dir, f) for f in os.listdir(noisy_dir) if f.endswith('.npy')]
        assert len(self.clean_files) == len(self.noisy_files), "Clean and noisy files must have the same length"

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        clean_data = np.load(self.clean_files[idx])
        noisy_data = np.load(self.noisy_files[idx])

        # Assuming the data is 1D, if it's 2D or 3D, you might need to adjust the following lines
        clean_data = torch.from_numpy(clean_data).float().unsqueeze(0)  # Add channel dimension
        noisy_data = torch.from_numpy(noisy_data).float().unsqueeze(0)  # Add channel dimension

        return {'source': noisy_data, 'target': clean_data}