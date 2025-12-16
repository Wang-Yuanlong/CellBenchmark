import os
import sys
from typing import List

import numpy as np

import torch
from torch.utils.data import Dataset
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


class _PathDataset(Dataset):
    def __init__(self, paths: List[str], abs_path, transform=None):
        self.paths = paths
        self.abs_path = abs_path
        self.transform = transform

    def __len__(self):
        return len(self.paths)
    
    def _load_image(self, rel_path: str) -> torch.Tensor:
        path_parts = rel_path.split('_')
        data_path = os.path.join(
            self.abs_path,
            path_parts[0],
            'Plate' + path_parts[1],
            path_parts[-1] + '_s1.npy'
        )
        
        img = np.load(data_path) 
        
        img = torch.from_numpy(img).float()
        img = img.permute(2, 0, 1)
        return img
        

    def __getitem__(self, i):
        x = self._load_image(self.paths[i])       
        if self.transform is not None:
            x = self.transform(x)
        return x, i
    