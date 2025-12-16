import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class BaseSSLDataset(Dataset, ABC):
    """
    Base class for SSL pretraining datasets.
    
    Subclasses must implement:
        - _load_image(): Load image from disk
        - _get_metadata(): Extract metadata for a sample
    """
    
    def __init__(
        self,
        csv_path: str,
        img_folder: str,
        transform: Optional[Callable] = None,
        split: Optional[str] = None,
    ):
        """
        Args:
            csv_path: Path to metadata CSV file
            img_folder: Root folder containing images
            transform: Transform to apply to images
            split: Dataset split ('train', 'val', 'test')
        """
        self.img_folder = img_folder
        self.transform = transform
        self.split = split
        
                                  
        self.meta_data = pd.read_csv(csv_path)
        if split is not None and 'split' in self.meta_data.columns:
            self.meta_data = self.meta_data[
                self.meta_data['split'] == split
            ].reset_index(drop=True)
    
    def __len__(self) -> int:
        return len(self.meta_data)
    
    @abstractmethod
    def _load_image(self, index: int) -> torch.Tensor:
        """
        Load image for the given index.
        
        Args:
            index: Sample index
            
        Returns:
            Image tensor of shape [C, H, W]
        """
        pass
    
    @abstractmethod
    def _get_metadata(self, index: int) -> Dict[str, Any]:
        """
        Get metadata for the given index.
        
        Args:
            index: Sample index
            
        Returns:
            Dictionary with 'smiles', 'dose', and any other metadata
        """
        pass
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get a sample.
        
        Returns:
            Dictionary containing:
                - 'views': Transformed image views (if transform is multi-view)
                           or single transformed image
                - 'smiles': SMILES string
                - 'dose': Dose/concentration value
                - Additional dataset-specific metadata
        """
        image = self._load_image(index)
        metadata = self._get_metadata(index)
        
        if self.transform is not None:
            image = self.transform(image)
        if metadata is not None:
            return {
                'views': image,
                **metadata,
            }
        else:
            return {
                'views': image,
            }
        
class WSLDataset(BaseSSLDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        treatments = sorted(self.meta_data['treatment'].str.split('_').str[0].unique())
        self.gene2idx = {g: i for i, g in enumerate(treatments)}
        self.num_classes = len(treatments)
        
        
    def _load_image(self, index: int) -> torch.Tensor:
        rel_path = self.meta_data.iloc[index]['well_id']
        path_parts = rel_path.split('_')
        
        data_path = os.path.join(
            self.img_folder,
            path_parts[0],
            'Plate' + path_parts[1],
            path_parts[-1] + '_s1.npy'
        )
        
        img = np.load(data_path)
        
        return img
    
    def _get_metadata(self, index: int) -> Dict[str, Any]:
        row = self.meta_data.iloc[index]
        treatment = row['treatment'].split('_')[0]
        label_idx = self.gene2idx[treatment]
        return {
            'label': label_idx,

        }
         


class RxRx3Dataset(BaseSSLDataset):
    """
    RxRx3 dataset.
    
    Image shape: [6, 512, 512] (6-channel microscopy)
    """
    
    def _load_image(self, index: int) -> torch.Tensor:
        rel_path = self.meta_data.iloc[index]['well_id']
        path_parts = rel_path.split('_')
        
        data_path = os.path.join(
            self.img_folder,
            path_parts[0],
            'Plate' + path_parts[1],
            path_parts[-1] + '_s1.npy'
        )
        
        img = np.load(data_path)
        
        return img
    
    def _get_metadata(self, index: int) -> Dict[str, Any]:
        pass