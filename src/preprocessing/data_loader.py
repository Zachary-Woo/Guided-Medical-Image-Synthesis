"""
Data loading utilities for medical image datasets.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np


class MedicalImageDataset(Dataset):
    """
    Dataset class for loading medical images and their corresponding masks.
    
    Attributes:
        data_dir (str): Directory containing the dataset
        image_paths (list): List of paths to images
        mask_paths (list): List of paths to masks (if available)
        transform (callable, optional): Transform to apply to images
        mask_transform (callable, optional): Transform to apply to masks
    """
    
    def __init__(self, data_dir, image_folder="images", mask_folder=None, 
                 transform=None, mask_transform=None):
        """
        Initialize the dataset.
        
        Args:
            data_dir (str): Directory containing the dataset
            image_folder (str): Folder name containing images
            mask_folder (str, optional): Folder name containing masks
            transform (callable, optional): Transform to apply to images
            mask_transform (callable, optional): Transform to apply to masks
        """
        self.data_dir = data_dir
        self.transform = transform
        self.mask_transform = mask_transform
        
        # Get image paths
        image_dir = os.path.join(data_dir, image_folder)
        self.image_paths = [os.path.join(image_dir, fname) 
                          for fname in sorted(os.listdir(image_dir))
                          if fname.endswith(('.png', '.jpg', '.jpeg', '.tif'))]
        
        # Get mask paths if available
        self.mask_paths = None
        if mask_folder:
            mask_dir = os.path.join(data_dir, mask_folder)
            if os.path.exists(mask_dir):
                self.mask_paths = [os.path.join(mask_dir, fname) 
                                for fname in sorted(os.listdir(mask_dir))
                                if fname.endswith(('.png', '.jpg', '.jpeg', '.tif'))]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Load mask if available
        mask = None
        if self.mask_paths:
            mask = Image.open(self.mask_paths[idx]).convert('L')
            
            if self.mask_transform:
                mask = self.mask_transform(mask)
        
        if mask is not None:
            return image, mask
        else:
            return image


def get_dataloader(dataset, batch_size=32, shuffle=True, num_workers=4):
    """
    Create a DataLoader from a Dataset.
    
    Args:
        dataset (Dataset): Dataset to load from
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle the data
        num_workers (int): Number of workers for loading
        
    Returns:
        DataLoader: DataLoader for the dataset
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    ) 