"""
Data loading utilities for medical image datasets.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import logging


class MedicalImageDataset(Dataset):
    """
    Dataset class for loading medical images and optionally their corresponding masks.
    Can load images from a specified subfolder or the root data directory.

    Attributes:
        data_dir (str): Root directory containing the dataset.
        image_folder (str): Subfolder containing images relative to data_dir.
                            Use "." to load images directly from data_dir.
        mask_folder (str, optional): Subfolder containing masks relative to data_dir.
                                    If None, only images are loaded.
        image_paths (list): List of full paths to images.
        mask_paths (list, optional): List of full paths to masks (if mask_folder provided).
        transform (callable, optional): Transform to apply to images.
        mask_transform (callable, optional): Transform to apply to masks.
        has_masks (bool): Flag indicating if masks were successfully found and loaded.
    """

    def __init__(self, data_dir, image_folder=".", mask_folder=None,
                 transform=None, mask_transform=None):
        """
        Initialize the dataset.

        Args:
            data_dir (str): Root directory containing the dataset.
            image_folder (str): Subfolder containing images. Use "." for root.
            mask_folder (str, optional): Subfolder containing masks. If None, only images loaded.
            transform (callable, optional): Transform to apply to images.
            mask_transform (callable, optional): Transform to apply to masks.
        """
        self.data_dir = data_dir
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.transform = transform
        self.mask_transform = mask_transform
        self.has_masks = False

        # Determine image directory
        if image_folder == ".":
            image_dir = data_dir
        else:
            image_dir = os.path.join(data_dir, image_folder)

        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        # Get image paths
        self.image_paths = sorted([
            os.path.join(image_dir, fname)
            for fname in os.listdir(image_dir)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))
        ])

        if not self.image_paths:
            logging.warning(f"No images found in directory: {image_dir}")

        # Get mask paths if mask_folder is provided
        self.mask_paths = []
        if mask_folder:
            if mask_folder == ".":
                 mask_dir = data_dir
            else:
                 mask_dir = os.path.join(data_dir, mask_folder)

            if os.path.isdir(mask_dir):
                mask_candidates = sorted([
                    os.path.join(mask_dir, fname)
                    for fname in os.listdir(mask_dir)
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))
                ])
                # assume mask names correspond to image names
                img_basenames = {os.path.splitext(os.path.basename(p))[0] for p in self.image_paths}
                for mask_path in mask_candidates:
                    mask_basename = os.path.splitext(os.path.basename(mask_path))[0]
                    # Check if mask basename (without ext) matches any image basename
                    if mask_basename in img_basenames:
                        self.mask_paths.append(mask_path)

                if len(self.mask_paths) == len(self.image_paths):
                    self.has_masks = True
                    logging.info(f"Found matching masks for all {len(self.image_paths)} images in {mask_dir}")
                elif self.mask_paths:
                    logging.warning(f"Found {len(self.mask_paths)} masks in {mask_dir}, but expected {len(self.image_paths)}. Check naming.")
                    self.has_masks = False # Force False if counts don't match
                    self.mask_paths = [] # Clear partial masks to avoid errors in __getitem__
                else:
                    logging.warning(f"Mask folder '{mask_dir}' provided but no matching masks found.")
            else:
                logging.warning(f"Mask directory not found: {mask_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if idx >= len(self.image_paths):
            raise IndexError("Index out of range")

        # Load image
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
        except Exception as e:
             logging.error(f"Error loading image {self.image_paths[idx]}: {e}")
             # Return dummy data or raise error?
             return torch.zeros(3, 64, 64), torch.zeros(1, 64, 64) if self.has_masks else torch.zeros(3, 64, 64)

        if self.transform:
            image = self.transform(image)

        # Load mask if available and expected
        if self.has_masks:
            if idx < len(self.mask_paths):
                 try:
                    mask = Image.open(self.mask_paths[idx]).convert('L')
                 except Exception as e:
                     logging.error(f"Error loading mask {self.mask_paths[idx]}: {e}")
                     mask_tensor = torch.zeros_like(image[0:1,:,:])
                     return image, mask_tensor

                 if self.mask_transform:
                     mask = self.mask_transform(mask)
                 return image, mask
            else:
                logging.error(f"Mask path missing for index {idx} despite has_masks=True.")
                mask_tensor = torch.zeros_like(image[0:1,:,:])
                return image, mask_tensor
        else:
            # Return image and a dummy label if no masks are expected/found
            # This ensures consistent output format (tuple) for ConcatDataset
            dummy_label = torch.tensor(0, dtype=torch.long)
            return image, dummy_label


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