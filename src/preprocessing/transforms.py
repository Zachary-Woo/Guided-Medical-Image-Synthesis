"""
Transformations for medical images and masks.
"""

import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image


class MedicalImageTransforms:
    """
    Standard transformations for medical images.
    """
    
    @staticmethod
    def get_image_transforms(img_size=512, normalize=True):
        """
        Get transformations for medical images.
        
        Args:
            img_size (int): Size to resize images to
            normalize (bool): Whether to normalize images
            
        Returns:
            transforms: Composition of transforms
        """
        transform_list = [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ]
        
        if normalize:
            # Standard normalization for medical images
            transform_list.append(T.Normalize([0.5], [0.5]))
            
        return T.Compose(transform_list)
    
    @staticmethod
    def get_mask_transforms(img_size=512, binary=True):
        """
        Get transformations for segmentation masks.
        
        Args:
            img_size (int): Size to resize masks to
            binary (bool): Whether to binarize masks
            
        Returns:
            transforms: Composition of transforms
        """
        transform_list = [
            T.Resize((img_size, img_size), interpolation=Image.NEAREST),
            T.ToTensor(),
        ]
        
        if binary:
            # Convert mask to binary (0 or 1)
            transform_list.append(lambda x: (x > 0.5).float())
            
        return T.Compose(transform_list)
    
    @staticmethod
    def mask_to_edge(mask, thickness=1):
        """
        Convert binary mask to edge map for ControlNet conditioning.
        
        Args:
            mask (torch.Tensor): Binary mask (B, 1, H, W)
            thickness (int): Edge thickness
            
        Returns:
            torch.Tensor: Edge map
        """
        if not isinstance(mask, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
            
        # Convert to numpy for edge detection
        mask_np = mask.squeeze(1).cpu().numpy().astype(np.uint8)
        edges = np.zeros_like(mask_np)
        
        for i in range(mask_np.shape[0]):
            # Simple edge detection using erosion
            from scipy.ndimage import binary_erosion
            eroded = binary_erosion(mask_np[i], iterations=thickness)
            edges[i] = mask_np[i] - eroded
            
        # Convert back to torch tensor
        edge_tensor = torch.from_numpy(edges).unsqueeze(1).float()
        
        return edge_tensor
    
    @staticmethod
    def augment_training_data(img_size=512):
        """
        Get transforms for data augmentation during training.
        
        Args:
            img_size (int): Size to resize images to
            
        Returns:
            transforms: Composition of transforms
        """
        transform_list = [
            T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ]
        
        return T.Compose(transform_list) 