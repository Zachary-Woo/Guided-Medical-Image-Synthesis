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
            # Standard normalization for many datasets
            transform_list.append(T.Normalize([0.5] * 3, [0.5] * 3)) # Assume 3 channels for general use
            
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
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
        ]
        
        if binary:
            # Convert mask to binary (0 or 1)
            transform_list.append(lambda x: (x > 0.5).float())
            
        return T.Compose(transform_list)

# Utility function added previously but perhaps better placed here?
def tensor_to_pil(tensor):
    """
    Convert a tensor to a PIL image.

    Args:
        tensor (torch.Tensor): Image tensor (C, H, W) in range [-1, 1] or [0, 1]

    Returns:
        PIL.Image: PIL image
    """
    # Scale to [0, 1] if in [-1, 1]
    if tensor.min() < 0:
        tensor = (tensor + 1) / 2

    # Clamp to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)

    # Convert to numpy
    if tensor.is_cuda:
        tensor = tensor.cpu()

    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)

    img_np = tensor.permute(1, 2, 0).numpy()

    # Convert single-channel grayscale to RGB if needed for saving/display
    if img_np.shape[2] == 1:
         img_np = np.tile(img_np, (1, 1, 3))

    # Convert to uint8
    img_np = (img_np * 255).astype(np.uint8)

    # Convert to PIL
    pil_img = Image.fromarray(img_np)

    return pil_img 