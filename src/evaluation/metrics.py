"""
Metrics for evaluating generated medical images.
"""

import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
from torchmetrics.functional import dice_score, jaccard_index
from cleanfid import fid


def calculate_fid(real_images_path, generated_images_path, device="cuda"):
    """
    Calculate Fréchet Inception Distance (FID) between real and generated images.
    
    Args:
        real_images_path (str): Path to directory with real images
        generated_images_path (str): Path to directory with generated images
        device (str): Device to use for FID calculation
        
    Returns:
        float: FID score (lower is better)
    """
    score = fid.compute_fid(
        real_images_path,
        generated_images_path,
        device=device
    )
    return score


def calculate_structural_similarity(real_images, generated_images):
    """
    Calculate structural similarity index (SSIM) between real and generated images.
    
    Args:
        real_images (np.ndarray): Real images (N, H, W, C) in [0, 1]
        generated_images (np.ndarray): Generated images (N, H, W, C) in [0, 1]
        
    Returns:
        float: Mean SSIM score (higher is better)
    """
    scores = []
    for i in range(real_images.shape[0]):
        score = ssim(
            real_images[i], 
            generated_images[i],
            channel_axis=2,
            data_range=1.0
        )
        scores.append(score)
    
    return np.mean(scores)


def calculate_dice_coefficient(real_masks, generated_masks):
    """
    Calculate Dice coefficient between real and generated masks.
    
    Args:
        real_masks (torch.Tensor): Real masks (N, C, H, W)
        generated_masks (torch.Tensor): Generated masks (N, C, H, W)
        
    Returns:
        float: Mean Dice coefficient (higher is better)
    """
    # Ensure binary masks
    real_masks = (real_masks > 0.5).float()
    generated_masks = (generated_masks > 0.5).float()
    
    # Calculate Dice score
    dice = dice_score(generated_masks, real_masks)
    
    return dice.item()


def calculate_iou(real_masks, generated_masks):
    """
    Calculate Intersection over Union (IoU) between real and generated masks.
    
    Args:
        real_masks (torch.Tensor): Real masks (N, C, H, W)
        generated_masks (torch.Tensor): Generated masks (N, C, H, W)
        
    Returns:
        float: Mean IoU (higher is better)
    """
    # Ensure binary masks
    real_masks = (real_masks > 0.5).float()
    generated_masks = (generated_masks > 0.5).float()
    
    # Calculate IoU
    iou = jaccard_index(generated_masks, real_masks)
    
    return iou.item()


def calculate_mask_accuracy(real_masks, generated_masks):
    """
    Calculate pixel-wise accuracy between real and generated masks.
    
    Args:
        real_masks (torch.Tensor): Real masks (N, C, H, W)
        generated_masks (torch.Tensor): Generated masks (N, C, H, W)
        
    Returns:
        float: Mean pixel-wise accuracy (higher is better)
    """
    # Ensure binary masks
    real_masks = (real_masks > 0.5).float()
    generated_masks = (generated_masks > 0.5).float()
    
    # Calculate accuracy
    correct = (real_masks == generated_masks).float()
    accuracy = correct.sum() / correct.numel()
    
    return accuracy.item()


def evaluate_model_performance(real_images, generated_images, real_masks=None, generated_masks=None):
    """
    Evaluate model performance using multiple metrics.
    
    Args:
        real_images (torch.Tensor or np.ndarray): Real images
        generated_images (torch.Tensor or np.ndarray): Generated images
        real_masks (torch.Tensor, optional): Real masks
        generated_masks (torch.Tensor, optional): Generated masks
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    metrics = {}
    
    # Convert tensors to numpy if needed
    if isinstance(real_images, torch.Tensor):
        real_images_np = real_images.permute(0, 2, 3, 1).cpu().numpy()
        generated_images_np = generated_images.permute(0, 2, 3, 1).cpu().numpy()
    else:
        real_images_np = real_images
        generated_images_np = generated_images
    
    # Calculate SSIM
    metrics['ssim'] = calculate_structural_similarity(real_images_np, generated_images_np)
    
    # Calculate mask-based metrics if masks are provided
    if real_masks is not None and generated_masks is not None:
        metrics['dice'] = calculate_dice_coefficient(real_masks, generated_masks)
        metrics['iou'] = calculate_iou(real_masks, generated_masks)
        metrics['mask_accuracy'] = calculate_mask_accuracy(real_masks, generated_masks)
    
    return metrics 