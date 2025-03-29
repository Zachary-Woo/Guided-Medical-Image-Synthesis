"""
Visualization utilities for medical images.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import io
import cv2


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
    
    # Convert single-channel grayscale to RGB if needed
    if img_np.shape[2] == 1:
        img_np = np.tile(img_np, (1, 1, 3))
    
    # Convert to uint8
    img_np = (img_np * 255).astype(np.uint8)
    
    # Convert to PIL
    pil_img = Image.fromarray(img_np)
    
    return pil_img


def plot_images(images, titles=None, figsize=(15, 5), filename=None):
    """
    Plot a list of images side by side.
    
    Args:
        images (list): List of images (PIL, numpy, or tensors)
        titles (list, optional): List of titles
        figsize (tuple, optional): Figure size
        filename (str, optional): Filename to save plot
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    n_images = len(images)
    fig, axes = plt.subplots(1, n_images, figsize=figsize)
    
    if n_images == 1:
        axes = [axes]
    
    for i, img in enumerate(images):
        # Convert tensor to PIL if necessary
        if isinstance(img, torch.Tensor):
            img = tensor_to_pil(img)
        
        # Convert numpy to PIL if necessary
        elif isinstance(img, np.ndarray):
            if img.ndim == 2:
                img = np.tile(img[:, :, np.newaxis], (1, 1, 3))
            elif img.shape[2] == 1:
                img = np.tile(img, (1, 1, 3))
            
            if img.dtype == np.float32 or img.dtype == np.float64:
                img = (img * 255).astype(np.uint8)
            
            img = Image.fromarray(img)
        
        # Display image
        axes[i].imshow(img)
        axes[i].axis("off")
        
        # Add title if provided
        if titles and i < len(titles):
            axes[i].set_title(titles[i])
    
    plt.tight_layout()
    
    # Save if filename is provided
    if filename:
        plt.savefig(filename, bbox_inches="tight")
    
    return fig


def overlay_mask(image, mask, alpha=0.5, color=(255, 0, 0)):
    """
    Overlay a mask on an image.
    
    Args:
        image (PIL.Image or np.ndarray): Image
        mask (PIL.Image or np.ndarray): Mask
        alpha (float): Opacity of the overlay
        color (tuple): RGB color for the overlay
        
    Returns:
        PIL.Image: Image with mask overlay
    """
    # Convert to numpy if needed
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image.copy()
    
    if isinstance(mask, Image.Image):
        mask_np = np.array(mask)
    else:
        mask_np = mask.copy()
    
    # Convert mask to binary
    if mask_np.dtype == np.float32 or mask_np.dtype == np.float64:
        mask_np = (mask_np > 0.5).astype(np.uint8)
    else:
        mask_np = (mask_np > 128).astype(np.uint8)
    
    # Ensure image is RGB
    if image_np.ndim == 2:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    elif image_np.shape[2] == 1:
        image_np = cv2.cvtColor(image_np.squeeze(2), cv2.COLOR_GRAY2RGB)
    
    # Create colored mask
    color_mask = np.zeros_like(image_np)
    for c in range(3):
        color_mask[:, :, c] = mask_np * color[c]
    
    # Overlay mask on image
    overlaid = cv2.addWeighted(image_np, 1, color_mask, alpha, 0)
    
    # Convert back to PIL
    return Image.fromarray(overlaid)


def create_comparison_grid(real_images, generated_images, real_masks=None, 
                          generated_masks=None, n_samples=4, figsize=(15, 10)):
    """
    Create a comparison grid of real and generated images.
    
    Args:
        real_images (list or tensor): Real images
        generated_images (list or tensor): Generated images
        real_masks (list or tensor, optional): Real masks
        generated_masks (list or tensor, optional): Generated masks
        n_samples (int): Number of samples to display
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    # Convert tensors to list if needed
    if isinstance(real_images, torch.Tensor):
        real_images = [tensor_to_pil(real_images[i]) for i in range(min(n_samples, real_images.shape[0]))]
    else:
        real_images = real_images[:n_samples]
    
    if isinstance(generated_images, torch.Tensor):
        generated_images = [tensor_to_pil(generated_images[i]) for i in range(min(n_samples, generated_images.shape[0]))]
    else:
        generated_images = generated_images[:n_samples]
    
    # Create figure
    n_rows = 2
    if real_masks is not None and generated_masks is not None:
        n_rows = 4
    
    fig, axes = plt.subplots(n_rows, n_samples, figsize=figsize)
    
    # Plot real images
    for i in range(n_samples):
        axes[0, i].imshow(real_images[i])
        axes[0, i].set_title("Real Image")
        axes[0, i].axis("off")
    
    # Plot generated images
    for i in range(n_samples):
        axes[1, i].imshow(generated_images[i])
        axes[1, i].set_title("Generated Image")
        axes[1, i].axis("off")
    
    # Plot masks if provided
    if real_masks is not None and generated_masks is not None:
        # Convert masks to list if needed
        if isinstance(real_masks, torch.Tensor):
            real_masks = [tensor_to_pil(real_masks[i]) for i in range(min(n_samples, real_masks.shape[0]))]
        else:
            real_masks = real_masks[:n_samples]
        
        if isinstance(generated_masks, torch.Tensor):
            generated_masks = [tensor_to_pil(generated_masks[i]) for i in range(min(n_samples, generated_masks.shape[0]))]
        else:
            generated_masks = generated_masks[:n_samples]
        
        # Plot real masks
        for i in range(n_samples):
            axes[2, i].imshow(real_masks[i], cmap="gray")
            axes[2, i].set_title("Real Mask")
            axes[2, i].axis("off")
        
        # Plot generated masks
        for i in range(n_samples):
            axes[3, i].imshow(generated_masks[i], cmap="gray")
            axes[3, i].set_title("Generated Mask")
            axes[3, i].axis("off")
    
    plt.tight_layout()
    return fig


def plot_metrics(metrics_dict, title="Training Metrics", figsize=(10, 5), filename=None):
    """
    Plot training metrics.
    
    Args:
        metrics_dict (dict): Dictionary of metrics
        title (str): Plot title
        figsize (tuple): Figure size
        filename (str, optional): Filename to save plot
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    n_metrics = len(metrics_dict)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    
    for i, (metric_name, values) in enumerate(metrics_dict.items()):
        axes[i].plot(values)
        axes[i].set_title(metric_name)
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel(metric_name)
        axes[i].grid(True, linestyle="--", alpha=0.7)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # Save if filename is provided
    if filename:
        plt.savefig(filename, bbox_inches="tight")
    
    return fig 