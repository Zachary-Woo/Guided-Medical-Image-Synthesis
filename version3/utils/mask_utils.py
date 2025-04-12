"""
Mask Utilities for Brain MRI Tumor Masks

This module provides utility functions for working with brain tumor masks, including:
- Converting between different mask formats
- Analyzing tumor masks for location, size, etc.
- Preparing masks for ControlNet conditioning
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def create_circular_tumor_mask(size=512, location=(256, 256), radius=50, blur_radius=15):
    """
    Create a synthetic circular tumor mask.
    
    Args:
        size (int): Size of the square mask
        location (tuple): (x, y) coordinates of the tumor center
        radius (int): Radius of the tumor in pixels
        blur_radius (int): Gaussian blur radius to apply to the mask edges
        
    Returns:
        PIL.Image: Binary mask with white tumor on black background
    """
    # Create blank mask
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    
    # Draw tumor as white circle
    x, y = location
    draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=255)
    
    # Apply Gaussian blur to make edges more realistic
    if blur_radius > 0:
        mask_np = np.array(mask)
        mask_np = cv2.GaussianBlur(mask_np, (blur_radius*2+1, blur_radius*2+1), blur_radius/3)
        mask = Image.fromarray(mask_np)
    
    return mask

def create_irregular_tumor_mask(size=512, location=(256, 256), avg_radius=50, irregularity=0.3, spikes=10, blur_radius=15):
    """
    Create a synthetic irregular tumor mask with a more realistic shape.
    
    Args:
        size (int): Size of the square mask
        location (tuple): (x, y) coordinates of the tumor center
        avg_radius (int): Average radius of the tumor in pixels
        irregularity (float): 0-1 value controlling shape irregularity
        spikes (int): Number of spikes/protrusions
        blur_radius (int): Gaussian blur radius to apply to the mask edges
        
    Returns:
        PIL.Image: Binary mask with white tumor on black background
    """
    # Create blank mask
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    
    # Get center coordinates
    cx, cy = location
    
    # Generate points around a circle
    angles = np.linspace(0, 2*np.pi, spikes, endpoint=False)
    
    # Create random radius variations
    np.random.seed(42)  # For reproducibility, but can be varied
    radii = np.random.normal(avg_radius, avg_radius * irregularity, spikes)
    
    # Create list of (x, y) points for the tumor boundary
    points = []
    for angle, radius in zip(angles, radii):
        x = cx + radius * np.cos(angle)
        y = cy + radius * np.sin(angle)
        points.append((x, y))
    
    # Draw the polygon
    draw.polygon(points, fill=255)
    
    # Apply Gaussian blur to make edges more realistic
    if blur_radius > 0:
        mask_np = np.array(mask)
        mask_np = cv2.GaussianBlur(mask_np, (blur_radius*2+1, blur_radius*2+1), blur_radius/3)
        mask = Image.fromarray(mask_np)
    
    return mask

def analyze_tumor_mask(mask_image):
    """
    Analyze a tumor mask to extract properties like location, size, etc.
    
    Args:
        mask_image (PIL.Image or numpy.ndarray): Tumor mask
        
    Returns:
        dict: Dictionary with tumor properties
    """
    # Convert to numpy array if PIL image
    if isinstance(mask_image, Image.Image):
        mask_np = np.array(mask_image.convert("L"))
    else:
        mask_np = mask_image.copy()
    
    # Threshold to make binary if not already
    if mask_np.max() > 1:
        _, mask_np = cv2.threshold(mask_np, 127, 1, cv2.THRESH_BINARY)
    
    # Get mask shape
    h, w = mask_np.shape
    
    # Find tumor center
    tumor_pixels = np.where(mask_np > 0)
    if len(tumor_pixels[0]) == 0:
        logger.warning("No tumor found in mask")
        return {
            "has_tumor": False,
            "tumor_center": (w//2, h//2),
            "tumor_size_pixels": 0,
            "tumor_fraction": 0,
            "tumor_position": "center"
        }
    
    y_center = np.mean(tumor_pixels[0])
    x_center = np.mean(tumor_pixels[1])
    
    # Calculate tumor size
    tumor_size_pixels = len(tumor_pixels[0])
    tumor_fraction = tumor_size_pixels / (h * w)
    
    # Determine tumor position (left/right/center, top/bottom/middle)
    x_rel = x_center / w
    y_rel = y_center / h
    
    if x_rel < 0.4:
        x_position = "left"
    elif x_rel > 0.6:
        x_position = "right"
    else:
        x_position = "center"
    
    if y_rel < 0.4:
        y_position = "frontal"  # Top in image is frontal in brain
    elif y_rel > 0.6:
        y_position = "occipital"  # Bottom in image is occipital in brain
    else:
        y_position = "central"
    
    if y_position == "central":
        if x_position == "center":
            position = "central"
        else:
            position = f"{x_position} temporal"  # Side middle is temporal
    else:
        position = f"{x_position} {y_position}"
    
    # Determine size category
    if tumor_fraction < 0.02:
        size_category = "small"
    elif tumor_fraction < 0.07:
        size_category = "medium"
    else:
        size_category = "large"
    
    return {
        "has_tumor": True,
        "tumor_center": (x_center, y_center),
        "tumor_size_pixels": tumor_size_pixels,
        "tumor_fraction": tumor_fraction,
        "tumor_position": position,
        "size_category": size_category,
        "x_position": x_position,
        "y_position": y_position
    }

def mask_to_controlnet_input(mask, use_color=True, brain_outline=True):
    """
    Convert a binary tumor mask to a format suitable for ControlNet input.
    
    Args:
        mask (PIL.Image or numpy.ndarray): Binary tumor mask
        use_color (bool): If True, use colored output (green for tumor)
        brain_outline (bool): If True, add a brain outline to the mask
        
    Returns:
        PIL.Image: Processed mask for ControlNet input
    """
    # Convert to numpy array if PIL image
    if isinstance(mask, Image.Image):
        mask_np = np.array(mask.convert("L"))
    else:
        mask_np = mask.copy()
    
    # Create empty RGB image
    h, w = mask_np.shape
    if use_color:
        condition_np = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Green for tumor
        condition_np[mask_np > 127, 1] = 255
    else:
        # Use grayscale
        condition_np = np.zeros((h, w, 3), dtype=np.uint8)
        condition_np[..., 0] = mask_np
        condition_np[..., 1] = mask_np
        condition_np[..., 2] = mask_np
    
    # Add brain outline if requested
    if brain_outline:
        # Create a new image for the brain outline
        brain_mask = np.zeros((h, w), dtype=np.uint8)
        center_x, center_y = w // 2, h // 2
        radius = min(w, h) // 2 - 20  # Slightly smaller than the image
        
        # Draw the brain outline
        cv2.circle(brain_mask, (center_x, center_y), radius, 128, 3)
        
        # Add to red channel (makes a light red outline)
        condition_np[brain_mask > 0, 0] = 128
    
    # Convert back to PIL
    return Image.fromarray(condition_np)

def create_mask_from_description(description, size=512):
    """
    Create a tumor mask based on a text description.
    
    Args:
        description (str): Text description of the tumor location
        size (int): Size of the output mask
        
    Returns:
        PIL.Image: Binary tumor mask
    """
    # Define region mapping from text to coordinates
    # Format: (x, y) coordinates in a virtual 300x300 space
    # that will be scaled to the actual output size
    region_map = {
        "frontal": (150, 120),
        "temporal": (150, 180),
        "left temporal": (100, 180),
        "right temporal": (200, 180),
        "left frontal": (100, 120),
        "right frontal": (200, 120),
        "parietal": (150, 150),
        "left parietal": (100, 150),
        "right parietal": (200, 150), 
        "occipital": (150, 200),
        "left occipital": (100, 200),
        "right occipital": (200, 200),
        "cerebellum": (150, 230),
        "brain stem": (150, 250),
        "thalamus": (150, 170),
        "left thalamus": (130, 170),
        "right thalamus": (170, 170),
        "basal ganglia": (150, 160)
    }
    
    # Default location and size
    location = (150, 150)  # Center in the virtual space
    avg_radius = 30
    
    # Look for size hints
    description = description.lower()
    if "small" in description:
        avg_radius = 15
    elif "large" in description:
        avg_radius = 45
    
    # Look for location hints
    for region, coords in region_map.items():
        if region in description.lower():
            location = coords
            logger.info(f"Creating mask for tumor in {region} region")
            break
    
    # Scale coordinates to the actual image size
    x, y = location
    x = int(x * size / 300)
    y = int(y * size / 300)
    radius = int(avg_radius * size / 300)
    
    # Create the mask - use irregular for more realism
    if "irregular" in description.lower():
        mask = create_irregular_tumor_mask(size, (x, y), radius)
    else:
        mask = create_circular_tumor_mask(size, (x, y), radius)
    
    return mask

def blend_mask_with_image(image, mask, alpha=0.5, color=(0, 255, 0)):
    """
    Blend a tumor mask with an image for visualization.
    
    Args:
        image (PIL.Image or numpy.ndarray): Base image
        mask (PIL.Image or numpy.ndarray): Binary mask
        alpha (float): Blend opacity
        color (tuple): RGB color to use for the mask
        
    Returns:
        PIL.Image: Blended visualization
    """
    # Convert to numpy arrays
    if isinstance(image, Image.Image):
        image_np = np.array(image.convert("RGB"))
    else:
        image_np = image.copy()
    
    if isinstance(mask, Image.Image):
        mask_np = np.array(mask.convert("L"))
    else:
        mask_np = mask.copy()
    
    # Create a color overlay
    overlay = np.zeros_like(image_np, dtype=np.uint8)
    overlay[mask_np > 127] = color
    
    # Blend images
    blended = cv2.addWeighted(image_np, 1.0, overlay, alpha, 0)
    
    # Draw the contour
    contours, _ = cv2.findContours(
        (mask_np > 127).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(blended, contours, -1, (255, 255, 0), 2)
    
    return Image.fromarray(blended)

def prepare_mask_batch(masks, size=512):
    """
    Prepare a batch of masks for ControlNet conditioning.
    
    Args:
        masks (list): List of mask images or file paths
        size (int): Target size for masks
        
    Returns:
        torch.Tensor: Batch of masks in format ready for ControlNet
    """
    batch = []
    
    for mask in masks:
        # Load mask if it's a path
        if isinstance(mask, (str, Path)):
            mask = Image.open(mask).convert("L")
        
        # Convert to ControlNet input format
        controlnet_input = mask_to_controlnet_input(mask)
        
        # Resize if needed
        if controlnet_input.width != size or controlnet_input.height != size:
            controlnet_input = controlnet_input.resize((size, size), Image.LANCZOS)
        
        # Convert to numpy and normalize
        input_np = np.array(controlnet_input).astype(np.float32) / 255.0
        
        # Add to batch
        batch.append(input_np)
    
    # Stack and convert to tensor
    batch_np = np.stack(batch)
    batch_tensor = torch.from_numpy(batch_np).permute(0, 3, 1, 2)  # NHWC -> NCHW
    
    return batch_tensor 