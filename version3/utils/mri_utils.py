#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MRI Utility Functions
This module provides various utility functions for MRI image preprocessing and manipulation.
"""

import os
import numpy as np
from PIL import Image
import torch
import nibabel as nib
from skimage import exposure
from scipy import ndimage

def load_nifti(file_path):
    """
    Load a NIfTI file and return its data as a numpy array.
    
    Args:
        file_path: Path to the NIfTI file
        
    Returns:
        numpy.ndarray: The image data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        img = nib.load(file_path)
        return img.get_fdata()
    except Exception as e:
        raise IOError(f"Error loading NIfTI file: {e}")

def normalize_intensity(volume, percentiles=(1, 99), method="minmax"):
    """
    Normalize the intensity values of a volume.
    
    Args:
        volume: 3D numpy array
        percentiles: Tuple of lower and upper percentiles for robust scaling
        method: Normalization method, one of:
                - 'minmax': Scale to [0, 1]
                - 'zscore': Zero mean, unit variance
                - 'robust': Robust scaling using percentiles
                
    Returns:
        numpy.ndarray: Normalized volume
    """
    if method not in ["minmax", "zscore", "robust"]:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # Create a copy to avoid modifying the original
    volume_normalized = volume.copy()
    
    # Replace infinite values with NaN
    volume_normalized[np.isinf(volume_normalized)] = np.nan
    
    # Replace NaN with the mean of non-NaN values
    if np.any(np.isnan(volume_normalized)):
        nan_mask = np.isnan(volume_normalized)
        volume_normalized[nan_mask] = np.nanmean(volume_normalized)
    
    if method == "minmax":
        # Min-max normalization to [0, 1]
        min_val = np.min(volume_normalized)
        max_val = np.max(volume_normalized)
        
        # Avoid division by zero
        if max_val > min_val:
            volume_normalized = (volume_normalized - min_val) / (max_val - min_val)
        else:
            volume_normalized = np.zeros_like(volume_normalized)
            
    elif method == "zscore":
        # Z-score normalization (zero mean, unit variance)
        mean_val = np.mean(volume_normalized)
        std_val = np.std(volume_normalized)
        
        # Avoid division by zero
        if std_val > 0:
            volume_normalized = (volume_normalized - mean_val) / std_val
        else:
            volume_normalized = np.zeros_like(volume_normalized)
            
    elif method == "robust":
        # Robust scaling using percentiles
        p_low, p_high = np.percentile(volume_normalized, percentiles)
        
        # Avoid division by zero
        if p_high > p_low:
            volume_normalized = np.clip(volume_normalized, p_low, p_high)
            volume_normalized = (volume_normalized - p_low) / (p_high - p_low)
        else:
            volume_normalized = np.zeros_like(volume_normalized)
    
    return volume_normalized

def extract_brain(volume, threshold=0.1):
    """
    Simple brain extraction using thresholding and morphological operations.
    
    Args:
        volume: 3D numpy array of the brain MRI
        threshold: Threshold value for creating the binary mask
        
    Returns:
        numpy.ndarray: Brain-extracted volume
    """
    # Normalize the volume to [0, 1]
    vol_norm = normalize_intensity(volume, method="minmax")
    
    # Create a binary mask using thresholding
    mask = vol_norm > threshold
    
    # Apply morphological operations to clean up the mask
    # Fill holes
    mask = ndimage.binary_fill_holes(mask)
    
    # Remove small objects
    labels, num_features = ndimage.label(mask)
    sizes = ndimage.sum(mask, labels, range(1, num_features + 1))
    
    # Keep only the largest connected component (the brain)
    if len(sizes) > 0:
        max_label = np.argmax(sizes) + 1
        mask = labels == max_label
    
    # Apply the mask to the original volume
    brain_extracted = volume.copy()
    brain_extracted[~mask] = 0
    
    return brain_extracted

def extract_slice(volume, slice_idx=None, axis=2):
    """
    Extract a 2D slice from a 3D volume.
    
    Args:
        volume: 3D numpy array
        slice_idx: Index of the slice to extract (if None, the middle slice is used)
        axis: Axis along which to extract the slice (0=sagittal, 1=coronal, 2=axial)
        
    Returns:
        numpy.ndarray: 2D slice
    """
    if axis not in [0, 1, 2]:
        raise ValueError(f"Invalid axis: {axis}. Must be 0, 1, or 2.")
    
    # Determine the slice index if not provided
    if slice_idx is None:
        slice_idx = volume.shape[axis] // 2
    
    # Extract the slice
    if axis == 0:
        slice_2d = volume[slice_idx, :, :]
    elif axis == 1:
        slice_2d = volume[:, slice_idx, :]
    else:  # axis == 2
        slice_2d = volume[:, :, slice_idx]
    
    return slice_2d

def enhance_contrast(image, clip_limit=0.03):
    """
    Enhance the contrast of an image using CLAHE.
    
    Args:
        image: 2D numpy array
        clip_limit: Clipping limit for CLAHE
        
    Returns:
        numpy.ndarray: Contrast-enhanced image
    """
    # Normalize to [0, 1] if necessary
    if image.max() > 1.0:
        image = image / image.max()
        
    # Apply CLAHE
    return exposure.equalize_adapthist(image, clip_limit=clip_limit)

def mri_to_pil(slice_data, normalize=True, enhance=False):
    """
    Convert an MRI slice to a PIL image.
    
    Args:
        slice_data: 2D numpy array representing the MRI slice
        normalize: Whether to normalize the intensity values
        enhance: Whether to enhance contrast
        
    Returns:
        PIL.Image: The MRI slice as a PIL image
    """
    # Create a copy to avoid modifying the original
    img_data = slice_data.copy()
    
    # Normalize if requested
    if normalize:
        img_data = normalize_intensity(img_data, method="minmax")
    
    # Enhance contrast if requested
    if enhance:
        img_data = enhance_contrast(img_data)
    
    # Convert to uint8
    img_data = (img_data * 255).astype(np.uint8)
    
    # Create PIL image
    pil_image = Image.fromarray(img_data)
    
    return pil_image

def create_rgb_from_modalities(t1, t2, flair, normalize=True):
    """
    Create an RGB image from three MRI modalities.
    
    Args:
        t1: 2D numpy array of T1 slice
        t2: 2D numpy array of T2 slice
        flair: 2D numpy array of FLAIR slice
        normalize: Whether to normalize each modality
        
    Returns:
        PIL.Image: RGB image combining the three modalities
    """
    # Normalize if requested
    if normalize:
        t1 = normalize_intensity(t1, method="minmax")
        t2 = normalize_intensity(t2, method="minmax")
        flair = normalize_intensity(flair, method="minmax")
    
    # Create RGB array
    rgb = np.stack([t1, t2, flair], axis=2)
    
    # Scale to 0-255 and convert to uint8
    rgb = (rgb * 255).astype(np.uint8)
    
    # Create PIL image
    return Image.fromarray(rgb)

def get_central_slices(volume, margin=5):
    """
    Extract the central slices from a volume with a margin.
    
    Args:
        volume: 3D numpy array
        margin: Number of slices to take on each side of the center
        
    Returns:
        list: List of 2D numpy arrays representing the central slices
    """
    # Determine the center slice
    center = volume.shape[2] // 2
    
    # Calculate slice range
    start = max(0, center - margin)
    end = min(volume.shape[2], center + margin + 1)
    
    # Extract slices
    slices = [volume[:, :, i] for i in range(start, end)]
    
    return slices

def resize_slice(slice_data, target_size=(256, 256), order=1):
    """
    Resize an MRI slice to the target size.
    
    Args:
        slice_data: 2D numpy array representing the MRI slice
        target_size: Tuple of (height, width) for the output size
        order: Interpolation order (0=nearest, 1=linear, 2=quadratic, etc.)
        
    Returns:
        numpy.ndarray: Resized slice
    """
    from skimage.transform import resize
    
    # Resize the slice
    return resize(slice_data, target_size, order=order, mode='constant', anti_aliasing=True)

def create_segmentation_overlay(mri_slice, seg_slice, alpha=0.5):
    """
    Create an overlay of a segmentation on an MRI slice.
    
    Args:
        mri_slice: 2D numpy array of the MRI slice
        seg_slice: 2D numpy array of the segmentation mask
        alpha: Opacity of the segmentation overlay
        
    Returns:
        PIL.Image: MRI with segmentation overlay
    """
    # Normalize MRI slice
    mri_norm = normalize_intensity(mri_slice, method="minmax")
    mri_rgb = np.stack([mri_norm, mri_norm, mri_norm], axis=2)
    
    # Create a colormap for the segmentation
    # Assuming seg_slice contains integer labels
    from matplotlib import cm
    cmap = cm.get_cmap('rainbow')
    
    unique_labels = np.unique(seg_slice)
    unique_labels = unique_labels[unique_labels > 0]  # Exclude background
    
    # Create RGB segmentation image
    seg_rgb = np.zeros_like(mri_rgb)
    
    for label in unique_labels:
        mask = seg_slice == label
        color = np.array(cmap(label / (len(unique_labels) + 1))[0:3])
        
        for i in range(3):
            seg_rgb[:, :, i][mask] = color[i]
    
    # Blend the images
    overlay = mri_rgb * (1 - alpha) + seg_rgb * alpha
    
    # Convert to uint8
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)
    
    # Create PIL image
    return Image.fromarray(overlay)

def batch_preprocess_mri(input_dir, output_dir, modality="t1", slice_axis=2, resize=(512, 512)):
    """
    Batch preprocess MRI volumes in a directory and save as PNG images.
    
    Args:
        input_dir: Directory containing NIfTI files
        output_dir: Directory to save PNG images
        modality: Modality to process (used for file naming)
        slice_axis: Axis to extract slices from (0=sagittal, 1=coronal, 2=axial)
        resize: Target size for the slices
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all NIfTI files in the input directory
    nifti_files = [f for f in os.listdir(input_dir) if f.endswith(('.nii', '.nii.gz'))]
    
    for nifti_file in nifti_files:
        # Load the NIfTI file
        file_path = os.path.join(input_dir, nifti_file)
        volume = load_nifti(file_path)
        
        # Extract the central slices
        central_slices = get_central_slices(volume)
        
        # Process each slice
        for i, slice_data in enumerate(central_slices):
            # Normalize and resize the slice
            slice_norm = normalize_intensity(slice_data)
            slice_resized = resize_slice(slice_norm, resize)
            
            # Convert to PIL image
            pil_image = mri_to_pil(slice_resized, normalize=False)
            
            # Save the image
            base_name = os.path.splitext(nifti_file)[0]
            if base_name.endswith('.nii'):
                base_name = os.path.splitext(base_name)[0]
                
            output_path = os.path.join(output_dir, f"{base_name}_{modality}_slice{i}.png")
            pil_image.save(output_path)

def create_tensor_from_mri(mri_path, device='cuda'):
    """
    Create a tensor from an MRI volume for model input.
    
    Args:
        mri_path: Path to the NIfTI file
        device: Device to put the tensor on
        
    Returns:
        torch.Tensor: Tensor representation of the MRI
    """
    # Load the NIfTI file
    volume = load_nifti(mri_path)
    
    # Get the central slice
    slice_data = extract_slice(volume)
    
    # Normalize the slice
    slice_norm = normalize_intensity(slice_data)
    
    # Resize to 512x512
    slice_resized = resize_slice(slice_norm, (512, 512))
    
    # Convert to tensor and add batch and channel dimensions
    tensor = torch.tensor(slice_resized, dtype=torch.float32, device=device)
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    
    # Normalize to [-1, 1]
    tensor = tensor * 2 - 1
    
    return tensor

def pil_to_mri(image):
    """
    Convert a PIL image to an MRI slice (normalized 0-1 float array).
    
    Args:
        image (PIL.Image): Input PIL image
    
    Returns:
        np.ndarray: MRI slice as normalized float array
    """
    # Convert to numpy array
    image_array = np.array(image)
    
    # Convert to float and normalize to 0-1
    mri_slice = image_array.astype(np.float32) / 255.0
    
    return mri_slice

def apply_brain_mask(volume, mask):
    """
    Apply brain mask to the volume.
    
    Args:
        volume (np.ndarray): Input volume
        mask (np.ndarray): Binary brain mask
    
    Returns:
        np.ndarray: Masked volume
    """
    return volume * mask

def z_score_normalization(volume, mask=None):
    """
    Apply Z-score normalization to the volume.
    
    Args:
        volume (np.ndarray): Input volume
        mask (np.ndarray, optional): Binary mask to define the region for computing mean and std
    
    Returns:
        np.ndarray: Z-score normalized volume
    """
    if mask is None:
        # Use non-zero voxels if no mask is provided
        mask = volume > 0
    
    # Calculate mean and std within the mask
    mean_val = np.mean(volume[mask])
    std_val = np.std(volume[mask])
    
    # Avoid division by zero
    if std_val == 0:
        return volume - mean_val
    
    # Apply Z-score normalization
    normalized = (volume - mean_val) / std_val
    
    return normalized

def histogram_matching(source, reference, mask=None):
    """
    Match histogram of source image to reference image.
    
    Args:
        source (np.ndarray): Source image to be transformed
        reference (np.ndarray): Reference image
        mask (np.ndarray, optional): Binary mask defining the region for matching
    
    Returns:
        np.ndarray: Histogram-matched image
    """
    if mask is None:
        source_values = source.flatten()
        reference_values = reference.flatten()
    else:
        source_values = source[mask].flatten()
        reference_values = reference[mask].flatten()
    
    # Get unique values and indices for source and reference
    source_unique, source_indices = np.unique(source_values, return_inverse=True)
    reference_unique, reference_counts = np.unique(reference_values, return_counts=True)
    
    # Calculate CDFs
    reference_cdf = np.cumsum(reference_counts) / reference_counts.sum()
    
    # Create mapping function
    source_size = source_unique.size
    reference_size = reference_unique.size
    
    # Map each source value to the reference value with the closest CDF
    source_cdf = np.arange(1, source_size + 1) / source_size
    mapping = np.interp(source_cdf, reference_cdf, reference_unique)
    
    # Apply mapping
    result = np.empty_like(source)
    if mask is None:
        result = mapping[source_indices].reshape(source.shape)
    else:
        result = source.copy()
        result[mask] = mapping[source_indices]
    
    return result

def resample_volume(volume, target_shape):
    """
    Resample volume to target shape.
    
    Args:
        volume (np.ndarray): Input volume
        target_shape (tuple): Target shape for resampling
    
    Returns:
        np.ndarray: Resampled volume
    """
    # Calculate zoom factors
    zoom_factors = [t / s for t, s in zip(target_shape, volume.shape)]
    
    # Apply zoom
    resampled = ndimage.zoom(volume, zoom_factors, order=3)
    
    return resampled

def apply_window_level(volume, window=None, level=None):
    """
    Apply window/level (contrast/brightness) adjustment to the volume.
    
    Args:
        volume (np.ndarray): Input volume
        window (float, optional): Window width (contrast)
        level (float, optional): Window center (brightness)
    
    Returns:
        np.ndarray: Windowed volume
    """
    if window is None or level is None:
        # Auto window/level based on percentiles
        p_low = np.percentile(volume[volume > 0], 1)
        p_high = np.percentile(volume[volume > 0], 99)
        
        if window is None:
            window = p_high - p_low
        
        if level is None:
            level = (p_high + p_low) / 2
    
    # Apply window/level
    low = level - window/2
    high = level + window/2
    
    return np.clip(volume, low, high)

def batch_to_pil(tensor):
    """
    Convert a batch of tensors to PIL images.
    
    Args:
        tensor (torch.Tensor): Batch of images [B, C, H, W]
    
    Returns:
        list: List of PIL images
    """
    images = []
    for i in range(tensor.shape[0]):
        # Get single image
        img = tensor[i]
        
        # Denormalize
        img = (img * 0.5 + 0.5).clamp(0, 1)
        
        # Convert to numpy
        img = img.cpu().numpy()
        
        # Handle different channel dimensions
        if img.shape[0] == 1:  # Grayscale
            img = img[0]
            img = (img * 255).astype(np.uint8)
            pil_img = Image.fromarray(img, mode='L')
        else:  # RGB
            img = img.transpose(1, 2, 0)
            img = (img * 255).astype(np.uint8)
            pil_img = Image.fromarray(img, mode='RGB')
        
        images.append(pil_img)
    
    return images

def tensor_to_segmentation_mask(tensor, num_classes=4):
    """
    Convert segmentation tensor to multi-class mask.
    
    Args:
        tensor (torch.Tensor): Segmentation tensor [B, C, H, W] or [B, H, W]
        num_classes (int): Number of classes
    
    Returns:
        list: List of segmentation masks as numpy arrays
    """
    masks = []
    
    # Handle single-channel case (class indices)
    if len(tensor.shape) == 3 or tensor.shape[1] == 1:
        if len(tensor.shape) == 4:
            tensor = tensor.squeeze(1)
        
        for i in range(tensor.shape[0]):
            mask = tensor[i].cpu().numpy()
            masks.append(mask)
    
    # Handle multi-channel case (one-hot or probabilities)
    else:
        for i in range(tensor.shape[0]):
            # Convert to numpy
            img = tensor[i].cpu().numpy()
            
            # Get class with maximum probability
            mask = np.argmax(img, axis=0)
            masks.append(mask)
    
    return masks

def combine_segmentation_masks(masks, colors=None):
    """
    Combine segmentation masks into a single RGB image.
    
    Args:
        masks (list): List of binary masks, one for each class
        colors (list, optional): List of RGB colors for each class
    
    Returns:
        np.ndarray: Combined RGB image
    """
    if colors is None:
        # Default colors for BraTS classes: background, necrotic, edema, enhancing
        colors = [
            [0, 0, 0],  # Background - black
            [255, 0, 0],  # Necrotic - red
            [0, 255, 0],  # Edema - green
            [0, 0, 255],  # Enhancing - blue
        ]
    
    num_classes = len(masks)
    h, w = masks[0].shape
    
    # Create RGB image
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Fill with colors
    for i in range(num_classes):
        rgb_mask[masks[i] > 0] = colors[i]
    
    return rgb_mask

def overlay_segmentation(image, segmentation, alpha=0.5, colors=None):
    """
    Overlay segmentation mask on top of an image.
    
    Args:
        image (np.ndarray): Base image (grayscale or RGB)
        segmentation (np.ndarray): Segmentation mask
        alpha (float): Transparency of the overlay
        colors (list, optional): List of RGB colors for each class
    
    Returns:
        np.ndarray: Image with segmentation overlay
    """
    # Ensure image is RGB
    if len(image.shape) == 2 or image.shape[2] == 1:
        image_rgb = np.stack([image] * 3, axis=-1)
        if image_rgb.shape[2] == 1:
            image_rgb = image_rgb.squeeze(2)
    else:
        image_rgb = image.copy()
    
    # Convert image to uint8 if needed
    if image_rgb.dtype != np.uint8:
        image_rgb = (image_rgb * 255).astype(np.uint8)
    
    # Handle segmentation mask as multi-class
    if colors is None:
        # Default colors for BraTS classes
        colors = [
            [0, 0, 0],  # Background - black
            [255, 0, 0],  # Necrotic - red
            [0, 255, 0],  # Edema - green
            [0, 0, 255],  # Enhancing - blue
        ]
    
    # Create segmentation overlay
    segmentation_rgb = np.zeros_like(image_rgb)
    for i in range(1, len(colors)):  # Skip background
        segmentation_rgb[segmentation == i] = colors[i]
    
    # Blend images
    mask = segmentation > 0
    blended = np.where(
        np.stack([mask] * 3, axis=-1),
        (1 - alpha) * image_rgb + alpha * segmentation_rgb,
        image_rgb
    )
    
    return blended.astype(np.uint8)

def crop_to_brain(volume, mask=None, margin=10):
    """
    Crop volume to brain region.
    
    Args:
        volume (np.ndarray): Input volume
        mask (np.ndarray, optional): Binary brain mask. If None, will be computed
        margin (int): Margin around brain region
    
    Returns:
        np.ndarray: Cropped volume
        tuple: Bounding box (min_z, max_z, min_y, max_y, min_x, max_x)
    """
    if mask is None:
        mask = extract_brain(volume)
    
    # Find bounding box
    z_indices, y_indices, x_indices = np.where(mask > 0)
    
    min_z, max_z = np.min(z_indices), np.max(z_indices)
    min_y, max_y = np.min(y_indices), np.max(y_indices)
    min_x, max_x = np.min(x_indices), np.max(x_indices)
    
    # Add margin
    min_z = max(0, min_z - margin)
    max_z = min(volume.shape[0] - 1, max_z + margin)
    min_y = max(0, min_y - margin)
    max_y = min(volume.shape[1] - 1, max_y + margin)
    min_x = max(0, min_x - margin)
    max_x = min(volume.shape[2] - 1, max_x + margin)
    
    # Crop volume
    cropped = volume[min_z:max_z+1, min_y:max_y+1, min_x:max_x+1]
    bbox = (min_z, max_z, min_y, max_y, min_x, max_x)
    
    return cropped, bbox 