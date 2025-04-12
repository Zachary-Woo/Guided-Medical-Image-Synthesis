import numpy as np
from PIL import Image
from typing import Union, Tuple, Optional, Literal
import SimpleITK as sitk
import os
import nibabel as nib

def normalize_intensity(
    img: np.ndarray, 
    normalization: Literal["min_max", "z_score", "percentile"] = "min_max",
    min_val: float = 0.0,
    max_val: float = 1.0,
    percentile_low: float = 1.0,
    percentile_high: float = 99.0
) -> np.ndarray:
    """
    Normalize the intensity of an MRI image.
    
    Args:
        img: Input image as numpy array
        normalization: Normalization method to use
            - "min_max": Normalize to range [min_val, max_val]
            - "z_score": Z-score normalization (mean=0, std=1)
            - "percentile": Normalize based on percentiles
        min_val: Minimum value for min_max normalization
        max_val: Maximum value for min_max normalization
        percentile_low: Lower percentile for percentile normalization
        percentile_high: Upper percentile for percentile normalization
        
    Returns:
        Normalized image as numpy array
    """
    if img.size == 0:
        return img
    
    if normalization == "min_max":
        img_min = img.min()
        img_max = img.max()
        
        # Avoid division by zero
        if img_min == img_max:
            return np.zeros_like(img)
        
        normalized = (img - img_min) / (img_max - img_min)
        
        if min_val != 0.0 or max_val != 1.0:
            normalized = normalized * (max_val - min_val) + min_val
            
    elif normalization == "z_score":
        mean = img.mean()
        std = img.std()
        
        if std == 0:
            return np.zeros_like(img)
            
        normalized = (img - mean) / std
        
    elif normalization == "percentile":
        low = np.percentile(img, percentile_low)
        high = np.percentile(img, percentile_high)
        
        if low == high:
            return np.zeros_like(img)
            
        normalized = np.clip(img, low, high)
        normalized = (normalized - low) / (high - low)
        
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")
        
    return normalized

def adjust_contrast(
    img: np.ndarray, 
    contrast_factor: float = 1.0,
    brightness_factor: float = 0.0
) -> np.ndarray:
    """
    Adjust contrast and brightness of an image.
    
    Args:
        img: Input image as numpy array (normalized to [0, 1])
        contrast_factor: Contrast adjustment factor (1.0 = no change)
        brightness_factor: Brightness adjustment factor (0.0 = no change)
        
    Returns:
        Contrast-adjusted image as numpy array
    """
    # Apply contrast adjustment: new_val = (val - 0.5) * contrast_factor + 0.5 + brightness_factor
    adjusted = (img - 0.5) * contrast_factor + 0.5 + brightness_factor
    
    # Clip values to [0, 1] range
    return np.clip(adjusted, 0, 1)

def histogram_matching(
    source_img: np.ndarray, 
    reference_img: np.ndarray,
    histogram_levels: int = 1024,
    match_points: int = 100
) -> np.ndarray:
    """
    Apply histogram matching to make the histogram of source_img match reference_img.
    
    Args:
        source_img: Source image to transform
        reference_img: Reference image whose histogram will be matched
        histogram_levels: Number of histogram levels to use
        match_points: Number of points to match in the histograms
        
    Returns:
        Histogram-matched image
    """
    # Convert numpy arrays to SimpleITK images
    source_sitk = sitk.GetImageFromArray(source_img.astype(np.float32))
    reference_sitk = sitk.GetImageFromArray(reference_img.astype(np.float32))
    
    # Create histogram matcher
    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(histogram_levels)
    matcher.SetNumberOfMatchPoints(match_points)
    matcher.SetThresholdAtMeanIntensity(True)
    
    # Apply histogram matching
    matched_sitk = matcher.Execute(source_sitk, reference_sitk)
    
    # Convert back to numpy array
    matched_img = sitk.GetArrayFromImage(matched_sitk)
    
    return matched_img

def mri_to_rgb(
    img: np.ndarray, 
    colormap: Optional[str] = None,
    channels: Optional[int] = 3
) -> np.ndarray:
    """
    Convert single-channel MRI to RGB/RGBA image.
    
    Args:
        img: Input MRI image as numpy array (normalized to [0, 1])
        colormap: Colormap to use (None for grayscale)
        channels: Number of channels in output (3 for RGB, 4 for RGBA)
        
    Returns:
        RGB/RGBA image
    """
    if channels not in [3, 4]:
        raise ValueError("Channels must be 3 (RGB) or 4 (RGBA)")
    
    # Create RGB image (grayscale)
    if colormap is None:
        rgb_img = np.stack([img] * 3, axis=-1)
        
        if channels == 4:
            # Add alpha channel (fully opaque)
            alpha = np.ones_like(img)
            rgba_img = np.concatenate([rgb_img, alpha[..., np.newaxis]], axis=-1)
            return rgba_img
            
        return rgb_img
        
    else:
        # Import matplotlib for colormaps
        import matplotlib.pyplot as plt
        
        # Get colormap
        cmap = plt.get_cmap(colormap)
        
        # Apply colormap
        rgb_img = cmap(img)
        
        if channels == 3:
            # Remove alpha channel
            return rgb_img[..., :3]
            
        return rgb_img

def load_and_preprocess_nifti(
    file_path: str,
    slice_dim: int = 2,
    slice_idx: Optional[int] = None,
    normalization: str = "min_max",
    resize: Optional[Tuple[int, int]] = None
) -> Union[np.ndarray, list]:
    """
    Load and preprocess a NIfTI file.
    
    Args:
        file_path: Path to NIfTI file
        slice_dim: Dimension to slice (0, 1, or 2)
        slice_idx: Slice index to extract (None for all slices)
        normalization: Normalization method
        resize: Target size for resizing (None for no resizing)
        
    Returns:
        Preprocessed image(s) as numpy array or list of arrays
    """
    # Load NIfTI file
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    nifti_img = nib.load(file_path)
    img_data = nifti_img.get_fdata()
    
    # Extract slice or slices
    if slice_idx is not None:
        if slice_dim == 0:
            img_slice = img_data[slice_idx, :, :]
        elif slice_dim == 1:
            img_slice = img_data[:, slice_idx, :]
        else:
            img_slice = img_data[:, :, slice_idx]
            
        # Normalize intensity
        img_slice = normalize_intensity(img_slice, normalization=normalization)
        
        # Resize if needed
        if resize is not None:
            img_pil = Image.fromarray((img_slice * 255).astype(np.uint8))
            img_pil = img_pil.resize(resize, Image.BILINEAR)
            img_slice = np.array(img_pil).astype(np.float32) / 255.0
            
        return img_slice
        
    else:
        # Process all slices
        slices = []
        
        num_slices = img_data.shape[slice_dim]
        
        for i in range(num_slices):
            if slice_dim == 0:
                img_slice = img_data[i, :, :]
            elif slice_dim == 1:
                img_slice = img_data[:, i, :]
            else:
                img_slice = img_data[:, :, i]
                
            # Skip empty slices
            if np.all(img_slice == 0):
                continue
                
            # Normalize intensity
            img_slice = normalize_intensity(img_slice, normalization=normalization)
            
            # Resize if needed
            if resize is not None:
                img_pil = Image.fromarray((img_slice * 255).astype(np.uint8))
                img_pil = img_pil.resize(resize, Image.BILINEAR)
                img_slice = np.array(img_pil).astype(np.float32) / 255.0
                
            slices.append(img_slice)
            
        return slices 