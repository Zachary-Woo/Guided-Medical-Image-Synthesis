"""
Stain normalization utilities for histopathology images.
This module provides implementations of both Macenko and Reinhard normalization methods,
which are widely used in digital pathology to standardize H&E stain appearances.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class StainNormalizer:
    """Base class for stain normalization techniques"""
    
    def __init__(self):
        self.target_means = None
        self.target_stds = None
        
    def fit(self):
        """Fit normalizer to a target image"""
        raise NotImplementedError("Subclasses must implement fit method")
    
    def transform(self):
        """Transform source image to match target image staining"""
        raise NotImplementedError("Subclasses must implement transform method")
    
    def fit_transform(self, source_image, target_image):
        """Fit to target image and transform source image"""
        self.fit(target_image)
        return self.transform(source_image)


class ReinhardNormalizer(StainNormalizer):
    """
    Reinhard stain normalization as described in:
    "Color transfer between images" by Reinhard et al.
    
    This technique normalizes the color distribution of a source image to match a target image
    in LAB color space, which is effective for many histopathology applications.
    """
    
    def __init__(self):
        super().__init__()
        self.target_means = None
        self.target_stds = None
    
    def fit(self, target_image):
        """Fit normalizer to a target image"""
        if isinstance(target_image, str):
            target_image = np.array(Image.open(target_image).convert('RGB'))
        elif isinstance(target_image, Image.Image):
            target_image = np.array(target_image)
            
        # Convert to LAB color space
        target_lab = cv2.cvtColor(target_image, cv2.COLOR_RGB2LAB)
        
        # Calculate mean and std in LAB color space
        self.target_means = np.mean(target_lab, axis=(0, 1))
        self.target_stds = np.std(target_lab, axis=(0, 1))
        
        return self
    
    def transform(self, source_image):
        """Transform source image to match target image staining"""
        if self.target_means is None or self.target_stds is None:
            raise ValueError("Normalizer has not been fit to a target image. Call fit() first.")
        
        if isinstance(source_image, str):
            source_image = np.array(Image.open(source_image).convert('RGB'))
        elif isinstance(source_image, Image.Image):
            source_image = np.array(source_image)
        
        # Convert to LAB color space
        source_lab = cv2.cvtColor(source_image, cv2.COLOR_RGB2LAB)
        
        # Calculate source statistics
        source_means = np.mean(source_lab, axis=(0, 1))
        source_stds = np.std(source_lab, axis=(0, 1))
        
        # Normalize source image
        normalized_lab = source_lab.copy().astype(float)
        
        # For each channel
        for i in range(3):
            # Center the data
            normalized_lab[:, :, i] = normalized_lab[:, :, i] - source_means[i]
            
            # Scale the data
            if source_stds[i] > 0:
                normalized_lab[:, :, i] = normalized_lab[:, :, i] * (self.target_stds[i] / source_stds[i])
            
            # Recenter to target mean
            normalized_lab[:, :, i] = normalized_lab[:, :, i] + self.target_means[i]
        
        # Clip values to valid LAB range
        normalized_lab = np.clip(normalized_lab, 0, 255).astype(np.uint8)
        
        # Convert back to RGB
        normalized_rgb = cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2RGB)
        
        return normalized_rgb


class MacenkoNormalizer(StainNormalizer):
    """
    Macenko stain normalization as described in:
    "A method for normalizing histology slides for quantitative analysis" by Macenko et al.
    
    This method separates the H&E stains using SVD in optical density space, 
    and specifically targets the H&E separation for histopathology images.
    """
    
    def __init__(self, beta=0.15, alpha=1):
        super().__init__()
        self.stain_matrix_target = None
        self.concentrations_target = None
        self.max_concentration_target = None
        self.beta = beta  # Percentile for stain vectors
        self.alpha = alpha  # Regularization parameter
    
    def get_stain_matrix(self, image, beta=0.15):
        """Get stain matrix for an image using the Macenko method"""
        # Convert to optical density
        image = image.astype(float) + 1.0  # Avoid log(0)
        optical_density = -np.log(image / 255.0)
        
        # Reshape to one column per pixel
        optical_density_reshaped = optical_density.reshape((-1, 3))
        
        # Remove pixels with low optical density
        od_threshold = 0.15
        optical_density_pixels = optical_density_reshaped[np.all(optical_density_reshaped > od_threshold, axis=1)]
        
        if optical_density_pixels.size == 0:
            logger.warning("No pixels meet optical density threshold, using all pixels")
            optical_density_pixels = optical_density_reshaped
        
        # Compute SVD
        try:
            cov = np.cov(optical_density_pixels.T)
            _, eigvecs = np.linalg.eigh(cov)
            
            # Check dimensionality - ensure we're getting 3 eigenvectors
            if eigvecs.shape[1] < 3:
                logger.warning(f"Insufficient eigenvectors: {eigvecs.shape}. Adding synthetic components.")
                # If we don't have enough eigenvectors, create synthetic ones 
                # This can happen with synthetic/uniform color areas
                missing_dims = 3 - eigvecs.shape[1]
                synthetic_vecs = np.random.rand(eigvecs.shape[0], missing_dims)
                eigvecs = np.column_stack([eigvecs, synthetic_vecs])
            
            # Take the two principal eigenvectors (columns 1 and 2, indices 0-indexed)
            eigvecs = eigvecs[:, [1, 2]]
            
            # Project data onto eigenvectors
            proj = optical_density_pixels @ eigvecs
            
            # Find the angle of each projected pixel
            phi = np.arctan2(proj[:, 1], proj[:, 0])
            
            # Find the angles that enclose the percentile of data
            min_phi = np.percentile(phi, beta)
            max_phi = np.percentile(phi, 100 - beta)
            
            # Calculate the corresponding vectors
            v1 = np.dot(eigvecs, np.array([np.cos(min_phi), np.sin(min_phi)]))
            v2 = np.dot(eigvecs, np.array([np.cos(max_phi), np.sin(max_phi)]))
            
            # Ensure correct ordering of stain vectors (H first, then E)
            if v1[0] < v2[0]:
                stain_matrix = np.column_stack([v1, v2])
            else:
                stain_matrix = np.column_stack([v2, v1])
                
            # Make sure the stain matrix has the right shape (3x2)
            if stain_matrix.shape != (3, 2):
                logger.warning(f"Invalid stain matrix shape: {stain_matrix.shape}, expected (3, 2)")
                # Create a fallback stain matrix for H&E
                stain_matrix = np.array([
                    [0.650, 0.072],  # Red
                    [0.704, 0.990],  # Green
                    [0.286, 0.105]   # Blue
                ])
                
            return stain_matrix
            
        except np.linalg.LinAlgError as e:
            logger.error(f"Linear algebra error in stain matrix computation: {e}")
            # Create a default stain matrix as fallback
            return np.array([
                [0.650, 0.072],  # Red
                [0.704, 0.990],  # Green
                [0.286, 0.105]   # Blue
            ])
    
    def fit(self, target_image):
        """Fit normalizer to a target image"""
        if isinstance(target_image, str):
            target_image = np.array(Image.open(target_image).convert('RGB'))
        elif isinstance(target_image, Image.Image):
            target_image = np.array(target_image)
        
        # Ensure proper shape
        if len(target_image.shape) != 3 or target_image.shape[2] != 3:
            raise ValueError("Target image must be RGB")
        
        # Get stain matrix
        self.stain_matrix_target = self.get_stain_matrix(target_image, self.beta)
        
        # Get concentrations
        optical_density = -np.log((target_image.astype(float) + 1.0) / 255.0)
        optical_density_reshaped = optical_density.reshape((-1, 3))
        
        # Calculate the concentrations using pseudo-inverse
        stain_matrix_pinv = np.linalg.pinv(self.stain_matrix_target)
        self.concentrations_target = optical_density_reshaped @ stain_matrix_pinv.T
        
        # Calculate max concentration for later scaling
        self.max_concentration_target = np.percentile(self.concentrations_target, 99, axis=0)
        
        return self
    
    def transform(self, source_image):
        """Transform source image to match target image staining"""
        if self.stain_matrix_target is None:
            raise ValueError("Normalizer has not been fit to a target image. Call fit() first.")
        
        if isinstance(source_image, str):
            source_image = np.array(Image.open(source_image).convert('RGB'))
        elif isinstance(source_image, Image.Image):
            source_image = np.array(source_image)
        
        # Ensure proper shape
        if len(source_image.shape) != 3 or source_image.shape[2] != 3:
            raise ValueError("Source image must be RGB")
        
        # Get source stain matrix
        stain_matrix_source = self.get_stain_matrix(source_image, self.beta)
        
        # Get source concentrations
        optical_density = -np.log((source_image.astype(float) + 1.0) / 255.0)
        optical_density_reshaped = optical_density.reshape((-1, 3))
        
        # Calculate the concentrations using pseudo-inverse
        stain_matrix_source_pinv = np.linalg.pinv(stain_matrix_source)
        concentrations_source = optical_density_reshaped @ stain_matrix_source_pinv.T
        
        # Calculate max concentration for scaling
        max_concentration_source = np.percentile(concentrations_source, 99, axis=0)
        
        # Scale concentrations to match target
        concentrations_source_normalized = concentrations_source * (self.max_concentration_target / max_concentration_source)
        
        # Recreate the optical density image
        optical_density_normalized = concentrations_source_normalized @ self.stain_matrix_target.T
        
        # Convert back to RGB
        rgb_normalized = 255.0 * np.exp(-optical_density_normalized)
        rgb_normalized = rgb_normalized.reshape(source_image.shape)
        rgb_normalized = np.clip(rgb_normalized, 0, 255).astype(np.uint8)
        
        return rgb_normalized


def visualize_normalization(original, normalized, target=None, figsize=(15, 5)):
    """Visualize the results of stain normalization"""
    if target is not None:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        titles = ['Original Image', 'Normalized Image', 'Target Image']
        images = [original, normalized, target]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        titles = ['Original Image', 'Normalized Image']
        images = [original, normalized]
    
    for i, (ax, img, title) in enumerate(zip(axes, images, titles)):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    return fig


# Helper function to perform normalization using a reference dataset
def normalize_histopathology_image(image, reference_image=None, method='macenko'):
    """
    Normalize a histopathology image using the specified method.
    
    Args:
        image: PIL.Image or numpy array or path to image file
        reference_image: Reference image for normalization (PIL.Image, numpy array, or path)
        method: 'macenko' or 'reinhard'
        
    Returns:
        Normalized image as numpy array
    """
    # Load image if path is provided
    if isinstance(image, str):
        image = np.array(Image.open(image).convert('RGB'))
    elif isinstance(image, Image.Image):
        image = np.array(image)
    
    # If no reference is provided, use a default H&E reference
    if reference_image is None:
        # Create a synthetic H&E reference (simplified approach)
        reference = np.zeros((100, 100, 3), dtype=np.uint8)
        # H&E typical colors (simplified)
        reference[:50, :, 0] = 150  # Hematoxylin - purple/blue (more R and B)
        reference[:50, :, 1] = 50
        reference[:50, :, 2] = 150
        reference[50:, :, 0] = 200  # Eosin - pink (more R and G)
        reference[50:, :, 1] = 100
        reference[50:, :, 2] = 100
    else:
        # Load reference image if path is provided
        if isinstance(reference_image, str):
            reference = np.array(Image.open(reference_image).convert('RGB'))
        elif isinstance(reference_image, Image.Image):
            reference = np.array(reference_image)
        else:
            reference = reference_image
    
    # Create and apply normalizer
    if method.lower() == 'macenko':
        normalizer = MacenkoNormalizer()
    elif method.lower() == 'reinhard':
        normalizer = ReinhardNormalizer()
    else:
        raise ValueError(f"Unknown normalization method: {method}. Use 'macenko' or 'reinhard'")
    
    # Apply normalization
    normalized = normalizer.fit_transform(image, reference)
    
    return normalized 