#!/usr/bin/env python
"""
MRI Visualization Script for comparing generated and real MRI images

This script creates a visualization panel showing the condition mask,
generated MRI, and real MRI slices from BraTS dataset for comparison.
"""

import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path
from PIL import Image
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare generated MRI images with real BraTS dataset images"
    )
    parser.add_argument("--generated_dir", type=str, required=True,
                      help="Directory containing generated MRI results")
    parser.add_argument("--brats_dir", type=str, required=True,
                      help="BraTS patient directory for comparison")
    parser.add_argument("--modality", type=str, default="t1",
                      choices=["t1", "t2", "flair", "t1ce"],
                      help="MRI modality to use from BraTS dataset")
    parser.add_argument("--slice_level", type=str, default=None,
                      choices=["superior", "mid-axial", "inferior", "ventricles", "basal-ganglia", "cerebellum"],
                      help="Axial slice level (if not provided, will be read from generation metadata)")
    parser.add_argument("--output_path", type=str, default=None,
                      help="Output path for visualization (default: generated_dir/comparison.png)")
    parser.add_argument("--show", action="store_true",
                      help="Display the visualization in addition to saving it")
    
    return parser.parse_args()

def load_nifti_volume(file_path):
    """Load a NIfTI file and return its data."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    nifti_img = nib.load(file_path)
    data = nifti_img.get_fdata()
    return data

def normalize_intensity(volume, percentiles=(1, 99)):
    """Normalize intensity values to the [0, 1] range."""
    if np.all(volume == 0):
        return volume
    
    # Compute percentiles
    p_low = np.percentile(volume[volume > 0], percentiles[0])
    p_high = np.percentile(volume, percentiles[1])
    
    # Clip and normalize
    volume = np.clip(volume, p_low, p_high)
    volume = (volume - p_low) / (p_high - p_low)
    volume = np.clip(volume, 0, 1)
    
    return volume

def select_slice_from_volume(volume, slice_level):
    """
    Select an appropriate slice from the volume based on the slice level.
    
    Args:
        volume: 3D numpy array of MRI data
        slice_level: String indicating the axial slice level
        
    Returns:
        2D numpy array of the selected slice
    """
    # Get volume dimensions
    depth = volume.shape[2]
    
    # Map slice level to relative position in volume
    slice_map = {
        "superior": 0.25,     # Upper 1/4 of the brain
        "mid-axial": 0.5,     # Middle of the brain
        "ventricles": 0.55,   # Slightly below middle (lateral ventricles)
        "basal-ganglia": 0.6, # Lower middle (basal ganglia)
        "inferior": 0.75,     # Lower part of the brain
        "cerebellum": 0.85    # Bottom section (cerebellum)
    }
    
    # Get relative position based on slice level
    relative_pos = slice_map.get(slice_level, 0.5)  # Default to middle if not found
    
    # Calculate slice index
    slice_idx = int(depth * relative_pos)
    
    # Ensure valid index
    slice_idx = max(0, min(slice_idx, depth - 1))
    
    # Extract the slice
    selected_slice = volume[:, :, slice_idx]
    
    return selected_slice

def load_and_process_brats_slice(brats_dir, modality, slice_level):
    """
    Load and process a slice from a BraTS patient volume.
    
    Args:
        brats_dir: Path to BraTS patient directory
        modality: MRI modality to use (t1, t2, flair, t1ce)
        slice_level: Axial slice level to extract
        
    Returns:
        Normalized 2D slice as numpy array
    """
    # Find the modality file
    brats_path = Path(brats_dir)
    modality_files = list(brats_path.glob(f"*{modality}.nii.gz"))
    
    if not modality_files:
        raise FileNotFoundError(f"No {modality} file found in {brats_dir}")
    
    modality_file = modality_files[0]
    logger.info(f"Using BraTS file: {modality_file}")
    
    # Load volume
    volume = load_nifti_volume(modality_file)
    
    # Select slice
    slice_data = select_slice_from_volume(volume, slice_level)
    
    # Normalize intensity
    slice_data = normalize_intensity(slice_data)
    
    return slice_data

def load_generated_data(generated_dir):
    """
    Load generated MRI data including mask, image, and metadata.
    
    Args:
        generated_dir: Path to directory with generation results
        
    Returns:
        Dictionary containing generated data
    """
    generated_dir = Path(generated_dir)
    
    # First, check if this is a parent directory containing timestamped subdirectories
    # Look for the most recent timestamped directory (e.g. mri_20250424-153908)
    timestamped_dirs = list(generated_dir.glob("mri_*"))
    if timestamped_dirs:
        # Sort by creation time, most recent first
        latest_dir = sorted(timestamped_dirs, key=lambda p: p.stat().st_ctime, reverse=True)[0]
        logger.info(f"Using latest generated directory: {latest_dir}")
        generated_dir = latest_dir
    
    # Load generated image
    generated_img_path = generated_dir / "generated_mri.png"
    if not generated_img_path.exists():
        raise FileNotFoundError(f"Generated image not found: {generated_img_path}")
    
    generated_img = np.array(Image.open(generated_img_path).convert("L"))
    
    # Load condition mask
    mask_path = generated_dir / "condition_mask.png"
    if not mask_path.exists():
        raise FileNotFoundError(f"Condition mask not found: {mask_path}")
    
    mask_img = np.array(Image.open(mask_path).convert("RGB"))
    
    # Load metadata
    metadata_path = generated_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    return {
        "generated_img": generated_img,
        "mask_img": mask_img,
        "metadata": metadata
    }

def create_visualization(generated_data, brats_slice, output_path, show=False):
    """
    Create a visualization panel comparing generated and real MRI images.
    
    Args:
        generated_data: Dictionary with generated image data
        brats_slice: Real BraTS slice for comparison
        output_path: Path to save visualization
        show: Whether to display the visualization
        
    Returns:
        Path to saved visualization
    """
    # Get data from dictionaries
    mask_img = generated_data["mask_img"]
    generated_img = generated_data["generated_img"]
    metadata = generated_data["metadata"]
    
    # Create figure
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot condition mask
    axs[0].imshow(mask_img)
    axs[0].set_title("Condition Mask")
    axs[0].axis("off")
    
    # Plot generated MRI
    axs[1].imshow(generated_img, cmap="gray")
    axs[1].set_title("Generated MRI")
    axs[1].axis("off")
    
    # Plot real BraTS MRI
    axs[2].imshow(brats_slice, cmap="gray")
    axs[2].set_title("Real BraTS MRI")
    axs[2].axis("off")
    
    # Add metadata as figure title
    prompt = metadata.get("prompt", "Unknown prompt")
    slice_level = metadata.get("slice_level", "Unknown slice level")
    fig.suptitle(f"Prompt: {prompt}\nSlice Level: {slice_level}", fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_path)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Visualization saved to {output_path}")
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()
    
    return output_path

def main():
    """Main function."""
    args = parse_args()
    
    # Load generated data
    generated_data = load_generated_data(args.generated_dir)
    
    # Get slice level from metadata if not provided
    slice_level = args.slice_level
    if slice_level is None:
        slice_level = generated_data["metadata"].get("slice_level", "mid-axial")
    
    # Load and process BraTS slice
    try:
        brats_slice = load_and_process_brats_slice(
            args.brats_dir, 
            args.modality,
            slice_level
        )
    except Exception as e:
        logger.error(f"Error loading BraTS data: {e}")
        logger.warning("Continuing with a blank slice for comparison")
        brats_slice = np.zeros((512, 512))
    
    # Set default output path if not provided
    output_path = args.output_path
    if output_path is None:
        output_path = Path(args.generated_dir) / "comparison.png"
    
    # Create visualization
    create_visualization(
        generated_data, 
        brats_slice, 
        output_path,
        show=args.show
    )
    
    print(f"Visualization saved to: {output_path}")

if __name__ == "__main__":
    main() 