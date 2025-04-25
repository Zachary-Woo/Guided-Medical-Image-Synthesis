#!/usr/bin/env python
"""
Stain Normalization Testing Script - Tests different normalization methods
on histopathology images and generates comparison visualizations.
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add parent directory to path for importing project modules
parent_dir = str(Path(__file__).resolve().parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import stain normalization utility
try:
    from version2.utils.stain_normalization import (
        normalize_histopathology_image,
        visualize_normalization
    )
except ImportError:
    logger.error("Stain normalization utilities not found. Please run from the project root directory.")
    sys.exit(1)

def get_next_output_dir(base_dir):
    """
    Create a sequentially numbered output directory.
    
    Args:
        base_dir (str or Path): Base output directory path
        
    Returns:
        Path: Next available numbered directory
    """
    base_path = Path(base_dir)
    base_parent = base_path.parent
    base_name = base_path.name
    
    # Find all existing numbered directories
    existing_dirs = []
    for item in base_parent.glob(f"{base_name}_*"):
        if item.is_dir():
            try:
                # Extract the number after the underscore
                num = int(item.name.split('_')[-1])
                existing_dirs.append(num)
            except ValueError:
                # Skip directories that don't end with a number
                continue
    
    # Determine the next number
    next_num = 1
    if existing_dirs:
        next_num = max(existing_dirs) + 1
    
    # Create the new directory path
    next_dir = base_parent / f"{base_name}_{next_num}"
    return next_dir

def parse_args():
    parser = argparse.ArgumentParser(description="Test Stain Normalization Methods")
    
    parser.add_argument("--input_image", type=str, required=True,
                        help="Path to input histopathology image")
    parser.add_argument("--reference_image", type=str, default=None,
                        help="Path to reference image for stain normalization (optional)")
    parser.add_argument("--output_dir", type=str, default="output/stain_normalization_test",
                        help="Directory to save output images (will be auto-incremented)")
    parser.add_argument("--method", type=str, default="macenko",
                        choices=["macenko", "reinhard"],
                        help="Stain normalization method to test")
    parser.add_argument("--compare_synthetic", type=str, default=None,
                        help="Path to synthetic image to compare with normalized image")
    parser.add_argument("--target_size", type=int, default=512,
                        help="Target size for resizing images")
    parser.add_argument("--save_visualization", action="store_true",
                        help="Save visualization of test results")
    
    return parser.parse_args()

def preprocess_image(image_path, target_size=512):
    """Preprocess image to target size"""
    image = Image.open(image_path).convert("RGB")
    if image.width != target_size or image.height != target_size:
        image = image.resize((target_size, target_size), Image.LANCZOS)
    return np.array(image)

def main():
    args = parse_args()
    
    # Create sequentially numbered output directory
    output_dir = get_next_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using output directory: {output_dir}")
    
    # Set up logging file
    log_file = output_dir / "test.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(file_handler)
    
    logger.info(f"Logging to {log_file}")
    
    logger.info(f"Loading input image: {args.input_image}")
    input_np = preprocess_image(args.input_image, args.target_size)
    
    reference_np = None
    if args.reference_image:
        logger.info(f"Loading reference image: {args.reference_image}")
        reference_np = preprocess_image(args.reference_image, args.target_size)
        
        # Save reference image
        reference_path = output_dir / "reference_image.png"
        Image.fromarray(reference_np).save(reference_path)
    
    # Create visualization with before/after comparison
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Display original image
    axes[0].imshow(input_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    method = args.method
    logger.info(f"Testing {method} normalization...")
    
    try:
        # Apply normalization
        normalized_np = normalize_histopathology_image(
            input_np,
            reference_image=reference_np,
            method=method
        )
        
        # Save normalized image
        normalized_path = output_dir / f"normalized_{method}.png"
        Image.fromarray(normalized_np).save(normalized_path)
        logger.info(f"Saved normalized image to {normalized_path}")
        
        # Create and save visualization
        vis_fig = visualize_normalization(
            input_np,
            normalized_np,
            target=reference_np
        )
        vis_path = output_dir / f"visualization_{method}.png"
        vis_fig.savefig(vis_path)
        plt.close(vis_fig)
        logger.info(f"Saved visualization to {vis_path}")
        
        # Add to comparison plot
        axes[1].imshow(normalized_np)
        axes[1].set_title(f"{method.capitalize()} Normalization")
        axes[1].axis("off")
        
    except Exception as e:
        logger.error(f"Error during {method} normalization: {e}")
        axes[1].text(0.5, 0.5, f"Error: {e}", 
                     ha="center", va="center", transform=axes[1].transAxes)
        axes[1].axis("off")
    
    # Add synthetic image comparison if provided
    if args.compare_synthetic:
        try:
            synthetic_np = preprocess_image(args.compare_synthetic, args.target_size)
            
            # Create additional figure with synthetic comparison
            fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
            
            axes2[0].imshow(input_np)
            axes2[0].set_title("Original Image")
            axes2[0].axis("off")
            
            # Use the normalized image
            axes2[1].imshow(normalized_np)
            axes2[1].set_title(f"Normalized ({method})")
            axes2[1].axis("off")
            
            axes2[2].imshow(synthetic_np)
            axes2[2].set_title("Synthetic Image")
            axes2[2].axis("off")
            
            synthetic_comparison_path = output_dir / "synthetic_comparison.png"
            fig2.savefig(synthetic_comparison_path)
            plt.close(fig2)
            logger.info(f"Saved synthetic comparison to {synthetic_comparison_path}")
        except Exception as e:
            logger.error(f"Error comparing with synthetic image: {e}")
    
    # Save comparison plot
    comparison_path = output_dir / "normalization_comparison.png"
    fig.tight_layout()
    fig.savefig(comparison_path)
    plt.close(fig)
    logger.info(f"Saved normalization comparison to {comparison_path}")
    
    logger.info("Testing complete! Results saved to: " + str(output_dir))
    return 0

if __name__ == "__main__":
    sys.exit(main()) 