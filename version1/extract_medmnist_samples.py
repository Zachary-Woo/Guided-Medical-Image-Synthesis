import numpy as np
from PIL import Image
import os
import logging
import argparse
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Extract samples from MedMNIST dataset")
    parser.add_argument("--npz_file", type=str, default="./data/pathmnist_128.npz",
                        help="Path to the NPZ file")
    parser.add_argument("--output_dir", type=str, default="./data/pathmnist_samples",
                        help="Output directory for extracted samples")
    parser.add_argument("--num_samples", type=int, default=8,
                        help="Number of samples to extract")
    parser.add_argument("--image_key", type=str, default="train_images",
                        help="Key for the images in NPZ file")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")
    
    # Check if NPZ file exists
    if not os.path.exists(args.npz_file):
        # Try to find NPZ files in data directory
        data_dir = os.path.dirname(args.npz_file)
        npz_files = glob.glob(os.path.join(data_dir, "*.npz"))
        
        if npz_files:
            logger.warning(f"NPZ file {args.npz_file} not found.")
            logger.info(f"Found alternative NPZ files: {npz_files}")
            args.npz_file = npz_files[0]
            logger.info(f"Using {args.npz_file} instead.")
        else:
            logger.error(f"Error: NPZ file not found at {args.npz_file}")
            logger.error("Please ensure you have run the 'evaluate' command once to download the data.")
            return 1
    
    # Load NPZ file
    try:
        logger.info(f"Loading data from: {args.npz_file}")
        data = np.load(args.npz_file)
    except Exception as e:
        logger.error(f"Error loading NPZ file: {e}")
        return 1
    
    # Check if image key exists
    if args.image_key not in data:
        logger.error(f"Error: Cannot find '{args.image_key}' array within the NPZ file.")
        logger.info(f"Available keys: {list(data.keys())}")
        # Try to find a suitable alternative key
        image_keys = [k for k in data.keys() if 'image' in k.lower()]
        if image_keys:
            args.image_key = image_keys[0]
            logger.info(f"Using alternative key: {args.image_key}")
        else:
            return 1
    
    # Extract samples
    images = data[args.image_key]
    num_available = images.shape[0]
    logger.info(f"Found {num_available} images in '{args.image_key}'.")
    
    num_to_extract = min(args.num_samples, num_available)
    if num_to_extract < args.num_samples:
        logger.warning(f"Warning: Only found {num_available} images, extracting {num_to_extract}.")
    
    logger.info(f"Extracting {num_to_extract} samples to: {args.output_dir}")
    
    extracted_paths = []
    for i in range(num_to_extract):
        img_array = images[i]
        # Handle different image formats
        if img_array.ndim == 2:
            # Grayscale image, convert to RGB
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.ndim == 3 and img_array.shape[-1] not in [1, 3]:
            # First dimension might be channels, transpose
            img_array = np.transpose(img_array, (1, 2, 0))
            if img_array.shape[-1] == 1:
                img_array = np.tile(img_array, (1, 1, 3))
        
        img = Image.fromarray(img_array.astype(np.uint8))
        output_filename = f"sample_{i:04d}.png"
        output_path = os.path.join(args.output_dir, output_filename)
        try:
            img.save(output_path)
            extracted_paths.append(output_path)
            logger.info(f"Saved image {i} to {output_path}")
        except Exception as e:
            logger.error(f"Error saving image {i}: {e}")
    
    if extracted_paths:
        logger.info("\nExtraction complete.")
        logger.info("You can now use the following paths for the --conditioning_source_images argument:")
        # Print paths in a format easy to copy/paste
        paths_str = " ".join(extracted_paths)
        logger.info(paths_str)
        
        # Save paths to a file for easy use
        paths_file = os.path.join(args.output_dir, "sample_paths.txt")
        with open(paths_file, "w") as f:
            f.write(paths_str)
        logger.info(f"Paths also saved to: {paths_file}")
        
        return 0
    else:
        logger.error("No images were extracted successfully.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
