#!/usr/bin/env python
"""
Prepare histopathology datasets for LoRA fine-tuning.
This script downloads sample datasets, preprocesses images with stain normalization,
and generates configuration for training.
"""

import sys
import logging
import argparse
import json
import numpy as np
from pathlib import Path
import requests
import zipfile
import tarfile
from tqdm import tqdm
from PIL import Image
import yaml

# Add parent directory to path for importing project modules
parent_dir = str(Path(__file__).resolve().parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import stain normalization utility
try:
    from version2.utils.stain_normalization import (
        normalize_histopathology_image,
    )
except ImportError:
    print("Stain normalization utilities not found. Please run from the project root directory.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

# Dataset URLs
DATASET_URLS = {
    "kather_texture": "https://zenodo.org/record/53169/files/Kather_texture_2016_image_tiles_5000.zip",
    "breakhis": "https://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz",
    "pcam": "https://drive.google.com/uc?id=1Ka0XfEMiwgCYPdTI-vv6eUElOBnKFKQ2",
    "camelyon": "https://drive.google.com/uc?id=1bIXGIX8SQB8pddkHvVj2V22NOl4OkGk0"
}

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare histopathology datasets for LoRA fine-tuning")
    
    parser.add_argument("--dataset", type=str, default="kather_texture",
                        choices=["kather_texture", "breakhis", "pcam", "camelyon", "local"],
                        help="Dataset to prepare")
    parser.add_argument("--local_dataset_path", type=str, default=None,
                        help="Path to local dataset (if 'local' is selected)")
    parser.add_argument("--output_dir", type=str, default="version2/data/processed",
                        help="Output directory for processed dataset")
    parser.add_argument("--config_output", type=str, default="version2/configs/lora_config.yaml",
                        help="Path to output configuration file")
    parser.add_argument("--stain_norm", type=str, default="macenko",
                        choices=["macenko", "reinhard", "none"],
                        help="Stain normalization method")
    parser.add_argument("--reference_image", type=str, default=None,
                        help="Path to reference image for stain normalization")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of samples to prepare (subset of dataset)")
    parser.add_argument("--target_size", type=int, default=512,
                        help="Target image size")
    parser.add_argument("--train_val_split", type=float, default=0.9,
                        help="Train/validation split ratio")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    
    return parser.parse_args()

def download_dataset(dataset_name, output_dir):
    """
    Download a dataset if it doesn't exist locally.
    
    Args:
        dataset_name: Name of the dataset
        output_dir: Directory to save the dataset
        
    Returns:
        Path to downloaded dataset
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if dataset_name not in DATASET_URLS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    url = DATASET_URLS[dataset_name]
    download_path = output_dir / f"{dataset_name}_raw.zip"
    
    # Check if the dataset is already downloaded
    if download_path.exists():
        logger.info(f"Dataset already downloaded to {download_path}")
        return download_path
    
    # Handle special datasets that require manual download
    if dataset_name in ["pcam", "camelyon", "breakhis"]:
        logger.warning(f"{dataset_name} requires manual download. Please download from:")
        logger.warning(f"  {url}")
        logger.warning(f"and place the file at {download_path}")
        return None
    
    # Download the dataset
    logger.info(f"Downloading {dataset_name} dataset from {url}")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    
    with open(download_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Downloading {dataset_name}") as pbar:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    logger.info(f"Downloaded {dataset_name} dataset to {download_path}")
    return download_path

def extract_dataset(download_path, output_dir):
    """
    Extract a downloaded dataset.
    
    Args:
        download_path: Path to downloaded dataset
        output_dir: Directory to extract the dataset
        
    Returns:
        Path to extracted dataset
    """
    # Create output directory
    output_dir = Path(output_dir)
    extract_dir = output_dir / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if the dataset is already extracted
    if any(extract_dir.iterdir()):
        logger.info(f"Dataset already extracted to {extract_dir}")
        return extract_dir
    
    # Extract the dataset
    logger.info(f"Extracting dataset from {download_path}")
    
    if str(download_path).endswith('.zip'):
        with zipfile.ZipFile(download_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    elif str(download_path).endswith('.tar.gz'):
        with tarfile.open(download_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_dir)
    else:
        logger.error(f"Unsupported archive format: {download_path}")
        return None
    
    logger.info(f"Extracted dataset to {extract_dir}")
    return extract_dir

def find_image_files(directory, extensions=['.png', '.jpg', '.jpeg', '.tif', '.tiff']):
    """
    Recursively find all image files in a directory.
    
    Args:
        directory: Directory to search
        extensions: List of file extensions to include
        
    Returns:
        List of image file paths
    """
    image_files = []
    directory = Path(directory)
    
    for ext in extensions:
        image_files.extend(list(directory.glob(f"**/*{ext}")))
        image_files.extend(list(directory.glob(f"**/*{ext.upper()}")))
    
    return sorted(image_files)

def preprocess_image(image_input, output_path, target_size, stain_norm="macenko", reference_image=None, is_numpy_array=False):
    """
    Preprocess an image for LoRA fine-tuning.
    
    Args:
        image_input: Path to the image or a NumPy array
        output_path: Path to save the preprocessed image
        target_size: Target image size
        stain_norm: Stain normalization method
        reference_image: Reference image for stain normalization
        is_numpy_array: Flag indicating if image_input is a NumPy array
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load the image
        if is_numpy_array:
            if image_input.ndim == 2: # Handle grayscale images if necessary
                image = Image.fromarray(image_input).convert("RGB")
            elif image_input.ndim == 3 and image_input.shape[2] == 1:
                image = Image.fromarray(image_input.squeeze()).convert("RGB")
            elif image_input.ndim == 3 and image_input.shape[2] == 3:
                image = Image.fromarray(image_input)
            else:
                 raise ValueError(f"Unsupported NumPy array shape: {image_input.shape}")
        else:
            image = Image.open(image_input).convert("RGB")
        
        # Resize the image
        if image.width != target_size or image.height != target_size:
            image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
        
        # Apply stain normalization if requested
        if stain_norm.lower() != "none":
            image_np = np.array(image)
            normalized_np = normalize_histopathology_image(
                image_np,
                reference_image=reference_image,
                method=stain_norm
            )
            image = Image.fromarray(normalized_np)
        
        # Save the preprocessed image
        output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure output dir exists
        image.save(output_path)
        return True
    
    except Exception as e:
        input_source = "NumPy array" if is_numpy_array else str(image_input)
        logger.error(f"Error preprocessing image from {input_source}: {e}", exc_info=True)
        return False

def create_training_metadata(image_files, output_dir, caption="Histopathology slide showing tissue sample with cellular details, H&E stain"):
    """
    Create metadata for training.
    
    Args:
        image_files: List of image file paths
        output_dir: Directory to save the metadata
        caption: Caption to use for all images
        
    Returns:
        Path to metadata file
    """
    # Create metadata
    metadata = []
    for image_file in image_files:
        image_path = str(image_file.relative_to(output_dir))
        metadata.append({
            "file_name": image_path,
            "text": caption
        })
    
    # Save metadata
    metadata_path = Path(output_dir) / "metadata.jsonl"
    with open(metadata_path, 'w') as f:
        for item in metadata:
            f.write(json.dumps(item) + '\n')
    
    logger.info(f"Created metadata file with {len(metadata)} entries at {metadata_path}")
    return metadata_path

def create_class_specific_metadata(image_files, class_names, output_dir):
    """
    Create metadata with class-specific captions.
    
    Args:
        image_files: List of image file paths
        class_names: Dictionary mapping class folders to captions
        output_dir: Directory to save the metadata
        
    Returns:
        Path to metadata file
    """
    # Create metadata
    metadata = []
    for image_file in image_files:
        image_path = str(image_file.relative_to(output_dir))
        
        # Determine class from path
        for class_name, caption in class_names.items():
            if class_name in str(image_file):
                metadata.append({
                    "file_name": image_path,
                    "text": caption
                })
                break
        else:
            # Default caption if no class matches
            metadata.append({
                "file_name": image_path,
                "text": "Histopathology slide showing tissue sample with cellular details, H&E stain"
            })
    
    # Save metadata
    metadata_path = Path(output_dir) / "metadata.jsonl"
    with open(metadata_path, 'w') as f:
        for item in metadata:
            f.write(json.dumps(item) + '\n')
    
    logger.info(f"Created class-specific metadata file with {len(metadata)} entries at {metadata_path}")
    return metadata_path

def create_lora_config(output_dir, args):
    """
    Create configuration for LoRA fine-tuning.
    
    Args:
        output_dir: Directory containing processed dataset
        args: Command line arguments
        
    Returns:
        Path to configuration file
    """
    # Create config
    config = {
        "base_model_id": "runwayml/stable-diffusion-v1-5",
        "dataset": {
            "train_data_dir": str(Path(output_dir) / "train"),
            "validation_data_dir": str(Path(output_dir) / "validation")
        },
        "training": {
            "output_dir": "version2/models/lora_histopathology",
            "learning_rate": 1e-4,
            "train_batch_size": 1,
            "num_train_epochs": 100,
            "max_train_steps": 2000,
            "checkpointing_steps": 500,
            "validation_steps": 100,
            "gradient_accumulation_steps": 4,
            "seed": args.seed
        },
        "lora": {
            "r": 16,
            "alpha": 32,
            "target_modules": ["q_proj", "k_proj", "v_proj", "out_proj"],
            "dropout": 0.1
        },
        "preprocessing": {
            "resolution": args.target_size,
            "center_crop": True,
            "random_flip": True
        }
    }
    
    # Save config
    config_path = Path(args.config_output)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Created LoRA configuration at {config_path}")
    return config_path

def process_kather_texture_dataset(extract_dir, output_dir, args):
    """
    Process the Kather texture dataset.
    
    Args:
        extract_dir: Directory containing extracted dataset
        output_dir: Directory to save processed dataset
        args: Command line arguments
        
    Returns:
        Path to processed dataset
    """
    # Find all image files
    image_files = find_image_files(extract_dir)
    logger.info(f"Found {len(image_files)} images in {extract_dir}")
    
    # Sample images if needed
    np.random.seed(args.seed)
    if len(image_files) > args.num_samples:
        image_files = np.random.choice(image_files, args.num_samples, replace=False).tolist()
        logger.info(f"Sampled {len(image_files)} images for processing")
    
    # Create train/validation split
    np.random.shuffle(image_files)
    split_idx = int(len(image_files) * args.train_val_split)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    logger.info(f"Split dataset into {len(train_files)} training and {len(val_files)} validation images")
    
    # Load reference image for stain normalization if provided
    reference_image = None
    if args.reference_image:
        try:
            reference_image = np.array(Image.open(args.reference_image).convert('RGB'))
            logger.info(f"Loaded reference image for stain normalization: {args.reference_image}")
        except Exception as e:
            logger.error(f"Failed to load reference image: {e}")
            logger.warning("Proceeding without stain normalization reference")
    
    # Create output directories
    processed_dir = Path(output_dir)
    train_dir = processed_dir / "train"
    val_dir = processed_dir / "validation"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Process training images
    logger.info("Processing training images...")
    train_processed = []
    for image_file in tqdm(train_files, desc="Processing training images"):
        output_path = train_dir / image_file.name
        if preprocess_image(
            image_file, output_path, args.target_size,
            stain_norm=args.stain_norm, reference_image=reference_image
        ):
            train_processed.append(output_path)
    
    # Process validation images
    logger.info("Processing validation images...")
    val_processed = []
    for image_file in tqdm(val_files, desc="Processing validation images"):
        output_path = val_dir / image_file.name
        if preprocess_image(
            image_file, output_path, args.target_size,
            stain_norm=args.stain_norm, reference_image=reference_image
        ):
            val_processed.append(output_path)
    
    logger.info(f"Processed {len(train_processed)} training and {len(val_processed)} validation images")
    
    # Create class-specific captions based on folder structure
    class_names = {
        "01_TUMOR": "Histopathology slide showing colorectal adenocarcinoma epithelial tissue, H&E stain, tumor cells visible",
        "02_STROMA": "Histopathology slide showing colorectal stroma tissue with fibrous matrix, H&E stain",
        "03_COMPLEX": "Histopathology slide showing complex colorectal tissue structure with mixed cell types, H&E stain",
        "04_LYMPHO": "Histopathology slide showing lymphocyte aggregates in colorectal tissue, H&E stain, immune cells",
        "05_DEBRIS": "Histopathology slide showing debris and necrotic tissue in colorectal sample, H&E stain",
        "06_MUCOSA": "Histopathology slide showing normal colorectal mucosa tissue, H&E stain, healthy epithelial cells",
        "07_ADIPOSE": "Histopathology slide showing adipose tissue in colorectal sample, H&E stain, fat cells",
        "08_EMPTY": "Histopathology slide showing empty or background area in colorectal sample, H&E stain"
    }
    
    # Create metadata
    create_class_specific_metadata(train_processed, class_names, processed_dir)
    
    return processed_dir

def process_local_dataset(local_path_str, output_dir, args):
    """
    Process a local dataset, supporting both directories and .npz files.
    
    Args:
        local_path_str: Path to local dataset (directory or .npz file)
        output_dir: Directory to save processed dataset
        args: Command line arguments
        
    Returns:
        Path to processed dataset
    """
    if not local_path_str:
        logger.error("Local dataset path not provided")
        return None
    
    local_path = Path(local_path_str)
    if not local_path.exists():
        logger.error(f"Local dataset path does not exist: {local_path}")
        return None

    image_inputs = []
    is_numpy_input = False

    if local_path.is_file() and local_path.suffix == '.npz':
        logger.info(f"Loading images from .npz file: {local_path}")
        try:
            with np.load(local_path) as data:
                image_key = 'train_images' 
                if image_key not in data:
                    available_keys = list(data.keys())
                    logger.error(f"Could not find key '{image_key}' in {local_path}. Available keys: {available_keys}")
                    # Attempt to find a suitable key heuristically
                    potential_keys = [k for k in available_keys if 'image' in k.lower() or 'img' in k.lower()]
                    if potential_keys:
                         image_key = potential_keys[0]
                         logger.warning(f"Attempting to use key '{image_key}' instead.")
                    else:
                         logger.error("Could not identify a suitable image array key. Please check the .npz file structure.")
                         return None

                images_np = data[image_key]
                logger.info(f"Found {len(images_np)} images in .npz file under key '{image_key}' with shape {images_np.shape}")
                image_inputs = list(images_np) # Convert to list of arrays for iteration
                is_numpy_input = True
        except Exception as e:
            logger.error(f"Failed to load or process .npz file {local_path}: {e}", exc_info=True)
            return None

    elif local_path.is_dir():
        logger.info(f"Searching for image files in directory: {local_path}")
        image_inputs = find_image_files(local_path)
        logger.info(f"Found {len(image_inputs)} image files in directory")
        is_numpy_input = False
    else:
        logger.error(f"Unsupported local dataset path type: {local_path}. Must be a directory or .npz file.")
        return None
    
    if not image_inputs:
        logger.warning("No images found or loaded from the specified path.")
        return None

    # Sample images if needed
    np.random.seed(args.seed)
    if len(image_inputs) > args.num_samples:
        indices = np.random.choice(len(image_inputs), args.num_samples, replace=False)
        image_inputs = [image_inputs[i] for i in indices]
        logger.info(f"Sampled {len(image_inputs)} images for processing")
    
    # Create train/validation split indices
    num_images = len(image_inputs)
    indices = list(range(num_images))
    np.random.shuffle(indices)
    split_idx = int(num_images * args.train_val_split)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    logger.info(f"Split dataset into {len(train_indices)} training and {len(val_indices)} validation indices")
    
    # Load reference image for stain normalization if provided
    reference_image = None
    if args.reference_image:
        try:
            reference_image = np.array(Image.open(args.reference_image).convert('RGB'))
            logger.info(f"Loaded reference image for stain normalization: {args.reference_image}")
        except Exception as e:
            logger.error(f"Failed to load reference image: {e}")
            logger.warning("Proceeding without stain normalization reference")
            reference_image = None
    elif args.stain_norm.lower() != "none":
        logger.warning("No reference image provided via --reference_image. Using synthetic reference for normalization.")

    # Create output directories
    processed_dir = Path(output_dir)
    train_dir = processed_dir / "train"
    val_dir = processed_dir / "validation"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    train_processed_paths = []
    val_processed_paths = []

    # Process training images
    logger.info("Processing training images...")
    for i in tqdm(train_indices, desc="Processing training images"):
        img_input = image_inputs[i]
        # Generate a filename based on index if from NPZ, otherwise use original name
        filename = f"image_{i:06d}.png" if is_numpy_input else Path(img_input).name
        output_path = train_dir / filename
        if preprocess_image(
            img_input, output_path, args.target_size,
            stain_norm=args.stain_norm, reference_image=reference_image, is_numpy_array=is_numpy_input
        ):
            train_processed_paths.append(output_path)
    
    # Process validation images
    logger.info("Processing validation images...")
    for i in tqdm(val_indices, desc="Processing validation images"):
        img_input = image_inputs[i]
        filename = f"image_{i:06d}.png" if is_numpy_input else Path(img_input).name
        output_path = val_dir / filename
        if preprocess_image(
            img_input, output_path, args.target_size,
            stain_norm=args.stain_norm, reference_image=reference_image, is_numpy_array=is_numpy_input
        ):
            val_processed_paths.append(output_path)
    
    logger.info(f"Processed {len(train_processed_paths)} training and {len(val_processed_paths)} validation images")
    
    # Create metadata using the processed paths
    create_training_metadata(train_processed_paths + val_processed_paths, processed_dir)
    
    return processed_dir

def main():
    args = parse_args()
    
    # Set up logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Process dataset
    if args.dataset == "local":
        if not args.local_dataset_path:
            logger.error("Local dataset path not provided")
            return 1
        
        logger.info(f"Processing local dataset at {args.local_dataset_path}")
        processed_dir = process_local_dataset(args.local_dataset_path, args.output_dir, args)
    else:
        # Download dataset
        download_path = download_dataset(args.dataset, args.output_dir)
        if download_path is None:
            logger.error("Failed to download dataset")
            return 1
        
        # Extract dataset
        extract_dir = extract_dataset(download_path, args.output_dir)
        if extract_dir is None:
            logger.error("Failed to extract dataset")
            return 1
        
        # Process dataset
        if args.dataset == "kather_texture":
            processed_dir = process_kather_texture_dataset(extract_dir, args.output_dir, args)
        else:
            logger.error(f"Processing for dataset {args.dataset} not implemented yet")
            return 1
    
    if processed_dir is None:
        logger.error("Failed to process dataset")
        return 1
    
    # Create LoRA configuration
    config_path = create_lora_config(processed_dir, args)
    
    logger.info(f"Dataset preparation complete. Processed dataset saved to {processed_dir}")
    logger.info(f"LoRA configuration saved to {config_path}")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 