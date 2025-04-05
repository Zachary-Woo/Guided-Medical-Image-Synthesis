#!/usr/bin/env python
"""
Enhanced evaluation script for version2 generated images.
This script evaluates the performance of downstream tasks using synthetic medical images
generated with stain normalization and LoRA techniques.
"""

import os
import sys
import torch
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset
import json

# Add parent directory to path for importing project modules
parent_dir = str(Path(__file__).resolve().parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import utility functions from version1
# try:
#     from version1.src.preprocessing.data_loader import MedicalImageDataset
#     from version1.src.preprocessing.transforms import MedicalImageTransforms
#     from version1.src.evaluation.downstream import train_and_evaluate_downstream
#     from version1.src.utils.visualization import plot_metrics, create_comparison_grid
#     from version1.src.utils.config import Config
# except ImportError:
#     logging.error("Required modules from version1 not found. Please ensure version1 is in the project root directory.")
#     sys.exit(1)

# External libraries
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
import medmnist
from medmnist import INFO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

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
    """
    Parse command line arguments for evaluation.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Evaluate downstream tasks using synthetic medical images with version2 enhancements")

    parser.add_argument("--synthetic_data_dir", type=str, required=True,
                        help="Directory with generated synthetic images (flat structure or ImageFolder compatible)")
    parser.add_argument("--config", type=str, default="configs/medmnist_evaluation.yaml",
                        help="Path to configuration file (used for defaults and dataset name)")
    parser.add_argument("--output_dir", type=str, default="output/evaluation_v2",
                        help="Output directory for evaluation results and logs (will be auto-incremented)")
    parser.add_argument("--task", type=str, default="classification",
                        choices=["segmentation", "classification"],
                        help="Downstream task to evaluate")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=20,
                        help="Number of epochs for downstream training")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Image size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for downstream model")
    parser.add_argument("--synthetic_mask_folder", type=str, default="masks",
                        help="Subfolder name containing masks within synthetic_data_dir (if needed for segmentation)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--medmnist_download_dir", type=str, default="./data",
                        help="Root directory to download MedMNIST data")
    parser.add_argument("--metadata_file", type=str, default=None,
                        help="Path to metadata.json file with generation parameters (for logging/analysis)")
    parser.add_argument("--no_normalize", action="store_true",
                        help="Skip normalization step for input images (e.g., if already normalized)")
    
    return parser.parse_args()

def load_synthetic_data(synthetic_data_dir, image_size, task, transform, mask_transform=None, synthetic_mask_folder=None):
    """
    Loads the synthetic dataset generated by the ControlNet pipeline.

    Args:
        synthetic_data_dir (str): Path to the directory containing generated synthetic images.
        image_size (int): Target size for image resizing (should match real data).
        task (str): The downstream task type ('classification' or 'segmentation').
        transform (Transform or tuple): The transform(s) to apply. 
        mask_transform (Transform, optional): Mask transform.
        synthetic_mask_folder (str, optional): Subfolder containing synthetic masks.

    Returns:
        Dataset or None: The loaded synthetic dataset.
    """
    if not os.path.isdir(synthetic_data_dir):
        logger.warning(f"Synthetic data directory not found: {synthetic_data_dir}. Returning None.")
        return None

    img_tfm = transform[0] if isinstance(transform, tuple) else transform
    mask_tfm = transform[1] if isinstance(transform, tuple) else mask_transform

    if task == "segmentation":
        if not synthetic_mask_folder:
            logger.error("Synthetic mask folder must be specified for segmentation task.")
            return None
        try:
            logger.info(f"Loading synthetic segmentation data from: {synthetic_data_dir} (masks: {synthetic_mask_folder})")
            # Commenting out version1 dependency
            # return MedicalImageDataset(
            #     data_dir=synthetic_data_dir,
            #     image_folder=".",
            #     mask_folder=synthetic_mask_folder,
            #     transform=img_tfm,
            #     mask_transform=mask_tfm
            # )
            logger.warning("Synthetic data loading for segmentation depends on version1, which is commented out.")
            return None
        except Exception as e:
            logger.error(f"Failed to load synthetic segmentation data: {e}")
            return None

    elif task == "classification":
        # Check for ImageFolder structure
        try:
            subdirs = [d for d in os.listdir(synthetic_data_dir) if os.path.isdir(os.path.join(synthetic_data_dir, d))]
            is_image_folder = len(subdirs) > 0 and all(f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))
                                                   for subdir in subdirs
                                                   for f in os.listdir(os.path.join(synthetic_data_dir, subdir)))
        except Exception:
            is_image_folder = False

        if is_image_folder:
            try:
                logger.info(f"Loading synthetic classification data using ImageFolder from: {synthetic_data_dir}")
                return ImageFolder(root=synthetic_data_dir, transform=img_tfm)
            except Exception as e:
                logger.error(f"Failed to load synthetic data using ImageFolder: {e}")
                return None
        else:
            try:
                logger.info(f"Loading synthetic classification data from flat directory: {synthetic_data_dir}")
                # Commenting out version1 dependency
                # return MedicalImageDataset(
                #     data_dir=synthetic_data_dir,
                #     image_folder=".",
                #     mask_folder=None,
                #     transform=img_tfm,
                #     mask_transform=None
                # )
                logger.warning("Synthetic data loading for flat classification depends on version1, which is commented out.")
                return None
            except Exception as e:
                logger.error(f"Could not load synthetic classification data: {e}")
                return None
    else:
        logger.error(f"Unknown task type '{task}' for loading synthetic data.")
        return None

def extract_generation_metadata(metadata_file):
    """
    Extract metadata from a generation run.
    
    Args:
        metadata_file (str): Path to metadata.json file
        
    Returns:
        dict: Generation metadata
    """
    if not metadata_file or not os.path.exists(metadata_file):
        return {}
    
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        return metadata
    except Exception as e:
        logger.error(f"Error loading metadata file: {e}")
        return {}

def print_evaluation_summary(real_metrics, augmented_metrics, test_metric_key, generation_metadata):
    """
    Print a summary of the evaluation results.
    
    Args:
        real_metrics (dict): Metrics from real-only training
        augmented_metrics (dict): Metrics from augmented training
        test_metric_key (str): Key for test metric
        generation_metadata (dict): Generation metadata
    """
    print("\n" + "="*80)
    print(" EVALUATION SUMMARY ".center(80, "="))
    print("="*80)
    
    if generation_metadata:
        print("\nGeneration Parameters:")
        print(f"  Base Model:      {generation_metadata.get('base_model', 'Unknown')}")
        print(f"  ControlNet:      {generation_metadata.get('controlnet_model', 'Unknown')}")
        print(f"  LoRA Model:      {generation_metadata.get('lora_model', 'None')}")
        print(f"  Stain Norm:      {generation_metadata.get('stain_normalization', 'None')}")
        print(f"  Prompt:          {generation_metadata.get('prompt', 'Unknown')[:60]}...")
    
    print("\nTest Performance:")
    if real_metrics and test_metric_key in real_metrics:
        real_perf = real_metrics[test_metric_key]
        print(f"  Real Data Only:  {real_perf:.4f}")
    else:
        print("  Real Data Only:  Not available")
    
    if augmented_metrics and test_metric_key in augmented_metrics:
        aug_perf = augmented_metrics[test_metric_key]
        print(f"  With Synthetic:  {aug_perf:.4f}")
        
        if real_metrics and test_metric_key in real_metrics:
            improvement = aug_perf - real_perf
            percent = improvement / real_perf * 100
            print(f"  Improvement:     {improvement:.4f} ({'+' if improvement > 0 else ''}{percent:.2f}%)")
    else:
        print("  With Synthetic:  Not available")
    
    print("="*80)

def main():
    """
    Main evaluation function.
    """
    args = parse_args()
    
    # Create sequentially numbered output directory
    output_dir = get_next_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Update logging to include output file
    log_file = output_dir / "evaluate.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(file_handler)
    logger.info(f"Using output directory: {output_dir}")
    logger.info(f"Logging to {log_file}")
    
    # Load generation metadata if available
    metadata_file = args.metadata_file
    if not metadata_file and args.synthetic_data_dir:
        # Try to find metadata.json in the synthetic data directory
        potential_metadata = os.path.join(args.synthetic_data_dir, "metadata.json")
        if os.path.exists(potential_metadata):
            metadata_file = potential_metadata
            logger.info(f"Found metadata file: {metadata_file}")
    
    generation_metadata = extract_generation_metadata(metadata_file)
    if generation_metadata:
        logger.info(f"Loaded generation metadata with parameters: {list(generation_metadata.keys())}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Load Real Data using MedMNIST API --- #
    dataset_name = args.config.split('/')[-1].split('.')[0].replace('_config', '').replace('_evaluation', '')
    if not dataset_name.endswith('mnist'):
        logger.warning(f"Dataset name '{dataset_name}' doesn't look like a MedMNIST dataset. Assuming PathMNIST.")
        dataset_name = "pathmnist"
    
    logger.info(f"Loading real MedMNIST data: {dataset_name} (size={args.image_size})")
    try:
        DataClass = getattr(medmnist, dataset_name.capitalize())
    except AttributeError:
        logger.error(f"Invalid MedMNIST dataset name: {dataset_name}")
        logger.info("Available datasets: pathmnist, octmnist, pneumoniamnist, breastmnist, bloodmnist, dermamnist, etc.")
        sys.exit(1)

    info = INFO[dataset_name.lower()]
    n_channels = info['n_channels']
    n_classes = len(info['label'])
    medmnist_task = info['task']
    logger.info(f"MedMNIST info: channels={n_channels}, classes={n_classes}, task={medmnist_task}")

    # Define transforms
    norm_mean = [0.5] * n_channels if not args.no_normalize else [0.0] * n_channels
    norm_std = [0.5] * n_channels if not args.no_normalize else [1.0] * n_channels
    
    img_tfm = T.Compose([
        T.Resize((args.image_size, args.image_size)),
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std) if not args.no_normalize else lambda x: x
    ])

    # For segmentation tasks
    mask_tfm = T.Compose([
        T.Resize((args.image_size, args.image_size), interpolation=T.InterpolationMode.NEAREST),
        T.ToTensor(),
        lambda x: (x > 0.5).float()
    ]) if args.task == "segmentation" else None

    # Load appropriate datasets
    if args.task == "segmentation":
        real_train_dataset = DataClass(split='train', transform=img_tfm, target_transform=mask_tfm, download=True, root=args.medmnist_download_dir)
        real_val_dataset = DataClass(split='val', transform=img_tfm, target_transform=mask_tfm, download=True, root=args.medmnist_download_dir)
        real_test_dataset = DataClass(split='test', transform=img_tfm, target_transform=mask_tfm, download=True, root=args.medmnist_download_dir)
        data_transform = (img_tfm, mask_tfm)
    elif args.task == "classification":
        real_train_dataset = DataClass(split='train', transform=img_tfm, download=True, root=args.medmnist_download_dir)
        real_val_dataset = DataClass(split='val', transform=img_tfm, download=True, root=args.medmnist_download_dir)
        real_test_dataset = DataClass(split='test', transform=img_tfm, download=True, root=args.medmnist_download_dir)
        data_transform = img_tfm
    else:
        logger.error(f"Task '{args.task}' not compatible with MedMNIST loading logic.")
        sys.exit(1)

    logger.info(f"Loaded real datasets: Train={len(real_train_dataset)}, Val={len(real_val_dataset)}, Test={len(real_test_dataset)}")

    # --- Load Synthetic Data --- #
    logger.info(f"Loading synthetic data from: {args.synthetic_data_dir}")
    synthetic_dataset = load_synthetic_data(
        synthetic_data_dir=args.synthetic_data_dir,
        image_size=args.image_size,
        task=args.task,
        transform=data_transform,
        mask_transform=mask_tfm if args.task == "segmentation" else None,
        synthetic_mask_folder=args.synthetic_mask_folder
    )

    # --- Create Augmented Dataset and DataLoaders --- #
    if synthetic_dataset and len(synthetic_dataset) > 0:
        logger.info(f"Synthetic dataset size: {len(synthetic_dataset)}")
        augmented_train_dataset = ConcatDataset([real_train_dataset, synthetic_dataset])
        logger.info(f"Augmented training dataset size: {len(augmented_train_dataset)}")
    else:
        logger.warning("Synthetic dataset not loaded or empty. Augmentation step skipped.")
        augmented_train_dataset = real_train_dataset

    # Create dataloaders
    num_workers = min(os.cpu_count() or 4, 4)
    real_train_dataloader = DataLoader(
        real_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    augmented_train_dataloader = DataLoader(
        augmented_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_dataloader = DataLoader(
        real_val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_dataloader = DataLoader(
        real_test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # --- Run Comparison --- #
    all_real_metrics = None
    all_augmented_metrics = None

    # 1. Train and Evaluate on Real Data Only
    logger.info(f"=== Training downstream model on REAL data only ({args.task}) ===")
    try:
        # Commenting out version1 dependency
        # real_metrics_run = train_and_evaluate_downstream(
        #     train_loader=real_train_dataloader,
        #     val_loader=val_dataloader,
        #     test_loader=test_dataloader,
        #     task=args.task,
        #     num_epochs=args.num_epochs,
        #     device=device,
        #     learning_rate=args.learning_rate,
        #     model_save_path=os.path.join(output_dir, "downstream_model_real_only.pth"),
        #     n_classes=n_classes if args.task == "classification" else None,
        #     n_channels=n_channels if args.task == "segmentation" else None
        # )
        # all_real_metrics = real_metrics_run
        logger.warning("Downstream training on real data depends on version1, which is commented out.")
        logger.info("Finished training on real data.")
    except Exception as e:
        logger.error(f"Error during training/evaluation on real data: {e}")

    # 2. Train and Evaluate on Augmented Data (if synthetic data loaded and valid)
    if synthetic_dataset and len(synthetic_dataset) > 0 and len(augmented_train_dataset) > len(real_train_dataset):
        logger.info(f"=== Training downstream model on AUGMENTED data ({args.task}) ===")
        try:
            # Commenting out version1 dependency
            # augmented_metrics_run = train_and_evaluate_downstream(
            #     train_loader=augmented_train_dataloader,
            #     val_loader=val_dataloader,
            #     test_loader=test_dataloader,
            #     task=args.task,
            #     num_epochs=args.num_epochs,
            #     device=device,
            #     learning_rate=args.learning_rate,
            #     model_save_path=os.path.join(output_dir, "downstream_model_augmented.pth"),
            #     n_classes=n_classes if args.task == "classification" else None,
            #     n_channels=n_channels if args.task == "segmentation" else None
            # )
            # all_augmented_metrics = augmented_metrics_run
            logger.warning("Downstream training on augmented data depends on version1, which is commented out.")
            logger.info("Finished training on augmented data.")
        except Exception as e:
            logger.error(f"Error during training/evaluation on augmented data: {e}")
    else:
        logger.info("Skipping training on augmented data as synthetic dataset was not loaded or was empty.")

    # --- Plotting and Saving Results --- #
    logger.info("Plotting metrics...")

    # Commenting out plotting as it depends on metrics from version1
    # if all_real_metrics:
    #     metric_key_train = "train_loss"
    #     metric_key_val = "val_dice" if args.task == "segmentation" else "val_acc"
    #     ...
    #     logger.info(f"Saved metrics to {metrics_path}")
    logger.warning("Metrics plotting and saving skipped as downstream training from version1 is commented out.")

    logger.info(f"Evaluation complete. Results saved to {output_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 