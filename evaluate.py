#!/usr/bin/env python
"""
Evaluation script for downstream tasks using synthetic medical images.
"""

import os
import sys
import torch
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset, Subset
from src.preprocessing.data_loader import MedicalImageDataset
from src.preprocessing.transforms import MedicalImageTransforms
from src.evaluation.downstream import compare_performance
# from src.evaluation.metrics import evaluate_model_performance # Not used directly in main
from src.utils.visualization import plot_metrics, create_comparison_grid
from torchvision.datasets import ImageFolder
from torchvision import transforms as T # Use T alias
from sklearn.model_selection import train_test_split
from src.utils.config import Config


def parse_args():
    """
    Parse command line arguments for evaluation.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Evaluate downstream tasks using synthetic medical images")

    parser.add_argument("--real_data_dir", type=str, required=True, help="Directory with real data")
    parser.add_argument("--synthetic_data_dir", type=str, required=True,
                        help="Directory with generated synthetic images (flat structure or ImageFolder compatible)")
    parser.add_argument("--config", type=str, default="configs/medmnist_canny_demo.yaml", # Added config arg
                        help="Path to configuration file (used for defaults)")
    parser.add_argument("--output_dir", type=str, # Removed default, will get from config or set default later
                        help="Output directory for evaluation results and logs (overrides config)")
    parser.add_argument("--task", type=str, default="classification", # Default to classification (simpler)
                        choices=["segmentation", "classification"],
                        help="Downstream task to evaluate")
    parser.add_argument("--batch_size", type=int, help="Batch size (overrides config)")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs for downstream training (overrides config)")
    parser.add_argument("--image_size", type=int, help="Image size (overrides config)")
    parser.add_argument("--learning_rate", type=float, help="Learning rate for downstream model (overrides config)")
    parser.add_argument("--mask_folder", type=str, default="masks",
                        help="Subfolder name containing masks within real_data_dir (for segmentation)")
    parser.add_argument("--synthetic_mask_folder", type=str, default="masks",
                        help="Subfolder name containing masks within synthetic_data_dir (if needed for segmentation)")
    parser.add_argument("--val_split", type=float, help="Fraction of real data to use for validation set during downstream training.")
    parser.add_argument("--test_split", type=float, help="Fraction of real data to use for the final test set.")
    parser.add_argument("--seed", type=int, help="Random seed")

    return parser.parse_args()


def load_and_split_real_data(real_data_dir, image_size, task, mask_folder, test_split, val_split, seed):
    """
    Loads the real dataset and splits it into train, validation, and test subsets.

    Supports loading classification data using torchvision.datasets.ImageFolder (requires
    data to be organized in subdirectories named by class, e.g., root/class_0/img.png)
    and segmentation data using the custom MedicalImageDataset (requires an 'images'
    subfolder and a specified mask subfolder).

    Args:
        real_data_dir (str): Path to the root directory of the real dataset.
        image_size (int): Target size (height/width) for image resizing.
        task (str): The downstream task type ('classification' or 'segmentation').
        mask_folder (str): Subfolder name containing masks (only used for 'segmentation' task).
        test_split (float): Fraction of the dataset to use for the test set (e.g., 0.2).
        val_split (float): Fraction of the dataset to use for the validation set (e.g., 0.2).
        seed (int): Random seed for shuffling before splitting to ensure reproducibility.

    Returns:
        tuple: Contains:
            - real_train_dataset (Dataset): Subset for training.
            - real_val_dataset (Dataset or None): Subset for validation (None if val_split is 0).
            - real_test_dataset (Dataset): Subset for testing.
            - data_transform (Transform or tuple): The transform(s) used, needed for synthetic data.
                                                  Returns a single transform for classification,
                                                  or a tuple (image_transform, mask_transform)
                                                  for segmentation.

    Raises:
        FileNotFoundError: If the real_data_dir or required subfolders don't exist.
        ValueError: If task is unknown, or if splits result in zero samples.
    """
    transforms_builder = MedicalImageTransforms()

    if task == "segmentation":
        if not mask_folder:
            raise ValueError("mask_folder must be provided for segmentation task")
        image_transform = transforms_builder.get_image_transforms(img_size=image_size)
        mask_transform = transforms_builder.get_mask_transforms(img_size=image_size)
        real_dataset = MedicalImageDataset(
            data_dir=real_data_dir,
            image_folder="images", # Assuming images are in 'images' subfolder
            mask_folder=mask_folder,
            transform=image_transform,
            mask_transform=mask_transform
        )
    elif task == "classification":
        # Assumes ImageFolder structure (root/class_name/image.png)
        transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # Assuming 3 channels
        ])
        real_dataset = ImageFolder(root=real_data_dir, transform=transform)
    else:
        raise ValueError(f"Unknown task: {task}")

    # Split dataset indices
    dataset_size = len(real_dataset)
    indices = list(range(dataset_size))
    np.random.seed(seed) # Ensure consistent shuffle
    np.random.shuffle(indices)

    test_set_size = int(np.floor(test_split * dataset_size))
    val_set_size = int(np.floor(val_split * dataset_size))
    train_set_size = dataset_size - test_set_size - val_set_size

    if train_set_size <= 0 or test_set_size <= 0:
         raise ValueError("Train or test split resulted in zero samples. Adjust splits or check dataset size.")

    test_indices = indices[:test_set_size]
    val_indices = indices[test_set_size : test_set_size + val_set_size]
    train_indices = indices[test_set_size + val_set_size :]

    logging.info(f"Real data splits: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")

    # Create subset datasets
    real_train_dataset = Subset(real_dataset, train_indices)
    real_val_dataset = Subset(real_dataset, val_indices) if val_set_size > 0 else None
    real_test_dataset = Subset(real_dataset, test_indices)

    return real_train_dataset, real_val_dataset, real_test_dataset, transform if task == 'classification' else (image_transform, mask_transform)


def load_synthetic_data(synthetic_data_dir, image_size, task, transform, mask_transform=None, synthetic_mask_folder=None):
    """
    Loads the synthetic dataset generated by the ControlNet pipeline.

    Attempts to load data based on the task:
    - 'classification': Checks if `synthetic_data_dir` looks like an ImageFolder structure
      (contains multiple subdirectories). If so, loads using `ImageFolder`. If not,
      attempts to load images from the root of `synthetic_data_dir` using
      `MedicalImageDataset` (assumes single class or labels are irrelevant).
    - 'segmentation': Uses `MedicalImageDataset`, assuming images are in the root
      (`image_folder="."`) and masks are in `synthetic_mask_folder` (relative to
      `synthetic_data_dir`).

    Args:
        synthetic_data_dir (str): Path to the directory containing generated synthetic images.
        image_size (int): Target size for image resizing (should match real data).
        task (str): The downstream task type ('classification' or 'segmentation').
        transform (Transform or tuple): The transform(s) to apply. Should match the
                                        transforms used for the real data.
                                        Expects a single transform for classification,
                                        or (image_transform, mask_transform) for segmentation.
        mask_transform (Transform, optional): Explicitly passed mask transform (used if transform is not tuple).
        synthetic_mask_folder (str, optional): Subfolder containing synthetic masks (needed for segmentation).

    Returns:
        Dataset or None: The loaded synthetic dataset, or None if the directory is not found
                       or loading fails.
    """
    if not os.path.isdir(synthetic_data_dir):
        logging.warning(f"Synthetic data directory not found: {synthetic_data_dir}. Returning None.")
        return None

    img_tfm = transform[0] if isinstance(transform, tuple) else transform
    mask_tfm = transform[1] if isinstance(transform, tuple) else mask_transform

    if task == "segmentation":
        if not synthetic_mask_folder:
            logging.error("Synthetic mask folder must be specified for segmentation task when loading synthetic data.")
            return None # Segmentation requires masks
        try:
            logging.info(f"Loading synthetic segmentation data from: {synthetic_data_dir} (masks: {synthetic_mask_folder})")
            # Assuming generated images are flat in synthetic_data_dir, masks in subfolder
            return MedicalImageDataset(
                data_dir=synthetic_data_dir,
                image_folder=".", # Load images from root
                mask_folder=synthetic_mask_folder,
                transform=img_tfm,
                mask_transform=mask_tfm
            )
        except Exception as e:
            logging.error(f"Failed to load synthetic segmentation data: {e}")
            return None

    elif task == "classification":
        # Check for ImageFolder structure
        try:
             subdirs = [d for d in os.listdir(synthetic_data_dir) if os.path.isdir(os.path.join(synthetic_data_dir, d))]
             is_image_folder = len(subdirs) > 0 and all(f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))
                                                    for subdir in subdirs
                                                    for f in os.listdir(os.path.join(synthetic_data_dir, subdir)))
        except Exception:
             is_image_folder = False # Error listing means unlikely ImageFolder

        if is_image_folder:
            try:
                logging.info(f"Loading synthetic classification data using ImageFolder structure from: {synthetic_data_dir}")
                return ImageFolder(root=synthetic_data_dir, transform=img_tfm)
            except Exception as e:
                 logging.error(f"Failed to load synthetic data using ImageFolder: {e}")
                 return None
        else:
            # Load as flat directory using MedicalImageDataset (no masks)
            try:
                logging.info(f"Loading synthetic classification data from flat directory: {synthetic_data_dir}")
                # Note: This dataset won't have proper labels unless MedicalImageDataset is modified
                # or labels are inferred/assigned later.
                return MedicalImageDataset(
                    data_dir=synthetic_data_dir,
                    image_folder=".", # Load images from root
                    mask_folder=None,
                    transform=img_tfm,
                    mask_transform=None
                )
            except Exception as e:
                logging.error(f"Could not load synthetic classification data from flat directory {synthetic_data_dir}: {e}")
                return None
    else:
        logging.error(f"Unknown task type '{task}' for loading synthetic data.")
        return None


def main():
    """
    Main evaluation function.
    """
    args = parse_args()

    # Load config FIRST to get defaults
    try:
        config = Config.from_yaml(args.config)
    except Exception as e:
        logging.error(f"Error loading config file {args.config}: {e}. Using command-line args only.")
        # Provide default config structure if loading fails, or exit?
        # For now, try to proceed with args, but essential values might be missing.
        config = None # Indicate config loading failed

    # Determine parameters, prioritizing command-line args over config file
    output_dir = args.output_dir or (config.training.output_dir + "_evaluation" if config else "evaluation_results")
    batch_size = args.batch_size or (config.data.batch_size if config else 16)
    num_epochs = args.num_epochs or (config.training.downstream_num_epochs if config else 20)
    image_size = args.image_size or (config.data.image_size if config else 128)
    learning_rate = args.learning_rate or (config.training.downstream_learning_rate if config else 1e-4)
    real_data_dir = args.real_data_dir # Required arg
    synthetic_data_dir = args.synthetic_data_dir # Required arg
    task = args.task # Required arg
    mask_folder = args.mask_folder or (config.data.mask_folder if config else "masks")
    synthetic_mask_folder = args.synthetic_mask_folder or mask_folder # Default to same as real
    val_split = args.val_split or (config.data.val_split if config else 0.2)
    test_split = args.test_split or (config.data.test_split if config else 0.2)
    seed = args.seed or (config.training.seed if config else 42)

    # Use determined values from here on
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, "evaluate.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers= [
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ]
    )
    logging.info(f"Logging to {log_file}")
    logging.info(f"Using parameters: task={task}, image_size={image_size}, batch_size={batch_size}, epochs={num_epochs}, lr={learning_rate}")
    logging.info(f"Real data: {real_data_dir}, Synthetic data: {synthetic_data_dir}, Output: {output_dir}")

    # Set random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load and split real datasets
    logging.info(f"Loading real data from: {real_data_dir}")
    try:
        real_train_dataset, real_val_dataset, real_test_dataset, data_transform = load_and_split_real_data(
            real_data_dir=real_data_dir,
            image_size=image_size, # Use determined value
            task=task,
            mask_folder=mask_folder if task == "segmentation" else None,
            test_split=test_split, # Use determined value
            val_split=val_split,   # Use determined value
            seed=seed              # Use determined value
        )
    except Exception as e:
        logging.error(f"Failed to load or split real data: {e}")
        sys.exit(1)

    # Load synthetic dataset
    logging.info(f"Loading synthetic data from: {synthetic_data_dir}")
    synthetic_dataset = load_synthetic_data(
        synthetic_data_dir=synthetic_data_dir,
        image_size=image_size, # Use determined value
        task=task,
        transform=data_transform,
        mask_transform=data_transform[1] if isinstance(data_transform, tuple) else None,
        synthetic_mask_folder=synthetic_mask_folder # Use determined value
    )

    if synthetic_dataset:
        logging.info(f"Synthetic dataset size: {len(synthetic_dataset)}")
        # Create augmented dataset (real train + synthetic)
        augmented_train_dataset = ConcatDataset([real_train_dataset, synthetic_dataset])
        logging.info(f"Augmented training dataset size: {len(augmented_train_dataset)}")
    else:
        logging.warning("Synthetic dataset not loaded. Running evaluation only on real data.")
        augmented_train_dataset = real_train_dataset # Fallback to real data only

    # Create dataloaders
    num_workers = min(os.cpu_count(), 4) # Use fewer workers if CPU count is low
    real_train_dataloader = DataLoader(
        real_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    augmented_train_dataloader = DataLoader(
        augmented_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_dataloader = DataLoader(
        real_val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    ) if real_val_dataset else None

    test_dataloader = DataLoader(
        real_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # --- Run Comparison --- #
    all_real_metrics = None
    all_augmented_metrics = None

    # 1. Train and Evaluate on Real Data Only
    logging.info(f"=== Training downstream model on REAL data only ({task}) ===")
    try:
        real_metrics_run = compare_performance(
            train_loader=real_train_dataloader,
            val_loader=val_dataloader,
            test_loader=test_dataloader,
            task=task,
            num_epochs=num_epochs,      # Use determined value
            device=device,
            learning_rate=learning_rate, # Use determined value
            model_save_path=os.path.join(output_dir, "downstream_model_real_only.pth")
        )
        all_real_metrics = real_metrics_run
        logging.info("Finished training on real data.")
    except Exception as e:
        logging.error(f"Error during training/evaluation on real data: {e}")

    # 2. Train and Evaluate on Augmented Data (if synthetic data loaded)
    if synthetic_dataset:
        logging.info(f"=== Training downstream model on AUGMENTED data ({task}) ===")
        try:
            augmented_metrics_run = compare_performance(
                train_loader=augmented_train_dataloader,
                val_loader=val_dataloader,
                test_loader=test_dataloader,
                task=task,
                num_epochs=num_epochs,      # Use determined value
                device=device,
                learning_rate=learning_rate, # Use determined value
                model_save_path=os.path.join(output_dir, "downstream_model_augmented.pth")
            )
            all_augmented_metrics = augmented_metrics_run
            logging.info("Finished training on augmented data.")
        except Exception as e:
            logging.error(f"Error during training/evaluation on augmented data: {e}")
    else:
        logging.info("Skipping training on augmented data as synthetic dataset was not loaded.")

    # --- Plotting and Saving Results --- #
    logging.info("Plotting metrics...")

    if all_real_metrics:
         metric_key_train = "train_loss"
         metric_key_val = "val_dice" if task == "segmentation" else "val_acc"

         # Plot Training Loss Comparison
         plt.figure(figsize=(10, 5))
         plt.plot(all_real_metrics["train_loss"], label="Real Data Train Loss")
         if all_augmented_metrics:
             plt.plot(all_augmented_metrics["train_loss"], label="Augmented Data Train Loss")
         plt.xlabel("Epoch")
         plt.ylabel("Loss")
         plt.title(f"Downstream Task ({task}) Training Loss")
         plt.legend()
         plt.grid(True, linestyle="--", alpha=0.7)
         plt.savefig(os.path.join(output_dir, "comparison_training_loss.png"))
         plt.close()

         # Plot Validation Metric Comparison
         if metric_key_val in all_real_metrics and all_real_metrics[metric_key_val]: # Check if val metrics exist
             plt.figure(figsize=(10, 5))
             plt.plot(all_real_metrics[metric_key_val], label=f"Real Data Val {metric_key_val}")
             if all_augmented_metrics and metric_key_val in all_augmented_metrics and all_augmented_metrics[metric_key_val]:
                 plt.plot(all_augmented_metrics[metric_key_val], label=f"Augmented Data Val {metric_key_val}")
             plt.xlabel("Epoch")
             plt.ylabel(metric_key_val)
             plt.title(f"Downstream Task ({task}) Validation Performance")
             plt.legend()
             plt.grid(True, linestyle="--", alpha=0.7)
             plt.savefig(os.path.join(output_dir, f"comparison_validation_{metric_key_val}.png"))
             plt.close()

         # Print Final Test Metrics
         test_metric_key = "test_dice" if task == "segmentation" else "test_acc"
         if test_metric_key in all_real_metrics:
              logging.info(f"Final Test Metric ({test_metric_key}) with REAL data: {all_real_metrics[test_metric_key]:.4f}")
         if all_augmented_metrics and test_metric_key in all_augmented_metrics:
              logging.info(f"Final Test Metric ({test_metric_key}) with AUGMENTED data: {all_augmented_metrics[test_metric_key]:.4f}")
              if test_metric_key in all_real_metrics:
                   improvement = all_augmented_metrics[test_metric_key] - all_real_metrics[test_metric_key]
                   logging.info(f"Improvement: {improvement:.4f}")

    logging.info(f"Evaluation complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main() 