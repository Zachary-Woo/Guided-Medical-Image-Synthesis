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
from torch.utils.data import DataLoader, ConcatDataset
from src.preprocessing.data_loader import MedicalImageDataset
from src.preprocessing.transforms import MedicalImageTransforms
from src.evaluation.downstream import compare_performance
from src.evaluation.metrics import evaluate_model_performance
from src.utils.visualization import plot_metrics, create_comparison_grid


def parse_args():
    """
    Parse command line arguments for evaluation.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Evaluate downstream tasks using synthetic medical images")
    
    parser.add_argument("--real_data_dir", type=str, required=True, help="Directory with real data")
    parser.add_argument("--synthetic_data_dir", type=str, required=True, help="Directory with synthetic data")
    parser.add_argument("--output_dir", type=str, default="evaluation", help="Output directory for results")
    parser.add_argument("--task", type=str, default="segmentation", choices=["segmentation", "classification"], 
                        help="Downstream task to evaluate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--image_size", type=int, default=256, help="Image size")
    parser.add_argument("--mask_folder", type=str, default="masks", help="Folder with masks (for segmentation)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()


def load_datasets(real_data_dir, synthetic_data_dir, image_size, task, mask_folder=None):
    """
    Load real and synthetic datasets.
    
    Args:
        real_data_dir (str): Directory with real data
        synthetic_data_dir (str): Directory with synthetic data
        image_size (int): Image size
        task (str): "segmentation" or "classification"
        mask_folder (str, optional): Folder with masks
        
    Returns:
        tuple: Real dataset, augmented dataset (real + synthetic), test dataset
    """
    transforms = MedicalImageTransforms()
    
    if task == "segmentation":
        # For segmentation, we need both images and masks
        if not mask_folder:
            raise ValueError("mask_folder must be provided for segmentation task")
        
        image_transform = transforms.get_image_transforms(img_size=image_size)
        mask_transform = transforms.get_mask_transforms(img_size=image_size)
        
        # Load real dataset
        real_dataset = MedicalImageDataset(
            data_dir=real_data_dir,
            mask_folder=mask_folder,
            transform=image_transform,
            mask_transform=mask_transform
        )
        
        # Load synthetic dataset
        synthetic_dataset = MedicalImageDataset(
            data_dir=synthetic_data_dir,
            mask_folder=mask_folder,
            transform=image_transform,
            mask_transform=mask_transform
        )
    else:
        # For classification, we need images and class labels
        # This assumes a folder structure with class names as folder names
        from torchvision.datasets import ImageFolder
        from torchvision import transforms as T
        
        transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ])
        
        # Load real dataset
        real_dataset = ImageFolder(
            root=real_data_dir,
            transform=transform
        )
        
        # Load synthetic dataset
        synthetic_dataset = ImageFolder(
            root=synthetic_data_dir,
            transform=transform
        )
    
    # Split real dataset into train and test
    train_size = int(0.8 * len(real_dataset))
    test_size = len(real_dataset) - train_size
    
    real_train_dataset, real_test_dataset = torch.utils.data.random_split(
        real_dataset, [train_size, test_size]
    )
    
    # Create augmented dataset (real + synthetic)
    augmented_dataset = ConcatDataset([real_train_dataset, synthetic_dataset])
    
    return real_train_dataset, augmented_dataset, real_test_dataset


def main():
    """
    Main evaluation function.
    """
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(args.output_dir, "evaluate.log"))
        ]
    )
    
    # Set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Load datasets
    logging.info("Loading datasets...")
    real_dataset, augmented_dataset, test_dataset = load_datasets(
        real_data_dir=args.real_data_dir,
        synthetic_data_dir=args.synthetic_data_dir,
        image_size=args.image_size,
        task=args.task,
        mask_folder=args.mask_folder if args.task == "segmentation" else None
    )
    
    logging.info(f"Real dataset size: {len(real_dataset)}")
    logging.info(f"Augmented dataset size: {len(augmented_dataset)}")
    logging.info(f"Test dataset size: {len(test_dataset)}")
    
    # Create dataloaders
    real_dataloader = DataLoader(
        real_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    augmented_dataloader = DataLoader(
        augmented_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Compare performance
    logging.info(f"Comparing performance on {args.task} task...")
    real_metrics, augmented_metrics = compare_performance(
        real_data_loader=real_dataloader,
        augmented_data_loader=augmented_dataloader,
        test_loader=test_dataloader,
        task=args.task,
        num_epochs=args.num_epochs
    )
    
    # Plot and save metrics
    logging.info("Plotting metrics...")
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(real_metrics["train_loss"], label="Real data")
    plt.plot(augmented_metrics["train_loss"], label="Augmented data")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(os.path.join(args.output_dir, "training_loss.png"))
    
    # Plot validation metrics (dice or accuracy)
    metric_key = "val_dice" if args.task == "segmentation" else "val_acc"
    plt.figure(figsize=(10, 5))
    plt.plot(real_metrics[metric_key], label="Real data")
    plt.plot(augmented_metrics[metric_key], label="Augmented data")
    plt.xlabel("Epoch")
    plt.ylabel("Dice" if args.task == "segmentation" else "Accuracy")
    plt.title(f"Validation {metric_key}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(os.path.join(args.output_dir, f"validation_{metric_key}.png"))
    
    logging.info(f"Evaluation complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main() 