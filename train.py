#!/usr/bin/env python
"""
Training script for ControlNet medical image synthesis.
"""

import os
import sys
import torch
import numpy as np
import random
import logging
from torch.utils.data import DataLoader, random_split
from src.utils.config import get_config_from_args, parse_args
from src.preprocessing.data_loader import MedicalImageDataset
from src.preprocessing.transforms import MedicalImageTransforms
from src.controlnet_training.model import setup_controlnet, freeze_model_components
from src.controlnet_training.trainer import ControlNetTrainer


def set_seed(seed):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    """
    Main training function.
    """
    # Parse arguments
    args = parse_args()
    config = get_config_from_args(args)
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(config.training.output_dir, "train.log"))
        ]
    )
    
    # Set random seed
    set_seed(config.training.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Set up transforms
    transforms = MedicalImageTransforms()
    image_transform = transforms.get_image_transforms(img_size=config.data.image_size)
    mask_transform = transforms.get_mask_transforms(img_size=config.data.image_size)
    
    # Load dataset
    logging.info(f"Loading dataset: {config.data.dataset_name}")
    dataset = MedicalImageDataset(
        data_dir=config.data.data_dir,
        mask_folder=config.data.mask_folder,
        transform=image_transform,
        mask_transform=mask_transform
    )
    
    # Split dataset
    logging.info("Splitting dataset into train/val/test")
    val_size = int(len(dataset) * config.data.val_split)
    test_size = int(len(dataset) * config.data.test_split)
    train_size = len(dataset) - val_size - test_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    # Set up model components
    logging.info("Setting up model components")
    model_components = setup_controlnet(
        pretrained_model_id=config.model.pretrained_model_id,
        controlnet_id=config.model.controlnet_id
    )
    
    # Freeze components
    freeze_model_components(model_components, config.model.trainable_modules)
    
    # Set up trainer
    logging.info("Setting up trainer")
    trainer = ControlNetTrainer(
        model_components=model_components,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        output_dir=config.training.output_dir,
        log_wandb=config.training.log_wandb,
        mixed_precision=config.model.mixed_precision,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps
    )
    
    # Train model
    logging.info("Starting training")
    trainer.train(
        num_epochs=config.training.num_epochs,
        save_steps=config.training.save_steps
    )
    
    # Save final model
    logging.info("Saving final model")
    trainer.save_checkpoint("final")
    
    logging.info("Training complete")


if __name__ == "__main__":
    main() 