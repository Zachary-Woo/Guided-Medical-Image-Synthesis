#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import logging
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import cv2
from tqdm import tqdm
import random
import nibabel as nib
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Try to import SAM2 from GitHub repository first
SAM2_SOURCE = None
try:
    import segment_anything_2
    from segment_anything_2.build_sam2 import sam2_model_registry
    from segment_anything_2.modeling import Sam2MaskDecoder
    SAM2_SOURCE = "GITHUB_SAM2"
    logger.info("Using SAM2 from GitHub repository (segment-anything-2)")
except ImportError:
    try:
        # Try original SAM from GitHub
        import segment_anything
        from segment_anything import sam_model_registry
        from segment_anything.modeling import MaskDecoder
        SAM2_SOURCE = "GITHUB_SAM"
        logger.info("Using original SAM from GitHub repository (segment-anything)")
    except ImportError:
        # Fall back to transformers
        try:
            from transformers import Sam2Model, Sam2Processor, SamModel, SamProcessor
            SAM2_SOURCE = "HUGGINGFACE"
            logger.info("Using SAM/SAM2 from HuggingFace transformers")
        except ImportError:
            logger.error("Failed to import any SAM implementation. Make sure segment-anything-2, segment-anything, or transformers is installed.")
            sys.exit(1)

# Add parent directory to path for importing local modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.mri_utils import (
    load_nifti_volume,
    normalize_intensity,
    extract_slice,
    apply_brain_mask
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def find_brats_data(brats_path):
    """Find all subject directories in BraTS dataset with required files."""
    brats_path = Path(brats_path)
    subjects = []
    
    for subject_dir in brats_path.glob("*"):
        if not subject_dir.is_dir():
            continue
        
        # Look for the required files (T1CE and segmentation)
        t1ce_files = list(subject_dir.glob("*t1ce.nii.gz"))
        seg_files = list(subject_dir.glob("*seg.nii.gz"))
        
        if t1ce_files and seg_files:
            subjects.append({
                "id": subject_dir.name,
                "t1ce_path": t1ce_files[0],
                "seg_path": seg_files[0]
            })
    
    logger.info(f"Found {len(subjects)} subjects with T1CE and segmentation data")
    return subjects

class BraTSDataLoader:
    """Data loader for BraTS dataset for SAM2 fine-tuning."""
    
    def __init__(self, brats_path, slice_axis=2, image_size=1024, augmentation=True):
        """
        Initialize BraTS data loader.
        
        Args:
            brats_path: Path to BraTS dataset
            slice_axis: Axis for slice extraction (0, 1, or 2)
            image_size: Maximum size for resizing images
            augmentation: Whether to apply data augmentation
        """
        self.brats_path = Path(brats_path)
        self.slice_axis = slice_axis
        self.image_size = image_size
        self.augmentation = augmentation
        
        # Find all subjects with required files
        self.subjects = find_brats_data(brats_path)
        
        # Track loaded data for caching
        self.cached_subject = None
        self.cached_t1ce = None
        self.cached_seg = None
        
        logger.info(f"Initialized data loader with {len(self.subjects)} subjects")
    
    def load_nifti(self, file_path):
        """Load NIfTI file as numpy array."""
        nii_img = nib.load(file_path)
        return nii_img.get_fdata()
    
    def normalize_intensity(self, volume):
        """Normalize intensity of MRI volume."""
        p1, p99 = np.percentile(volume, (1, 99))
        volume = np.clip(volume, p1, p99)
        volume = (volume - p1) / (p99 - p1)
        return volume
    
    def get_batch(self):
        """Get a random batch of data (one image with multiple instances)."""
        # Select a random subject
        subject = random.choice(self.subjects)
        
        # Check if we already have this subject loaded (simple caching)
        if self.cached_subject != subject["id"]:
            # Load MRI and segmentation data
            t1ce_volume = self.load_nifti(subject["t1ce_path"])
            seg_volume = self.load_nifti(subject["seg_path"])
            
            # Normalize MRI data
            t1ce_volume = self.normalize_intensity(t1ce_volume)
            
            # Update cache
            self.cached_subject = subject["id"]
            self.cached_t1ce = t1ce_volume
            self.cached_seg = seg_volume
        else:
            # Use cached data
            t1ce_volume = self.cached_t1ce
            seg_volume = self.cached_seg
        
        # Get dimensions for the selected axis
        if self.slice_axis == 0:
            num_slices = t1ce_volume.shape[0]
        elif self.slice_axis == 1:
            num_slices = t1ce_volume.shape[1]
        else:  # self.slice_axis == 2
            num_slices = t1ce_volume.shape[2]
        
        # Select a random slice that contains tumor
        valid_slices = []
        for i in range(num_slices):
            # Extract slice
            if self.slice_axis == 0:
                seg_slice = seg_volume[i, :, :]
            elif self.slice_axis == 1:
                seg_slice = seg_volume[:, i, :]
            else:  # self.slice_axis == 2
                seg_slice = seg_volume[:, :, i]
            
            # Check if slice contains tumor
            if np.any(seg_slice > 0):
                valid_slices.append(i)
        
        # Fallback to any slice if no tumor slices found
        if not valid_slices:
            valid_slices = list(range(num_slices))
        
        # Select a random slice
        slice_idx = random.choice(valid_slices)
        
        # Extract the selected slice
        if self.slice_axis == 0:
            t1ce_slice = t1ce_volume[slice_idx, :, :]
            seg_slice = seg_volume[slice_idx, :, :]
        elif self.slice_axis == 1:
            t1ce_slice = t1ce_volume[:, slice_idx, :]
            seg_slice = seg_volume[:, slice_idx, :]
        else:  # self.slice_axis == 2
            t1ce_slice = t1ce_volume[:, :, slice_idx]
            seg_slice = seg_volume[:, :, slice_idx]
        
        # Resize if needed
        h, w = t1ce_slice.shape
        r = min(self.image_size / w, self.image_size / h)
        new_h, new_w = int(h * r), int(w * r)
        
        t1ce_slice_resized = cv2.resize(t1ce_slice, (new_w, new_h))
        seg_slice_resized = cv2.resize(seg_slice, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # Convert to RGB
        t1ce_rgb = np.stack([t1ce_slice_resized] * 3, axis=-1)
        t1ce_rgb = (t1ce_rgb * 255).astype(np.uint8)
        
        # Extract individual tumor instances
        # BraTS labels: 1=necrotic core, 2=edema, 4=enhancing tumor
        tumor_labels = np.unique(seg_slice_resized)
        tumor_labels = tumor_labels[tumor_labels > 0]  # Remove background
        
        masks = []
        points = []
        
        for label in tumor_labels:
            # Create binary mask for this tumor component
            tumor_mask = (seg_slice_resized == label).astype(np.uint8)
            
            # Skip very small tumor components
            if np.sum(tumor_mask) < 25:
                continue
            
            masks.append(tumor_mask)
            
            # Find a random point inside the tumor
            tumor_coords = np.argwhere(tumor_mask > 0)
            if len(tumor_coords) > 0:
                rand_idx = np.random.randint(len(tumor_coords))
                y, x = tumor_coords[rand_idx]
                points.append([[x, y]])
        
        # Data augmentation
        if self.augmentation and random.random() < 0.5:
            # Random horizontal flip
            if random.random() < 0.5:
                t1ce_rgb = t1ce_rgb[:, ::-1, :]
                for i in range(len(masks)):
                    masks[i] = masks[i][:, ::-1]
                    points[i][0][0] = new_w - points[i][0][0]
            
            # Random vertical flip
            if random.random() < 0.5:
                t1ce_rgb = t1ce_rgb[::-1, :, :]
                for i in range(len(masks)):
                    masks[i] = masks[i][::-1, :]
                    points[i][0][1] = new_h - points[i][0][1]
            
            # Random brightness/contrast
            if random.random() < 0.5:
                alpha = 0.8 + random.random() * 0.4  # 0.8-1.2
                beta = -10 + random.random() * 20    # -10 to 10
                t1ce_rgb = np.clip(alpha * t1ce_rgb + beta, 0, 255).astype(np.uint8)
        
        # If no tumor instances found, return None to skip this batch
        if not masks:
            return None
        
        return t1ce_rgb, np.array(masks), np.array(points), np.ones([len(masks), 1])

def load_sam_model(args):
    """
    Load the SAM model using local pip-installed repositories.
    First tries to use SAM2 (segment-anything-2) and falls back to original SAM if needed.
    
    Args:
        args: Command line arguments including checkpoint path, model ID, etc.
        
    Returns:
        model: The loaded SAM/SAM2 model
        processor: Image processor for the model
        resume_step: Step to resume training from (0 if not resuming)
    """
    logger.info("Loading SAM model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resume_step = 0
    
    # First, try to load from checkpoint if provided
    if args.sam_checkpoint and os.path.exists(args.sam_checkpoint):
        logger.info(f"Loading checkpoint from {args.sam_checkpoint}")
        checkpoint = torch.load(args.sam_checkpoint, map_location=device)
        
        # Check if this is a full checkpoint (with optimizer state) or just the model
        if 'model_state_dict' in checkpoint:
            model_dict = checkpoint['model_state_dict']
            # If we have step information, use it to resume training
            if 'step' in checkpoint:
                resume_step = checkpoint['step']
                logger.info(f"Resuming from step {resume_step}")
        else:
            # Assume the checkpoint is just the model state dict
            model_dict = checkpoint
        
        # Try to first load as SAM2
        try:
            import segment_anything_2  # Import the local pip installation
            # Prepare SAM2 model configuration
            if args.sam_config and os.path.exists(args.sam_config):
                logger.info(f"Loading SAM2 configuration from {args.sam_config}")
                with open(args.sam_config, "r") as f:
                    model_config = json.load(f)
                model = segment_anything_2.build_sam2_vit_l(args.sam_config)
            else:
                logger.info("Using default SAM2 ViT-L configuration")
                model = segment_anything_2.build_sam2_vit_l()
            
            # Load checkpoint weights
            model.load_state_dict(model_dict, strict=False)
            processor = segment_anything_2.sam2_model_registry["vit_l"]
            
            logger.info("Successfully loaded SAM2 model from checkpoint")
        except (ImportError, ValueError, KeyError) as e:
            logger.warning(f"Failed to load as SAM2: {e}. Trying original SAM...")
            
            # Try loading as original SAM
            try:
                import segment_anything  # Import the local pip installation
                if args.sam_config and os.path.exists(args.sam_config):
                    logger.info(f"Loading SAM configuration from {args.sam_config}")
                    with open(args.sam_config, "r") as f:
                        model_config = json.load(f)
                    model = segment_anything.build_sam_vit_h(model_config)
                else:
                    logger.info("Using default SAM ViT-H configuration")
                    model = segment_anything.build_sam_vit_h()
                
                # Load checkpoint weights
                model.load_state_dict(model_dict, strict=False)
                processor = segment_anything.SamPredictor(model)
                
                logger.info("Successfully loaded SAM model from checkpoint")
            except Exception as e2:
                logger.error(f"Failed to load from checkpoint as either SAM2 or SAM: {e2}")
                raise RuntimeError("Could not load model from checkpoint")
    
    # If no checkpoint provided or loading failed, try to load from HuggingFace or default
    else:
        if args.sam_model_id:
            try:
                logger.info(f"Loading SAM model from HuggingFace: {args.sam_model_id}")
                from transformers import SamModel, SamProcessor
                model = SamModel.from_pretrained(args.sam_model_id)
                processor = SamProcessor.from_pretrained(args.sam_model_id)
            except Exception as e:
                logger.error(f"Failed to load from HuggingFace: {e}")
                raise RuntimeError("Could not load model from HuggingFace")
        else:
            # Try local GitHub implementations
            try:
                # First try SAM2
                import segment_anything_2
                logger.info("Loading default SAM2 ViT-L model")
                model = segment_anything_2.build_sam2_vit_l()
                processor = segment_anything_2.sam2_model_registry["vit_l"]
                logger.info("Successfully loaded SAM2 model")
            except ImportError:
                # Fall back to original SAM
                try:
                    import segment_anything
                    logger.info("Loading default SAM ViT-H model")
                    model = segment_anything.build_sam_vit_h()
                    processor = segment_anything.SamPredictor(model)
                    logger.info("Successfully loaded SAM model")
                except ImportError as e:
                    logger.error(f"Neither SAM2 nor SAM is installed: {e}")
                    raise RuntimeError("No SAM implementation found. Install segment-anything-2 or segment-anything")
    
    # Move model to device
    model = model.to(device)
    
    # Freeze image encoder if specified
    if not args.train_image_encoder:
        logger.info("Freezing image encoder parameters")
        for param in model.image_encoder.parameters():
            param.requires_grad = False
    
    return model, processor, resume_step

def save_checkpoint(model, optimizer, step_or_name, args):
    """Save model checkpoint."""
    logger.info(f"Saving checkpoint at step {step_or_name}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create checkpoint path based on step or name
    if isinstance(step_or_name, int):
        checkpoint_path = os.path.join(args.output_dir, f"sam2_checkpoint_{step_or_name:06d}.pt")
    else:
        checkpoint_path = os.path.join(args.output_dir, f"sam2_checkpoint_{step_or_name}.pt")
    
    # Save model, optimizer state, and training arguments
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step_or_name if isinstance(step_or_name, int) else 0,
        'args': vars(args)
    }
    
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path}")
    return checkpoint_path

def validate(model, valid_loader, args):
    """Evaluate the model on validation data."""
    model.eval()
    total_iou = 0.0
    count = 0
    
    with torch.no_grad():
        for step, batch in enumerate(valid_loader):
            if step >= args.max_eval_steps:
                break
                
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            
            # Create point prompts from ground truth masks
            batch_points = []
            batch_labels = []
            
            for mask in masks:
                pos_indices = torch.nonzero(mask > 0.5, as_tuple=True)
                if pos_indices[0].numel() > 0:
                    # Randomly select a positive pixel
                    random_idx = random.randint(0, pos_indices[0].numel() - 1)
                    y, x = pos_indices[0][random_idx], pos_indices[1][random_idx]
                    point = torch.tensor([x, y])
                    label = torch.tensor([1])  # Foreground
                else:
                    # If no foreground, use center as background
                    h, w = mask.shape
                    point = torch.tensor([w//2, h//2])
                    label = torch.tensor([0])  # Background
                
                batch_points.append(point.unsqueeze(0))
                batch_labels.append(label)
            
            batch_points = torch.stack(batch_points).to(device)
            batch_labels = torch.stack(batch_labels).to(device)
            
            # Get image embeddings
            image_embeddings = model.image_encoder(images)
            
            # Process the points with the prompt encoder
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=batch_points,
                labels=batch_labels,
                boxes=None,
                masks=None,
            )
            
            # Predict masks
            if SAM2_SOURCE == "GITHUB_SAM2":
                mask_predictions, _ = model.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
            else:  # SAM2_SOURCE == "GITHUB_SAM"
                mask_predictions, _ = model.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
            
            # Calculate IoU
            mask_predictions_binary = (torch.sigmoid(mask_predictions) > 0.5).float()
            intersection = (mask_predictions_binary * masks.unsqueeze(1)).sum(dim=(2, 3))
            union = (mask_predictions_binary + masks.unsqueeze(1) - mask_predictions_binary * masks.unsqueeze(1)).sum(dim=(2, 3))
            iou = (intersection / (union + 1e-6)).mean()
            
            total_iou += iou.item()
            count += 1
    
    return total_iou / count if count > 0 else 0.0

def train_model(args):
    """Train the SAM2 model on BraTS dataset."""
    # Set up logger
    setup_logging(args.output_dir)
    logger.info("Starting SAM2 training on BraTS dataset")
    logger.info(f"Arguments: {args}")
    
    # Set seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Find all valid BraTS subjects
    subject_dirs = find_brats_data(args.brats_path)
    logger.info(f"Found {len(subject_dirs)} subjects in BraTS dataset")
    
    # Split into train and validation sets
    val_size = min(int(len(subject_dirs) * 0.1), 10)  # 10% for validation, but at most 10 subjects
    train_subjects = subject_dirs[:-val_size]
    val_subjects = subject_dirs[-val_size:]
    
    logger.info(f"Training on {len(train_subjects)} subjects, validating on {len(val_subjects)} subjects")
    
    # Create data loaders
    train_loader = BraTSDataLoader(
        subject_dirs=train_subjects,
        slice_axis=args.slice_axis,
        batch_size=args.batch_size,
        use_augmentation=args.augmentation,
        cache_data=True
    )
    
    val_loader = BraTSDataLoader(
        subject_dirs=val_subjects,
        slice_axis=args.slice_axis,
        batch_size=args.batch_size,
        use_augmentation=False,
        cache_data=True
    )
    
    # Load model
    model, processor, resume_step = load_sam_model(args)
    
    # Set up optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Set up scheduler
    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.max_train_steps, eta_min=args.min_learning_rate
        )
    else:
        scheduler = None
    
    # Train
    train(
        model=model,
        processor=processor,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        args=args,
        resume_step=resume_step
    )

def train(model, processor, train_loader, val_loader, optimizer, scheduler, args, resume_step=0):
    """Main training loop."""
    device = next(model.parameters()).device
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None
    
    # For tracking metrics
    best_val_dice = 0.0
    total_loss = 0.0
    step = resume_step
    
    # Start timer
    start_time = time.time()
    last_log_time = start_time
    
    logger.info(f"Starting training from step {resume_step}")
    
    # Training loop
    while step < args.max_train_steps:
        model.train()
        
        for batch in train_loader:
            # Skip steps if resuming
            if step < resume_step:
                step += 1
                continue
            
            # Get batch data
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=bool(args.mixed_precision)):
                if processor is not None and hasattr(processor, "preprocess"):
                    # HuggingFace processor
                    inputs = processor(images=images, return_tensors="pt").to(device)
                    outputs = model(**inputs, input_masks=masks.unsqueeze(1))
                    loss = outputs.loss
                else:
                    # Original SAM2 or SAM
                    image_embeddings = model.image_encoder(images)
                    loss = 0
                    
                    # Process each image in the batch individually
                    for i in range(images.shape[0]):
                        # Get sparse mask prompt representation
                        sparse_embeddings, dense_embeddings = model.prompt_encoder(
                            points=None,
                            boxes=None,
                            masks=masks[i:i+1].unsqueeze(1)  # [1, 1, H, W]
                        )
                        
                        # Predict masks
                        mask_predictions, _ = model.mask_decoder(
                            image_embeddings=image_embeddings[i:i+1],
                            image_pe=model.prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sparse_embeddings,
                            dense_prompt_embeddings=dense_embeddings,
                            multimask_output=False,
                        )
                        
                        # Compute loss - Binary cross-entropy loss
                        pred_masks = mask_predictions
                        target_masks = masks[i:i+1].float()
                        
                        # Compute BCE loss
                        bce_loss = F.binary_cross_entropy_with_logits(
                            pred_masks, target_masks, reduction="mean"
                        )
                        
                        # Compute Dice loss
                        pred_flat = torch.sigmoid(pred_masks).flatten()
                        target_flat = target_masks.flatten()
                        intersection = (pred_flat * target_flat).sum()
                        dice_loss = 1 - (2. * intersection) / (
                            pred_flat.sum() + target_flat.sum() + 1e-8
                        )
                        
                        # Combined loss
                        loss += bce_loss + dice_loss
                    
                    # Average loss over batch
                    loss = loss / images.shape[0]
            
            # Backward pass with gradient scaling if using mixed precision
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            # Update scheduler if using
            if scheduler is not None:
                scheduler.step()
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Accumulate loss for logging
            total_loss += loss.item()
            
            # Log every 50 steps
            if (step + 1) % args.log_every == 0:
                current_time = time.time()
                elapsed = current_time - last_log_time
                steps_per_sec = args.log_every / elapsed
                avg_loss = total_loss / args.log_every
                
                logger.info(
                    f"Step {step+1}/{args.max_train_steps} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Steps/sec: {steps_per_sec:.2f} | "
                    f"Elapsed: {(current_time - start_time) / 60:.1f} min"
                )
                
                # Reset metrics
                total_loss = 0.0
                last_log_time = current_time
            
            # Save checkpoint periodically
            if (step + 1) % args.save_steps == 0:
                save_checkpoint(model, optimizer, step + 1, args)
            
            # Run validation periodically
            if (step + 1) % args.eval_steps == 0:
                val_dice = validate(model, val_loader, args)
                logger.info(f"Validation Dice: {val_dice:.4f}")
                
                # Save best model
                if val_dice > best_val_dice:
                    best_val_dice = val_dice
                    save_checkpoint(model, optimizer, "best", args)
                    logger.info(f"New best model with Dice: {val_dice:.4f}")
            
            # Increment step
            step += 1
            
            # Check if we've reached max steps
            if step >= args.max_train_steps:
                break
    
    # Save final model
    save_checkpoint(model, optimizer, "final", args)
    logger.info("Training complete!")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune SAM2 on BraTS dataset for brain tumor segmentation"
    )
    
    parser.add_argument("--brats_path", type=str, required=True,
                      help="Path to BraTS dataset with MRI and segmentation files")
    parser.add_argument("--output_dir", type=str, default="version3/models/sam2_finetuned",
                      help="Directory to save fine-tuned model")
    
    # Model parameters
    parser.add_argument("--sam_checkpoint", type=str, default="sam2_hiera_small.pt",
                      help="Path to SAM2 checkpoint (for local implementation)")
    parser.add_argument("--sam_config", type=str, default="sam2_hiera_s.yaml",
                      help="Path to SAM2 config file (for local implementation)")
    parser.add_argument("--sam_model_id", type=str, default="facebook/sam2",
                      help="SAM2 model ID (for transformers implementation)")
    parser.add_argument("--train_image_encoder", action="store_true",
                      help="Whether to train the image encoder (requires more GPU memory)")
    
    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                      help="Learning rate for training")
    parser.add_argument("--weight_decay", type=float, default=4e-5,
                      help="Weight decay for optimizer")
    parser.add_argument("--max_train_steps", type=int, default=5000,
                      help="Maximum number of training steps")
    parser.add_argument("--logging_steps", type=int, default=100,
                      help="Log training metrics every N steps")
    parser.add_argument("--save_steps", type=int, default=1000,
                      help="Save checkpoint every N steps")
    parser.add_argument("--image_size", type=int, default=1024,
                      help="Maximum image size for training")
    parser.add_argument("--slice_axis", type=int, default=2,
                      help="Axis for slicing 3D volumes (0, 1, or 2)")
    parser.add_argument("--augmentation", action="store_true",
                      help="Apply data augmentation during training")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility")
    parser.add_argument("--do_validation", action="store_true",
                      help="Perform validation during training")
    
    return parser.parse_args()

def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.error("CUDA is required for SAM2 fine-tuning")
        return 1
    
    logger.info(f"Using device: {torch.cuda.get_device_name(0)}")
    logger.info(f"Training SAM2 on BraTS dataset: {args.brats_path}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure file handler for logging
    file_handler = logging.FileHandler(Path(args.output_dir) / "training.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    
    try:
        # Train SAM2 model
        if SAM2_SOURCE == "GITHUB_SAM2" or SAM2_SOURCE == "GITHUB_SAM":
            logger.info("Using local SAM2 implementation for training")
            train_model(args)
        else:
            logger.info("Using HuggingFace transformers implementation for training")
            train_model(args)
        
        logger.info(f"Training completed successfully. Model saved to {args.output_dir}")
        return 0
    
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 