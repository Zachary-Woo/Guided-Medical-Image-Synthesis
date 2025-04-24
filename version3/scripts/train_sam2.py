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
import torch.nn.functional as F
import cv2
import random
import nibabel as nib
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
    load_nifti,
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
    
    def __init__(self, subjects, slice_axis=2, image_size=1024, batch_size=1, augmentation=True):
        """
        Initialize BraTS data loader.
        
        Args:
            subjects: List of subject dictionaries (from find_brats_data)
            slice_axis: Axis for slice extraction (0, 1, or 2)
            image_size: Maximum size for resizing images
            batch_size: Number of images per batch (currently processes one image at a time)
            augmentation: Whether to apply data augmentation
        """
        self.subjects = subjects
        self.slice_axis = slice_axis
        self.image_size = image_size
        self.batch_size = batch_size # Store batch size
        self.augmentation = augmentation
        
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
    Load the SAM model.
    Prioritizes: local checkpoint -> local segment_anything_2 -> local segment_anything -> HuggingFace ID.
    
    Args:
        args: Command line arguments including checkpoint path, model ID, etc.
        
    Returns:
        model: The loaded SAM/SAM2 model
        processor: Image processor for the model (or predictor for local SAM)
        resume_step: Step to resume training from (0 if not resuming)
    """
    logger.info("Loading SAM model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resume_step = 0
    model = None
    processor = None

    # 1. Try loading from checkpoint
    checkpoint_path = args.sam_checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Attempting to load checkpoint from {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model_dict = checkpoint.get('model_state_dict', checkpoint)
            resume_step = checkpoint.get('step', 0)
            if resume_step > 0:
                 logger.info(f"Resuming from step {resume_step}")

            # Try to load as SAM2 first
            try:
                import segment_anything_2
                logger.info("Attempting to load checkpoint as segment_anything_2 model...")
                # Assuming default vit_l for checkpoint loading if config missing
                model = segment_anything_2.build_sam2_vit_l() # Add configuration loading if needed
                model.load_state_dict(model_dict, strict=False)
                # SAM2 doesn't have a separate processor, handled by model interaction
                processor = None
                logger.info("Successfully loaded checkpoint as SAM2 model.")
            except Exception as e1:
                logger.warning(f"Failed to load checkpoint as SAM2: {e1}. Trying as original SAM...")
                # Try loading as original SAM
                try:
                    import segment_anything
                    logger.info("Attempting to load checkpoint as segment_anything model...")
                    # Assuming default vit_h for checkpoint loading
                    model = segment_anything.sam_model_registry["vit_h"]()
                    model.load_state_dict(model_dict, strict=False)
                    # Original SAM uses SamPredictor
                    from segment_anything import SamPredictor
                    processor = SamPredictor(model)
                    logger.info("Successfully loaded checkpoint as SAM model.")
                except Exception as e2:
                    logger.error(f"Failed to load checkpoint as either SAM2 or SAM: {e2}")
                    model = None # Ensure model is None if loading failed
        except Exception as e:
            logger.error(f"Error loading checkpoint file {checkpoint_path}: {e}")
            model = None
    
    # 2. If no model loaded from checkpoint, try local installations
    if model is None:
        logger.info("No valid checkpoint found or loaded. Trying local installations...")
        try:
            # Try SAM2 first
            import segment_anything_2
            logger.info("Loading default SAM2 ViT-L model from local installation.")
            # Add configuration loading logic if needed based on args.sam_config
            model = segment_anything_2.build_sam2_vit_l()
            processor = None
            logger.info("Successfully loaded local SAM2 model.")
        except ImportError:
            logger.info("segment_anything_2 not found locally. Trying original segment_anything...")
            # Fall back to original SAM
            try:
                import segment_anything
                logger.info("Loading default SAM ViT-H model from local installation.")
                model = segment_anything.sam_model_registry["vit_h"]()
                from segment_anything import SamPredictor
                processor = SamPredictor(model)
                logger.info("Successfully loaded local SAM model.")
            except ImportError:
                logger.warning("Neither segment_anything_2 nor segment_anything found locally.")
                model = None
                processor = None

    # 3. If still no model, try HuggingFace ID
    if model is None:
        logger.info("No local SAM/SAM2 model loaded. Trying HuggingFace ID...")
        if args.sam_model_id:
            try:
                logger.info(f"Loading SAM model from HuggingFace: {args.sam_model_id}")
                # Use original SAM classes from transformers for compatibility
                from transformers import SamModel, SamProcessor
                model = SamModel.from_pretrained(args.sam_model_id)
                processor = SamProcessor.from_pretrained(args.sam_model_id)
                logger.info(f"Successfully loaded {args.sam_model_id} from HuggingFace.")
            except Exception as e:
                logger.error(f"Failed to load from HuggingFace ID {args.sam_model_id}: {e}")
                model = None # Ensure model is None
        else:
            logger.warning("No HuggingFace Model ID provided.")

    # Final check: If no model could be loaded, raise error
    if model is None:
        raise RuntimeError("Failed to load SAM model from checkpoint, local installation, or HuggingFace Hub.")

    # Move model to device
    model = model.to(device)

    # Freeze image encoder if specified
    if not args.train_image_encoder:
        # Check for common attribute names for the image encoder
        encoder_attr = None
        if hasattr(model, 'image_encoder'):
            encoder_attr = 'image_encoder'
        elif hasattr(model, 'vision_encoder'): # Some HF models use this
             encoder_attr = 'vision_encoder'
        
        if encoder_attr:
            logger.info(f"Freezing {encoder_attr} parameters")
            for param in getattr(model, encoder_attr).parameters():
                param.requires_grad = False
        else:
             logger.warning("Could not find image encoder attribute to freeze.")

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

def validate(model, processor, valid_loader, args):
    """Evaluate the model on validation data."""
    model.eval()
    total_iou = 0.0
    count = 0
    device = next(model.parameters()).device

    logger.info(f"Running validation for max {args.max_eval_steps} steps...")
    with torch.no_grad():
        for step in range(args.max_eval_steps):
            batch_data = valid_loader.get_batch()
            if batch_data is None:
                logger.warning("Validation loader returned None, skipping batch.")
                continue # Skip if no valid data could be loaded

            image, gt_masks, points, labels = batch_data
            
            # Convert numpy arrays to tensors and move to device
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(device) # Add batch dim
            gt_masks_tensor = torch.from_numpy(gt_masks).unsqueeze(1).float().to(device) # Add channel dim
            points_tensor = torch.from_numpy(points).float().to(device)
            labels_tensor = torch.from_numpy(labels).float().to(device)
            
            # Prepare inputs for the model (handle different processor types)
            if processor is not None and hasattr(processor, "preprocess"):
                 # HuggingFace Processor
                 inputs = processor(images=image_tensor, input_points=points_tensor, input_labels=labels_tensor, return_tensors="pt").to(device)
                 image_embeddings = model.get_image_embeddings(inputs["pixel_values"])
                 sparse_embeddings, dense_embeddings = model.get_prompt_embeddings(
                    input_points=inputs["input_points"],
                    input_labels=inputs["input_labels"],
                    input_boxes=None # Assuming no box prompts for validation
                 )
            elif processor is not None and isinstance(processor, segment_anything.SamPredictor):
                 # Original SAM Predictor
                 processor.set_image(image) # Predictor expects numpy HWC
                 image_embeddings = processor.get_image_embedding().to(device)
                 # Original SAM expects points in format [[x,y], [x,y]], labels [1, 0]
                 input_points = points_tensor.squeeze(1).cpu().numpy() # Remove batch dim for predictor
                 input_labels = labels_tensor.squeeze(1).cpu().numpy()
                 sparse_embeddings, dense_embeddings = model.prompt_encoder(
                     points=(torch.as_tensor(input_points, device=device).unsqueeze(1), torch.as_tensor(input_labels, device=device).unsqueeze(1)),
                     boxes=None,
                     masks=None,
                 )
            else:
                # SAM2 local model or unknown
                logger.warning("Processor type not recognized or None, attempting generic forward pass.")
                image_embeddings = model.image_encoder(image_tensor)
                sparse_embeddings, dense_embeddings = model.prompt_encoder(
                    points=(points_tensor, labels_tensor),
                    boxes=None,
                    masks=None,
                )

            # Predict masks
            if hasattr(model, 'mask_decoder'):
                mask_predictions, iou_predictions = model.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False, # Get single best mask
                )
            else:
                 logger.error("Model does not have a mask_decoder attribute.")
                 continue # Skip batch if model structure is wrong

            # Calculate IoU
            pred_masks_prob = torch.sigmoid(mask_predictions)
            pred_masks_binary = (pred_masks_prob > 0.5).float()
            
            # Ensure masks have the same shape [B, 1, H, W]
            # gt_masks_tensor should be [B, NumInstances, H, W], need to handle B=1, NumInstances > 1
            # For validation, let's assume we take the union of GT masks if multiple exist
            gt_mask_union = (gt_masks_tensor.sum(dim=1, keepdim=True) > 0).float()
            
            # Resize GT mask to match prediction size for IoU calculation
            target_masks_resized = F.interpolate(
                gt_mask_union,
                size=pred_masks_binary.shape[-2:], # Get H, W from prediction
                mode='nearest'
            )
            
            # Calculate intersection and union
            intersection = (pred_masks_binary * target_masks_resized).sum(dim=(1, 2, 3))
            union = (pred_masks_binary + target_masks_resized).sum(dim=(1, 2, 3)) - intersection
            iou = (intersection / (union + 1e-6)).mean() # Mean over batch (should be 1)
            
            total_iou += iou.item()
            count += 1

    avg_iou = total_iou / count if count > 0 else 0.0
    logger.info(f"Validation complete. Average IoU: {avg_iou:.4f} over {count} steps.")
    return avg_iou

def train_model(args):
    """Train the SAM2 model on BraTS dataset."""
    # Set up logger - Logging is already configured in main, so this isn't needed here
    # setup_logging(args.output_dir) # Removed this line
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
        subjects=train_subjects,
        slice_axis=args.slice_axis,
        image_size=args.image_size,
        batch_size=args.batch_size,
        augmentation=args.augmentation
    )
    
    val_loader = BraTSDataLoader(
        subjects=val_subjects,
        slice_axis=args.slice_axis,
        image_size=args.image_size, 
        batch_size=args.batch_size,
        augmentation=False
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
    # Use scaler only if mixed precision is fp16 or bf16
    scaler = torch.amp.GradScaler('cuda', enabled=(args.mixed_precision in ['fp16', 'bf16']))
    
    # For tracking metrics
    best_val_iou = 0.0 # Changed from best_val_dice
    total_loss = 0.0
    step = resume_step
    
    # Start timer
    start_time = time.time()
    last_log_time = start_time
    
    logger.info(f"Starting training from step {resume_step}")
    
    # Training loop
    while step < args.max_train_steps:
        model.train()
        
        # Explicitly get batch data
        batch_data = train_loader.get_batch()
        if batch_data is None:
            logger.warning("Train loader returned None, skipping step.")
            continue # Skip if no valid data could be loaded
            
        image, gt_masks, points, labels = batch_data
        
        # Skip steps if resuming (do this after getting batch to keep counts consistent)
        if step < resume_step:
            step += 1
            continue
        
        # Convert numpy arrays to tensors and move to device
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(device)
        gt_masks_tensor = torch.from_numpy(gt_masks).unsqueeze(1).float().to(device) # Add channel dim
        points_tensor = torch.from_numpy(points).float().to(device)
        labels_tensor = torch.from_numpy(labels).float().to(device)
        
        # Zero gradients before forward pass
        optimizer.zero_grad()

        # Forward pass with mixed precision
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16 if args.mixed_precision == 'fp16' else torch.bfloat16, enabled=(args.mixed_precision in ['fp16', 'bf16'])):
            loss = 0
            # Prepare inputs for the model (handle different processor types)
            if processor is not None and hasattr(processor, "preprocess"):
                 # HuggingFace Processor - needs image, points, labels
                 inputs = processor(images=image_tensor, input_points=points_tensor, input_labels=labels_tensor, return_tensors="pt").to(device)
                 # We need to compute loss manually here as HF SAM doesn't return loss directly with point prompts
                 image_embeddings = model.get_image_embeddings(inputs["pixel_values"])
                 sparse_embeddings, dense_embeddings = model.get_prompt_embeddings(
                    input_points=inputs["input_points"],
                    input_labels=inputs["input_labels"],
                    input_boxes=None
                 )
            elif processor is not None and isinstance(processor, segment_anything.SamPredictor):
                 # Original SAM Predictor
                 # Important: Predictor works instance by instance, not batched
                 # This training loop assumes batch_size=1 from the loader
                 if image_tensor.shape[0] != 1:
                     logger.error("Training loop assumes batch_size=1 for original SAM Predictor")
                     continue
                 
                 processor.set_image(image) # Expects numpy HWC
                 image_embeddings = processor.get_image_embedding().to(device)
                 # Original SAM expects points in format [[x,y]], labels [1]
                 input_points = points_tensor.squeeze(1).cpu().numpy() # Instance points
                 input_labels = labels_tensor.squeeze(1).cpu().numpy() # Instance labels
                 
                 sparse_embeddings, dense_embeddings = model.prompt_encoder(
                     points=(torch.as_tensor(input_points, device=device).unsqueeze(1), torch.as_tensor(input_labels, device=device).unsqueeze(1)),
                     boxes=None,
                     masks=None,
                 )
            else:
                # SAM2 local model or unknown
                logger.warning("Processor type not recognized or None, attempting generic forward pass.")
                image_embeddings = model.image_encoder(image_tensor)
                sparse_embeddings, dense_embeddings = model.prompt_encoder(
                    points=(points_tensor, labels_tensor),
                    boxes=None,
                    masks=None,
                )
            
            # Predict masks - common step
            if hasattr(model, 'mask_decoder'):
                mask_predictions, iou_predictions = model.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False, # Get single best mask
                )
            else:
                 logger.error("Model does not have a mask_decoder attribute.")
                 continue # Skip batch if model structure is wrong
            
            # Compute loss - Binary cross-entropy + Dice loss
            # Ensure gt_masks_tensor has the same shape as pred_masks: [B, 1, H, W]
            # gt_masks_tensor is [B, NumInstances, H, W], take union for loss
            gt_mask_union = (gt_masks_tensor.sum(dim=1, keepdim=True) > 0).float()

            pred_masks = mask_predictions
            # Resize GT mask to match prediction size for loss calculation
            target_masks_resized = F.interpolate(
                gt_mask_union,
                size=pred_masks.shape[-2:], # Get H, W from prediction
                mode='nearest'
            )
            
            # Compute BCE loss
            bce_loss = F.binary_cross_entropy_with_logits(
                pred_masks, target_masks_resized, reduction="mean"
            )
            
            # Compute Dice loss
            pred_flat = torch.sigmoid(pred_masks).flatten(1)
            target_flat = target_masks_resized.flatten(1)
            intersection = (pred_flat * target_flat).sum(1)
            dice_loss = 1 - (2. * intersection) / (
                pred_flat.sum(1) + target_flat.sum(1) + 1e-8
            )
            dice_loss = dice_loss.mean() # Average over batch
            
            # Combined loss
            loss = bce_loss + dice_loss

        # Backward pass with gradient scaling if using mixed precision
        if scaler:
            scaler.scale(loss).backward()
            # Optional: Gradient clipping
            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            # Optional: Gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Update scheduler if using
        if scheduler is not None:
            scheduler.step()
        
        # Accumulate loss for logging
        total_loss += loss.item()
        
        # Log every N steps
        if (step + 1) % args.log_every == 0:
            current_time = time.time()
            elapsed = current_time - last_log_time
            steps_per_sec = args.log_every / elapsed if elapsed > 0 else 0
            avg_loss = total_loss / args.log_every
            
            logger.info(
                f"Step {step+1}/{args.max_train_steps} | "
                f"Loss: {avg_loss:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                f"Steps/sec: {steps_per_sec:.2f} | "
                f"Elapsed: {(current_time - start_time) / 60:.1f} min"
            )
            
            # Reset metrics
            total_loss = 0.0
            last_log_time = current_time
        
        # Save checkpoint periodically
        if (step + 1) % args.save_steps == 0:
            save_checkpoint(model, optimizer, step + 1, args)
        
        # Run validation periodically if enabled
        if args.do_validation and (step + 1) % args.eval_steps == 0:
            val_iou = validate(model, processor, val_loader, args)
            model.train() # Switch back to train mode after validation
            
            # Save best model based on validation IoU
            if val_iou > best_val_iou:
                best_val_iou = val_iou
                save_checkpoint(model, optimizer, "best", args)
                logger.info(f"New best model with IoU: {val_iou:.4f}")
        
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
    parser.add_argument("--sam_checkpoint", type=str, default=None,
                      help="Path to local SAM/SAM2 checkpoint (will be tried first)")
    parser.add_argument("--sam_config", type=str, default=None,
                      help="Path to SAM2 config file (used if loading local checkpoint)")
    parser.add_argument("--sam_model_id", type=str, default="facebook/sam-vit-large",
                      help="HuggingFace SAM model ID (fallback if no checkpoint/local install)")
    parser.add_argument("--train_image_encoder", action="store_true",
                      help="Whether to train the image encoder (requires more GPU memory)")
    
    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                      help="Learning rate for training")
    parser.add_argument("--weight_decay", type=float, default=4e-5,
                      help="Weight decay for optimizer")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["cosine", "constant"],
                      help="Learning rate scheduler type")
    parser.add_argument("--min_learning_rate", type=float, default=1e-6,
                      help="Minimum learning rate for cosine scheduler")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"],
                      help="Enable mixed precision training (fp16 or bf16)")
    parser.add_argument("--max_train_steps", type=int, default=5000,
                      help="Maximum number of training steps")
    parser.add_argument("--log_every", type=int, default=50, # Renamed from logging_steps for clarity
                      help="Log training metrics every N steps")
    parser.add_argument("--save_steps", type=int, default=1000,
                      help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=500,
                      help="Run validation every N steps")
    parser.add_argument("--max_eval_steps", type=int, default=50,
                      help="Maximum number of batches to use for validation")
    parser.add_argument("--image_size", type=int, default=1024,
                      help="Maximum image size for training")
    parser.add_argument("--batch_size", type=int, default=1,
                      help="Batch size for training (per GPU)")
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