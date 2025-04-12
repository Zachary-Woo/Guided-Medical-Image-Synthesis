#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ControlNet Utility Functions
This module provides helper functions for preparing and processing MRI data for ControlNet
training and inference.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import json
from pathlib import Path
from PIL import Image
import random
from tqdm import tqdm
import logging
from typing import Dict, List, Optional, Tuple, Any
import time

from .mri_utils import (
    load_nifti, 
    normalize_intensity, 
    extract_brain, 
    extract_slice, 
    mri_to_pil,
    resize_slice,
    overlay_segmentation
)

logger = logging.getLogger(__name__)

class ControlNetMRIDataset(Dataset):
    """
    Dataset for ControlNet training on MRI data with masks.
    
    Handles both standard image files (PNG/JPG) and NIfTI volumes for training
    ControlNet models on brain MRI data.
    """
    def __init__(
        self,
        data_root: str,
        tokenizer,
        image_size: int = 512,
        center_crop: bool = True, 
        random_flip: bool = False,
        max_token_length: int = 77,
        metadata_file: str = "controlnet_train.jsonl",
        use_augmentation: bool = False,
        slice_selection: str = "random"  # "random", "center", or "all"
    ):
        """
        Initialize the ControlNet MRI dataset.
        
        Args:
            data_root: Root directory of the dataset
            tokenizer: Tokenizer for text prompts
            image_size: Size of output images
            center_crop: Whether to center crop images
            random_flip: Whether to randomly flip images horizontally
            max_token_length: Maximum token length for prompts
            metadata_file: Name of the metadata file (JSONL format)
            use_augmentation: Whether to apply data augmentations
            slice_selection: Method for selecting slices from NIfTI volumes
        """
        self.data_root = Path(data_root)
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.max_token_length = max_token_length
        self.use_augmentation = use_augmentation
        self.slice_selection = slice_selection
        
        # Load metadata
        self.metadata = []
        
        # Try to find the specified metadata file or fallback to metadata.jsonl
        metadata_path = self.data_root / metadata_file
        if not metadata_path.exists():
            metadata_path = self.data_root / "metadata.jsonl"
            
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, "r") as f:
            for line in f:
                item = json.loads(line)
                # Ensure it has the required keys - handle different format variations
                if "conditioning_image" in item and "image" in item:
                    self.metadata.append(item)
                elif "mask_path" in item and "image_path" in item:
                    # Convert to expected format
                    self.metadata.append({
                        "conditioning_image": item["mask_path"],
                        "image": item["image_path"],
                        "prompt": item.get("prompt", "Brain MRI scan")
                    })
        
        logger.info(f"Loaded {len(self.metadata)} examples from {data_root}")
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.metadata[idx]
        text = item.get("prompt", "Brain MRI scan")
        image_path = self.data_root / item["image"]
        mask_path = self.data_root / item["conditioning_image"]
        
        # Check if we're dealing with NIfTI files or standard images
        if str(image_path).endswith((".nii", ".nii.gz")):
            image, mask = self._load_nifti_pair(image_path, mask_path)
        else:
            image, mask = self._load_image_pair(image_path, mask_path)
        
        # Apply transformations
        image, mask = self._apply_transforms(image, mask)
        
        # Tokenize text
        input_ids = self.tokenizer(
            text,
            max_length=self.max_token_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids[0]
        
        return {
            "input_ids": input_ids,
            "pixel_values": image,
            "conditioning_pixel_values": mask,
            "text": text
        }
    
    def _load_image_pair(self, image_path: Path, mask_path: Path) -> Tuple[Image.Image, Image.Image]:
        """Load an image and its corresponding mask as PIL images."""
        # Load and process image
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")
        
        return image, mask
    
    def _load_nifti_pair(self, image_path: Path, mask_path: Path) -> Tuple[Image.Image, Image.Image]:
        """Load a NIfTI volume and its corresponding mask, extract a slice, and convert to PIL images."""
        # Load NIfTI volumes
        image_vol = load_nifti(image_path)
        mask_vol = load_nifti(mask_path)
        
        # Process the MRI data
        image_vol = normalize_intensity(image_vol)
        brain_mask = extract_brain(image_vol)
        image_vol = image_vol * brain_mask  # Apply brain mask
        
        # Find slices with content
        valid_slices = self._find_valid_slices(image_vol, mask_vol)
        
        # Select a slice based on the specified method
        slice_idx = self._select_slice(valid_slices, mask_vol)
        
        # Extract the selected slices
        image_slice = extract_slice(image_vol, slice_idx=slice_idx)
        mask_slice = extract_slice(mask_vol, slice_idx=slice_idx)
        
        # Ensure all tumor classes are represented in a single channel
        condition_image = np.zeros_like(mask_slice)
        if np.max(mask_slice) > 0:
            condition_image[mask_slice > 0] = 1
        
        # Convert to PIL images
        image_pil = mri_to_pil(image_slice)
        
        # Create a colorized mask for better conditioning
        mask_pil = mri_to_pil(condition_image)
        
        return image_pil, mask_pil
    
    def _find_valid_slices(self, image_vol: np.ndarray, mask_vol: np.ndarray) -> List[int]:
        """Find slices that contain relevant content in both the image and mask."""
        valid_slices = []
        
        for i in range(image_vol.shape[2]):
            image_slice = image_vol[:, :, i]
            mask_slice = mask_vol[:, :, i] if mask_vol.shape[2] > i else np.zeros_like(image_slice)
            
            # Check if slice has sufficient brain content
            if np.sum(image_slice > 0.1) > 100:  # At least 100 voxels with intensity > 0.1
                valid_slices.append(i)
        
        if not valid_slices:
            # Fallback to middle slices if no valid slices found
            valid_slices = list(range(
                max(0, image_vol.shape[2]//2 - 5),
                min(image_vol.shape[2], image_vol.shape[2]//2 + 5)
            ))
            
        return valid_slices
    
    def _select_slice(self, valid_slices: List[int], mask_vol: np.ndarray) -> int:
        """Select a slice based on the specified selection method."""
        if self.slice_selection == "center":
            # Select the middle valid slice
            return valid_slices[len(valid_slices) // 2]
        
        elif self.slice_selection == "random":
            # For tumor slices, give higher probability to slices with tumor
            tumor_slices = []
            for idx in valid_slices:
                if idx < mask_vol.shape[2] and np.sum(mask_vol[:, :, idx]) > 0:
                    tumor_slices.append(idx)
            
            # 80% chance to select a tumor slice if available
            if tumor_slices and random.random() < 0.8:
                return random.choice(tumor_slices)
            else:
                return random.choice(valid_slices)
        
        else:  # "all" or any other value
            # Just return a random slice (will be handled differently in a data loader)
            return random.choice(valid_slices)
    
    def _apply_transforms(self, image: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply transformations to the image and mask."""
        # Apply center crop if enabled
        if self.center_crop:
            # Calculate the center crop dimensions
            w, h = image.size
            min_dim = min(w, h)
            left = (w - min_dim) // 2
            top = (h - min_dim) // 2
            right = left + min_dim
            bottom = top + min_dim
            image = image.crop((left, top, right, bottom))
            mask = mask.crop((left, top, right, bottom))
        
        # Resize to target size
        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        
        # Apply random horizontal flip with 50% probability
        if self.random_flip and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Convert to tensors
        image_array = np.array(image).astype(np.float32) / 255.0
        mask_array = np.array(mask).astype(np.float32) / 255.0
        
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask_array).permute(2, 0, 1)
        
        return image_tensor, mask_tensor

def prepare_controlnet_dataset(
    source_dirs: List[str],
    output_dir: str,
    condition_source: str = "mask",  # "mask", "segmentation", or "canny_edge"
    slice_axis: int = 2,
    modality: str = "t1ce",
    train_ratio: float = 0.8
) -> Tuple[int, int]:
    """
    Prepare a dataset for ControlNet training by processing MRI volumes and masks.
    
    Args:
        source_dirs: List of directories containing MRI data
        output_dir: Directory to save the processed dataset
        condition_source: Source of conditioning signal
        slice_axis: Axis for slice extraction (0=sagittal, 1=coronal, 2=axial)
        modality: MRI modality to use
        train_ratio: Ratio of data to use for training vs. validation
        
    Returns:
        Tuple of (number of training examples, number of validation examples)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create image and condition directories
    image_dir = output_dir / "images"
    condition_dir = output_dir / "conditions"
    image_dir.mkdir(exist_ok=True)
    condition_dir.mkdir(exist_ok=True)
    
    # Find all subjects with MRI scans and segmentations
    subjects = []
    
    for source_dir in source_dirs:
        source_path = Path(source_dir)
        
        # Find all subject directories
        for subject_dir in source_path.glob("*"):
            if not subject_dir.is_dir():
                continue
                
            # Check if required files exist
            mri_file = list(subject_dir.glob(f"*{modality}.nii.gz"))
            seg_file = list(subject_dir.glob("*seg.nii.gz"))
            
            if mri_file and seg_file:
                subjects.append({
                    "subject_id": subject_dir.name,
                    "mri_path": mri_file[0],
                    "seg_path": seg_file[0]
                })
    
    logger.info(f"Found {len(subjects)} subjects with MRI and segmentation data")
    
    # Process each subject
    metadata = []
    
    for subject in tqdm(subjects, desc="Processing subjects"):
        # Load MRI and segmentation
        mri_data = load_nifti(subject["mri_path"])
        seg_data = load_nifti(subject["seg_path"])
        
        # Normalize MRI
        mri_data = normalize_intensity(mri_data)
        
        # Find valid slices with content
        valid_slices = []
        
        for i in range(mri_data.shape[slice_axis]):
            # Extract slices
            if slice_axis == 0:
                mri_slice = mri_data[i, :, :]
                seg_slice = seg_data[i, :, :]
            elif slice_axis == 1:
                mri_slice = mri_data[:, i, :]
                seg_slice = seg_data[:, i, :]
            else:  # slice_axis == 2
                mri_slice = mri_data[:, :, i]
                seg_slice = seg_data[:, :, i]
            
            # Check if slice has enough content
            if np.sum(mri_slice > 0.1) > 100:
                valid_slices.append(i)
        
        # Process valid slices
        for slice_idx in valid_slices:
            # Extract slices
            if slice_axis == 0:
                mri_slice = mri_data[slice_idx, :, :]
                seg_slice = seg_data[slice_idx, :, :]
            elif slice_axis == 1:
                mri_slice = mri_data[:, slice_idx, :]
                seg_slice = seg_data[:, slice_idx, :]
            else:  # slice_axis == 2
                mri_slice = mri_data[:, :, slice_idx]
                seg_slice = seg_data[:, :, slice_idx]
            
            # Resize to standard size
            mri_slice = resize_slice(mri_slice, (512, 512))
            seg_slice = resize_slice(seg_slice, (512, 512), order=0)
            
            # Create conditioning image based on specified source
            if condition_source == "mask":
                # Binary mask: 1 for any tumor, 0 for background
                condition_slice = np.zeros_like(seg_slice)
                condition_slice[seg_slice > 0] = 1
                condition_pil = mri_to_pil(condition_slice)
            
            elif condition_source == "segmentation":
                # Multi-class segmentation visualization
                # Convert to RGB
                condition_slice = overlay_segmentation(
                    np.zeros_like(seg_slice),
                    seg_slice.astype(np.int32),
                    alpha=1.0
                )
                condition_pil = Image.fromarray(condition_slice)
            
            elif condition_source == "canny_edge":
                # Edge detection
                from skimage import feature
                edges = feature.canny(mri_slice, sigma=1)
                edges = edges.astype(np.float32)
                condition_pil = mri_to_pil(edges)
            
            else:
                raise ValueError(f"Unknown condition source: {condition_source}")
            
            # Convert MRI to PIL image
            mri_pil = mri_to_pil(mri_slice)
            
            # Save images
            img_filename = f"{subject['subject_id']}_{modality}_slice{slice_idx:03d}.png"
            cond_filename = f"{subject['subject_id']}_{modality}_slice{slice_idx:03d}_cond.png"
            
            mri_pil.save(image_dir / img_filename)
            condition_pil.save(condition_dir / cond_filename)
            
            # Add to metadata
            has_tumor = np.any(seg_slice > 0)
            
            # Generate a descriptive prompt
            if has_tumor:
                # Determine tumor types present (simplified)
                tumor_types = []
                if np.any(seg_slice == 1):  # Necrotic
                    tumor_types.append("necrotic tumor core")
                if np.any(seg_slice == 2):  # Edema
                    tumor_types.append("peritumoral edema")
                if np.any(seg_slice == 3):  # Enhancing
                    tumor_types.append("enhancing tumor")
                
                tumor_desc = " and ".join(tumor_types)
                prompt = f"MRI {modality} scan of brain with {tumor_desc}"
            else:
                prompt = f"MRI {modality} scan of normal brain"
            
            metadata.append({
                "image": f"images/{img_filename}",
                "conditioning_image": f"conditions/{cond_filename}",
                "prompt": prompt,
                "has_tumor": has_tumor,
                "subject_id": subject['subject_id'],
                "slice_idx": slice_idx
            })
    
    # Split into train and validation sets
    random.shuffle(metadata)
    split_idx = int(len(metadata) * train_ratio)
    train_data = metadata[:split_idx]
    val_data = metadata[split_idx:]
    
    # Save metadata files
    with open(output_dir / "controlnet_train.jsonl", "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")
    
    with open(output_dir / "controlnet_val.jsonl", "w") as f:
        for item in val_data:
            f.write(json.dumps(item) + "\n")
    
    logger.info(f"Created dataset with {len(train_data)} training and {len(val_data)} validation examples")
    
    return len(train_data), len(val_data)

def generate_controlnet_samples(
    controlnet,
    pipeline,
    conditioning_image,
    num_samples: int = 4,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None,
    prompt: str = "MRI scan of brain with tumor",
    negative_prompt: str = "low quality, blurry, distorted, pixelated, artifact",
    controlnet_conditioning_scale: float = 1.0,
    output_dir: Optional[str] = None
) -> List[Image.Image]:
    """
    Generate samples using a trained ControlNet model.
    
    Args:
        controlnet: Trained ControlNet model
        pipeline: Diffusion pipeline
        conditioning_image: Conditioning image (PIL Image or path to image)
        num_samples: Number of samples to generate
        num_inference_steps: Number of denoising steps
        guidance_scale: Classifier-free guidance scale
        seed: Random seed for reproducibility
        prompt: Text prompt for generation
        negative_prompt: Negative prompt for guidance
        controlnet_conditioning_scale: Scale of ControlNet conditioning
        output_dir: Directory to save generated images
        
    Returns:
        List of generated PIL images
    """
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        generator = torch.Generator(pipeline.device).manual_seed(seed)
    else:
        generator = None
    
    # Load conditioning image if it's a path
    if isinstance(conditioning_image, str) or isinstance(conditioning_image, Path):
        conditioning_image = Image.open(conditioning_image).convert("RGB")
    
    # Resize conditioning image to 512x512
    conditioning_image = conditioning_image.resize((512, 512))
    
    # Generate samples
    images = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=conditioning_image,
        num_images_per_prompt=num_samples,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        generator=generator
    ).images
    
    # Save images if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = int(time.time())
        
        for i, img in enumerate(images):
            img.save(os.path.join(output_dir, f"sample_{timestamp}_{i+1}.png"))
    
    return images

def save_controlnet_checkpoint(
    controlnet,
    output_dir: str,
    filename: str = "controlnet-model",
    push_to_hub: bool = False,
    repo_id: Optional[str] = None
):
    """
    Save a trained ControlNet model checkpoint.
    
    Args:
        controlnet: Trained ControlNet model
        output_dir: Directory to save the checkpoint
        filename: Base filename for the checkpoint
        push_to_hub: Whether to push the model to the Hugging Face Hub
        repo_id: Repository ID for pushing to Hub
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the model
    if push_to_hub and repo_id:
        controlnet.save_pretrained(
            os.path.join(output_dir, filename),
            push_to_hub=True,
            repo_id=repo_id
        )
    else:
        controlnet.save_pretrained(os.path.join(output_dir, filename))
    
    logger.info(f"Saved ControlNet checkpoint to {output_dir}/{filename}") 