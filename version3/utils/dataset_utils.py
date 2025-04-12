import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import random
from typing import List, Dict, Tuple, Optional, Union
import albumentations as A
from pathlib import Path
import json
import nibabel as nib
from tqdm import tqdm
import cv2

def get_transforms(
    resize: Optional[Tuple[int, int]] = None,
    augment: bool = False, 
    augment_prob: float = 0.5
) -> A.Compose:
    """
    Get image transforms for MRI images.P
    
    Args:
        resize: Target size for resizing
        augment: Whether to apply augmentations
        augment_prob: Probability of applying augmentations
        
    Returns:
        Albumentations transforms
    """
    transforms_list = []
    
    # Add resize transform if specified
    if resize is not None:
        transforms_list.append(A.Resize(height=resize[0], width=resize[1]))
    
    # Add augmentations if specified
    if augment:
        transforms_list.extend([
            A.OneOf([
                A.RandomBrightnessContrast(p=augment_prob),
                A.RandomGamma(p=augment_prob),
            ], p=augment_prob),
            
            A.OneOf([
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=augment_prob),
                A.GridDistortion(p=augment_prob),
                A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=augment_prob),
            ], p=augment_prob),
            
            A.OneOf([
                A.GaussNoise(p=augment_prob),
                A.MultiplicativeNoise(p=augment_prob),
            ], p=augment_prob),
            
            A.RandomRotate90(p=augment_prob),
            A.Flip(p=augment_prob),
            A.ShiftScaleRotate(
                shift_limit=0.1, 
                scale_limit=0.2, 
                rotate_limit=30, 
                p=augment_prob,
                border_mode=cv2.BORDER_CONSTANT
            ),
        ])
    
    return A.Compose(transforms_list)

class MRIDataset(Dataset):
    """
    Dataset for MRI images with optional segmentation masks.
    """
    def __init__(
        self,
        image_dir: str,
        mask_dir: Optional[str] = None,
        transform: Optional[A.Compose] = None,
        image_size: Tuple[int, int] = (512, 512),
        prompt_template: str = "MRI scan of {}",
        anatomical_regions: List[str] = ["brain"],
        modalities: List[str] = ["T1", "T2", "FLAIR"],
        conditions: List[str] = ["normal", "tumor"],
        image_extension: str = ".png",
        mask_extension: str = ".png"
    ):
        """
        Initialize MRIDataset.
        
        Args:
            image_dir: Directory containing images
            mask_dir: Directory containing masks (optional)
            transform: Albumentations transforms to apply
            image_size: Target image size
            prompt_template: Template for generating prompts
            anatomical_regions: List of anatomical regions
            modalities: List of MRI modalities
            conditions: List of possible medical conditions
            image_extension: File extension for images
            mask_extension: File extension for masks
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.transform = transform
        self.image_size = image_size
        self.prompt_template = prompt_template
        self.anatomical_regions = anatomical_regions
        self.modalities = modalities
        self.conditions = conditions
        self.image_extension = image_extension
        self.mask_extension = mask_extension
        
        # Get all image files
        self.image_files = sorted([
            f for f in os.listdir(self.image_dir) 
            if f.endswith(self.image_extension)
        ])
        
        # Create mapping of image file to corresponding mask file
        self.mask_files = {}
        if self.mask_dir:
            for img_file in self.image_files:
                # Remove extension and add mask extension
                mask_file = img_file.replace(self.image_extension, self.mask_extension)
                
                # Check if mask file exists
                if os.path.exists(self.mask_dir / mask_file):
                    self.mask_files[img_file] = mask_file
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        img_file = self.image_files[idx]
        img_path = self.image_dir / img_file
        
        # Load image
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)
        
        # Load mask if available
        if img_file in self.mask_files and self.mask_dir:
            mask_file = self.mask_files[img_file]
            mask_path = self.mask_dir / mask_file
            mask = Image.open(mask_path).convert("L")
            mask = np.array(mask)
        else:
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]
        
        # Generate random prompt
        region = random.choice(self.anatomical_regions)
        modality = random.choice(self.modalities)
        condition = random.choice(self.conditions)
        
        prompt = self.prompt_template.format(
            f"{modality} {region} with {condition}"
        )
        
        # Convert to tensors
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float() / 255.0
        
        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "prompt": prompt,
            "file_name": img_file
        }

def save_dataset_metadata(
    output_dir: str,
    image_files: List[str],
    mask_files: Optional[Dict[str, str]] = None,
    split_ratios: Dict[str, float] = {"train": 0.8, "val": 0.1, "test": 0.1},
    seed: int = 42
) -> Dict[str, List[str]]:
    """
    Save dataset metadata and create train/val/test splits.
    
    Args:
        output_dir: Directory to save metadata
        image_files: List of image files
        mask_files: Dictionary mapping image files to mask files
        split_ratios: Ratios for train/val/test splits
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with train/val/test splits
    """
    # Set random seed for reproducible splits
    random.seed(seed)
    
    # Shuffle image files
    shuffled_files = image_files.copy()
    random.shuffle(shuffled_files)
    
    # Calculate split sizes
    total = len(shuffled_files)
    train_size = int(total * split_ratios["train"])
    val_size = int(total * split_ratios["val"])
    
    # Create splits
    train_files = shuffled_files[:train_size]
    val_files = shuffled_files[train_size:train_size + val_size]
    test_files = shuffled_files[train_size + val_size:]
    
    # Create splits dictionary
    splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }
    
    # Create metadata
    metadata = {
        "splits": splits,
        "mask_files": mask_files if mask_files else {},
        "total_images": total,
        "split_sizes": {
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files)
        }
    }
    
    # Save metadata
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    return splits

def convert_nifti_to_slices(
    nifti_dir: str,
    output_dir: str,
    slice_dim: int = 2,
    output_format: str = "png",
    min_slice_content: float = 0.01,
    normalize: bool = True
) -> List[str]:
    """
    Convert NIfTI volumes to 2D slices.
    
    Args:
        nifti_dir: Directory containing NIfTI files
        output_dir: Directory to save output slices
        slice_dim: Dimension to slice (0, 1, or 2)
        output_format: Output format (png or jpg)
        min_slice_content: Minimum percentage of non-zero voxels required
        normalize: Whether to normalize slice intensities
        
    Returns:
        List of output slice files
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all NIfTI files
    nifti_files = [f for f in os.listdir(nifti_dir) if f.endswith((".nii", ".nii.gz"))]
    output_files = []
    
    for nifti_file in tqdm(nifti_files, desc="Converting NIfTI files"):
        # Load NIfTI
        nifti_path = os.path.join(nifti_dir, nifti_file)
        nifti_img = nib.load(nifti_path)
        img_data = nifti_img.get_fdata()
        
        # Get number of slices
        num_slices = img_data.shape[slice_dim]
        
        for i in range(num_slices):
            # Extract slice
            if slice_dim == 0:
                img_slice = img_data[i, :, :]
            elif slice_dim == 1:
                img_slice = img_data[:, i, :]
            else:
                img_slice = img_data[:, :, i]
                
            # Skip slices with too little content
            non_zero_percentage = np.count_nonzero(img_slice) / img_slice.size
            if non_zero_percentage < min_slice_content:
                continue
                
            # Normalize if requested
            if normalize:
                if img_slice.min() != img_slice.max():
                    img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())
                else:
                    continue
            
            # Scale to 0-255 and convert to uint8
            img_slice = (img_slice * 255).astype(np.uint8)
            
            # Save slice
            base_name = os.path.splitext(nifti_file)[0]
            if base_name.endswith('.nii'):
                base_name = os.path.splitext(base_name)[0]
                
            output_file = f"{base_name}_slice{i:03d}.{output_format}"
            output_path = output_dir / output_file
            
            # Convert to PIL and save
            img_pil = Image.fromarray(img_slice)
            img_pil.save(output_path)
            
            output_files.append(output_file)
    
    return output_files

def create_dataloaders(
    dataset: Dataset,
    batch_size: int = 4,
    num_workers: int = 4,
    split_ratios: Dict[str, float] = {"train": 0.8, "val": 0.1, "test": 0.1},
    seed: int = 42
) -> Dict[str, DataLoader]:
    """
    Create train/val/test dataloaders from a dataset.
    
    Args:
        dataset: Dataset to split
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for dataloaders
        split_ratios: Ratios for train/val/test splits
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with train/val/test dataloaders
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(total_size * split_ratios["train"])
    val_size = int(total_size * split_ratios["val"])
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    } 