#!/usr/bin/env python
"""
BraTS Dataset Preparation Script

This script processes the BraTS dataset to extract axial slices from
MRI volumes along with corresponding segmentation masks.
It generates paired data suitable for training both LoRA and ControlNet models.
"""

import sys
import time
import logging
import argparse
import json
import random
from pathlib import Path
import numpy as np
import nibabel as nib
from PIL import Image
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare BraTS dataset for training"
    )
    parser.add_argument("--brats_path", type=str, required=True,
                      help="Path to BraTS dataset root")
    parser.add_argument("--output_dir", type=str, default="data/processed_brats",
                      help="Output directory for processed data")
    parser.add_argument("--sample_count", type=int, default=1000,
                      help="Target number of slices to extract")
    parser.add_argument("--modality", type=str, default="t1",
                      choices=["t1", "t2", "flair", "t1ce"],
                      help="MRI modality to extract")
    parser.add_argument("--image_size", type=int, default=512,
                      help="Target image size")
    parser.add_argument("--include_healthy", action="store_true",
                      help="Include healthy (no tumor) slices")
    
    return parser.parse_args()

def setup_directories(output_dir):
    """Create the necessary output directories."""
    output_dir = Path(output_dir)
    
    # Main directories
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    
    # Create directories
    for dir_path in [images_dir, masks_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return {
        "root": output_dir,
        "images": images_dir,
        "masks": masks_dir
    }

def find_brats_patients(brats_path):
    """Find all patient folders in the BraTS dataset."""
    brats_path = Path(brats_path)
    
    # Check if path exists
    if not brats_path.exists():
        raise FileNotFoundError(f"BraTS path not found: {brats_path}")
    
    # Directly look for patient folders inside the provided path
    patient_dirs = [d for d in brats_path.glob("*") if d.is_dir()]
    
    # Make sure we have directories with expected files
    valid_patients = []
    for patient_dir in patient_dirs:
        has_t1 = any(patient_dir.glob("*t1.nii*")) or any(patient_dir.glob("*T1.nii*"))
        
        # Only check if at least one T1 file exists for now
        if has_t1: # Removed 'and has_seg'
            valid_patients.append(patient_dir)
    
    if not valid_patients:
        raise ValueError(f"No valid BraTS patient directories found in {brats_path}")
    
    logger.info(f"Found {len(valid_patients)} valid patient directories")
    return valid_patients

def find_modality_file(patient_dir, modality):
    """Find the file for the requested modality in the patient directory."""
    # Map command-line modality to possible file patterns
    modality_patterns = {
        "t1": ["*t1.nii.gz", "*T1.nii.gz"],
        "t2": ["*t2.nii.gz", "*T2.nii.gz"],
        "flair": ["*flair.nii.gz", "*FLAIR.nii.gz"],
        "t1ce": ["*t1ce.nii.gz", "*T1ce.nii.gz", "*T1CE.nii.gz", "*T1GD.nii.gz"]
    }
    
    # Find segmentation file
    seg_patterns = ["*seg.nii.gz", "*_seg.nii.gz", "*mask.nii.gz", "*_mask.nii.gz"]
    seg_file = None
    for pattern in seg_patterns:
        seg_files = list(patient_dir.glob(pattern))
        if seg_files:
            seg_file = seg_files[0]
            break
    
    # Find modality file
    mod_file = None
    for pattern in modality_patterns[modality]:
        mod_files = list(patient_dir.glob(pattern))
        if mod_files:
            mod_file = mod_files[0]
            break
    
    return mod_file, seg_file

def extract_brain_region(image):
    """
    Simple brain extraction: set background (very low values) to 0.
    For proper brain extraction, tools like FSL BET would be better.
    """
    # Simple threshold to separate brain from background
    mask = image > np.percentile(image, 1)
    # Apply mask
    extracted = image.copy()
    extracted[~mask] = 0
    return extracted

def normalize_intensity(image):
    """Normalize image intensity to [0, 1] range."""
    # Clip outliers (typical in MRI)
    p1, p99 = np.percentile(image[image > 0], (1, 99))
    image = np.clip(image, p1, p99)
    
    # Normalize to [0, 1]
    image = (image - p1) / (p99 - p1)
    image = np.clip(image, 0, 1)
    
    return image

def map_tumor_region(label_value):
    """
    Map the BraTS segmentation label value to a description of the tumor region.
    
    In BraTS:
    - Label 1: Necrotic tumor core (NCR) and non-enhancing tumor (NET)
    - Label 2: Peritumoral edema (ED)
    - Label 4: Enhancing tumor (ET)
    
    Returns a tuple of (region_name, is_core)
    """
    if label_value == 1:
        return "necrotic core", True
    elif label_value == 2:
        return "edema", False
    elif label_value == 4:
        return "enhancing tumor", True
    else:
        return "undefined", False

def determine_tumor_location(seg_slice, slice_shape):
    """
    Determine the anatomical location of the tumor in the slice.
    Returns a string description like "left temporal" or "right frontal".
    """
    # Find center of mass of the tumor
    if np.sum(seg_slice > 0) == 0:
        return "none"  # No tumor in slice
    
    # Get center of mass coordinates
    y_indices, x_indices = np.where(seg_slice > 0)
    center_y, center_x = np.mean(y_indices), np.mean(x_indices)
    
    # Normalize coordinates to [0, 1]
    h, w = slice_shape
    norm_y, norm_x = center_y / h, center_x / w
    
    # Determine left/right
    if norm_x < 0.45:
        side = "left"
    elif norm_x > 0.55:
        side = "right"
    else:
        side = "midline"
    
    # Determine anatomical region (approximate based on y-coordinate)
    if norm_y < 0.35:
        region = "frontal"
    elif norm_y < 0.5:
        region = "parietal"
    elif norm_y < 0.7:
        region = "temporal"
    else:
        region = "occipital"
    
    if side == "midline":
        return region
    else:
        return f"{side} {region}"

def determine_tumor_size(seg_slice):
    """Determine the tumor size category based on its area in the slice."""
    tumor_area = np.sum(seg_slice > 0)
    if tumor_area == 0:
        return "none"
    
    total_area = seg_slice.shape[0] * seg_slice.shape[1]
    ratio = tumor_area / total_area
    
    if ratio < 0.02:
        return "small"
    elif ratio < 0.07:
        return "medium"
    else:
        return "large"

def generate_prompt(modality, tumor_location, tumor_size, has_tumor=True):
    """Generate a descriptive prompt for the image."""
    # Modality description
    modality_map = {
        "t1": "T1-weighted",
        "t2": "T2-weighted",
        "flair": "FLAIR",
        "t1ce": "T1-weighted post-contrast"
    }
    
    base_prompt = f"{modality_map[modality]} axial brain MRI scan"
    
    if not has_tumor:
        return f"{base_prompt} of normal brain, no abnormalities"
    
    if tumor_size == "none":
        return f"{base_prompt} of normal brain, no tumor"
    
    # Build prompt with tumor description
    prompt = f"{base_prompt} with {tumor_size} tumor in {tumor_location} region"
    
    # Add random details sometimes for variety
    details = [
        ", showing clear tumor boundaries",
        ", with surrounding edema",
        ", with good contrast between tissues",
        ", high-resolution clinical scan",
        ", showing detailed brain structures"
    ]
    
    if random.random() < 0.5:
        prompt += random.choice(details)
    
    return prompt

def create_tumor_mask(seg_slice):
    """
    Create a binary tumor mask from the BraTS segmentation labels.
    
    In BraTS, labels are:
    - 0: background/normal brain
    - 1: necrotic tumor core (NCR) and non-enhancing tumor (NET)
    - 2: peritumoral edema (ED)
    - 4: enhancing tumor (ET)
    
    For our task, we'll create a binary mask where:
    - 0: background/normal brain
    - 1: any tumor tissue (labels 1, 2, or 4)
    """
    binary_mask = (seg_slice > 0).astype(np.uint8) * 255
    return binary_mask

def process_patient(patient_dir, modality, output_dirs, image_size, include_healthy=False):
    """Process a single patient's scans."""
    # Find modality and segmentation files
    mod_file, seg_file = find_modality_file(patient_dir, modality)
    
    if mod_file is None or seg_file is None:
        logger.warning(f"Missing required files for patient {patient_dir.name}")
        return []
    
    # Load the nifti files
    mod_nifti = nib.load(mod_file)
    seg_nifti = nib.load(seg_file)
    
    # Get image data
    mod_data = mod_nifti.get_fdata()
    seg_data = seg_nifti.get_fdata()
    
    # Make sure segmentation is aligned with image data
    if mod_data.shape != seg_data.shape:
        logger.warning(f"Shape mismatch for patient {patient_dir.name}: {mod_data.shape} vs {seg_data.shape}")
        return []
    
    # Extract axial slices (standard BraTS orientation is (x, y, z))
    processed_slices = []
    
    # Iterate through slices
    for z_idx in range(mod_data.shape[2]):
        mod_slice = mod_data[:, :, z_idx]
        seg_slice = seg_data[:, :, z_idx]
        
        # Check if slice has tumor
        has_tumor = np.any(seg_slice > 0)
        
        # Skip healthy slices if not including them
        if not has_tumor and not include_healthy:
            continue
        
        # Skip slices with only background (no brain)
        if np.sum(mod_slice > 0) < mod_slice.size * 0.05:
            continue
        
        # Brain extraction
        mod_slice = extract_brain_region(mod_slice)
        
        # Normalize intensity
        mod_slice = normalize_intensity(mod_slice)
        
        # Convert to 8-bit for image saving
        mod_slice_8bit = (mod_slice * 255).astype(np.uint8)
        
        # Create binary tumor mask
        binary_mask = create_tumor_mask(seg_slice)
        
        # Determine tumor properties
        tumor_location = determine_tumor_location(seg_slice, mod_slice.shape)
        tumor_size = determine_tumor_size(seg_slice)
        
        # Generate prompt
        prompt = generate_prompt(modality, tumor_location, tumor_size, has_tumor)
        
        # Create PIL images
        slice_img = Image.fromarray(mod_slice_8bit).convert("L")
        mask_img = Image.fromarray(binary_mask).convert("L")
        
        # Resize to target size
        slice_img = slice_img.resize((image_size, image_size), Image.LANCZOS)
        mask_img = mask_img.resize((image_size, image_size), Image.NEAREST)
        
        # Generate unique filename
        timestamp = int(time.time() * 1000)
        random_id = random.randint(1000, 9999)
        filename_base = f"{patient_dir.name}_{z_idx}_{timestamp}_{random_id}"
        
        # Save the image and mask
        slice_path = output_dirs["images"] / f"{filename_base}.png"
        mask_path = output_dirs["masks"] / f"{filename_base}.png"
        
        slice_img.save(slice_path)
        mask_img.save(mask_path)
        
        # Record slice information
        slice_info = {
            "patient_id": patient_dir.name,
            "slice_idx": z_idx,
            "modality": modality,
            "has_tumor": bool(has_tumor),
            "tumor_location": tumor_location,
            "tumor_size": tumor_size,
            "image_path": str(slice_path.relative_to(output_dirs["root"])),
            "mask_path": str(mask_path.relative_to(output_dirs["root"])),
            "prompt": prompt
        }
        
        processed_slices.append(slice_info)
    
    return processed_slices

def prepare_metadata(processed_data, output_dir):
    """Prepare metadata files for training."""
    output_dir = Path(output_dir)
    
    # Save full metadata JSON
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(processed_data, f, indent=2)
    
    # Create text file with prompts for LoRA training
    with open(output_dir / "prompts.txt", "w") as f:
        for item in processed_data:
            f.write(f"{item['image_path']}\t{item['prompt']}\n")
    
    # Create JSONL file for Diffusers LoRA training
    with open(output_dir / "metadata.jsonl", "w") as f:
        for item in processed_data:
            entry = {
                "file_name": item["image_path"],
                "text": item["prompt"]
            }
            f.write(json.dumps(entry) + "\n")
    
    # Create training data splits (80% train, 20% validation)
    random.shuffle(processed_data)
    split_idx = int(len(processed_data) * 0.8)
    train_data = processed_data[:split_idx]
    val_data = processed_data[split_idx:]
    
    # Save split metadata
    with open(output_dir / "train_metadata.json", "w") as f:
        json.dump(train_data, f, indent=2)
    
    with open(output_dir / "val_metadata.json", "w") as f:
        json.dump(val_data, f, indent=2)
    
    # Save ControlNet training data format
    with open(output_dir / "controlnet_train.jsonl", "w") as f:
        for item in train_data:
            entry = {
                "image": item["image_path"],
                "conditioning_image": item["mask_path"],
                "prompt": item["prompt"]
            }
            f.write(json.dumps(entry) + "\n")
    
    with open(output_dir / "controlnet_val.jsonl", "w") as f:
        for item in val_data:
            entry = {
                "image": item["image_path"],
                "conditioning_image": item["mask_path"],
                "prompt": item["prompt"]
            }
            f.write(json.dumps(entry) + "\n")
    
    # Create LoRA config (for Diffusers)
    config = {
        "model_name_or_path": "stabilityai/stable-diffusion-2-1",
        "resolution": 512,
        "train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "mixed_precision": "fp16",
        "learning_rate": 1e-4,
        "lr_scheduler": "cosine",
        "lr_warmup_steps": 100,
        "rank": 4,
        "max_train_steps": 5000,
        "checkpointing_steps": 500,
        "seed": 42
    }
    
    with open(output_dir / "lora_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Metadata prepared with {len(train_data)} training and {len(val_data)} validation samples")

def main():
    """Main function."""
    args = parse_args()
    
    # Setup output directories
    output_dirs = setup_directories(args.output_dir)
    
    # Configure file handler for logging
    log_file = output_dirs["root"] / "preparation.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    
    logger.info("Starting BraTS data preparation")
    logger.info(f"BraTS path: {args.brats_path}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Modality: {args.modality}")
    logger.info(f"Target image size: {args.image_size}")
    logger.info(f"Include healthy slices: {args.include_healthy}")
    
    try:
        # Find BraTS patients
        patient_dirs = find_brats_patients(args.brats_path)
        
        # Sample a subset of patients if needed
        num_patients = len(patient_dirs)
        avg_slices_per_patient = 120  # Rough estimate based on typical MRI volumes
        
        if args.sample_count < num_patients * avg_slices_per_patient:
            patients_needed = max(3, args.sample_count // avg_slices_per_patient)
            if patients_needed < num_patients:
                logger.info(f"Randomly sampling {patients_needed} patients from {num_patients}")
                patient_dirs = random.sample(patient_dirs, patients_needed)
        
        # Process patients (potentially in parallel)
        all_processed_slices = []
        
        # Use ProcessPoolExecutor for parallel processing
        max_workers = max(1, multiprocessing.cpu_count() - 1)
        logger.info(f"Processing patients with {max_workers} parallel workers")
        
        with tqdm(total=len(patient_dirs), desc="Processing patients") as pbar:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all jobs
                future_to_patient = {
                    executor.submit(
                        process_patient, 
                        patient_dir, 
                        args.modality, 
                        output_dirs, 
                        args.image_size, 
                        args.include_healthy
                    ): patient_dir for patient_dir in patient_dirs
                }
                
                # Process results as they complete
                for future in as_completed(future_to_patient):
                    patient_dir = future_to_patient[future]
                    try:
                        patient_slices = future.result()
                        all_processed_slices.extend(patient_slices)
                        logger.info(f"Processed {len(patient_slices)} slices from {patient_dir.name}")
                    except Exception as e:
                        logger.error(f"Error processing {patient_dir.name}: {e}")
                    
                    pbar.update(1)
        
        # Sample from processed slices if we have more than requested
        if len(all_processed_slices) > args.sample_count:
            logger.info(f"Sampling {args.sample_count} slices from {len(all_processed_slices)} processed slices")
            all_processed_slices = random.sample(all_processed_slices, args.sample_count)
        
        # Prepare metadata files
        prepare_metadata(all_processed_slices, output_dirs["root"])
        
        logger.info(f"Data preparation complete. Processed {len(all_processed_slices)} slices.")
        print(f"\nData preparation complete. Processed {len(all_processed_slices)} slices.")
        print(f"Results saved to: {output_dirs['root']}")
    
    except Exception as e:
        logger.error(f"Error during data preparation: {e}", exc_info=True)
        print(f"Error during data preparation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 