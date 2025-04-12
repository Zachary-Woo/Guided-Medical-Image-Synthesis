#!/usr/bin/env python
"""
MRI Generation Script for Brain Tumor MRI Synthesis

This script generates anatomically plausible brain MRI images with tumors
in specified locations using a combination of LoRA (for style) and
ControlNet (for structural control).
"""

import argparse
import time
import torch
import numpy as np
import logging
from pathlib import Path
import json
from PIL import Image, ImageDraw
import cv2
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate anatomically accurate brain MRI images"
    )
    parser.add_argument("--prompt", type=str, required=True,
                      help="Text prompt for MRI generation")
    parser.add_argument("--mask", type=str,
                      help="Optional path to binary tumor mask (white=tumor)")
    parser.add_argument("--output_dir", type=str, default="output/version3/generated",
                      help="Output directory")
    parser.add_argument("--lora_weights", type=str, default="version3/models/mri_lora",
                      help="Path to LoRA weights directory")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility")
    parser.add_argument("--num_inference_steps", type=int, default=30,
                      help="Number of diffusion steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                      help="Classifier-free guidance scale")
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=1.0,
                      help="ControlNet conditioning scale")
    parser.add_argument("--create_mask", action="store_true",
                      help="Automatically create a tumor mask based on prompt")
    
    return parser.parse_args()

def create_output_dir(base_dir):
    """Create a uniquely timestamped output directory."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = Path(base_dir) / f"mri_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def setup_pipeline(args):
    """Set up the generation pipeline with LoRA and ControlNet."""
    logger.info("Setting up generation pipeline...")
    
    # Check for CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    if device == "cpu":
        logger.warning("CUDA not available. Generation will be slow on CPU!")
    
    # Load controlnet model
    logger.info("Loading ControlNet model...")
    controlnet_path = "lllyasviel/sd-controlnet-seg"  # Using segmentation ControlNet as the base
    try:
        controlnet = ControlNetModel.from_pretrained(
            controlnet_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
    except Exception as e:
        logger.error(f"Error loading ControlNet: {e}")
        # Fall back to Canny ControlNet if segmentation ControlNet fails
        logger.info("Falling back to Canny ControlNet...")
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
    
    # Load base diffusion model (SD2-1 is more modern than SD1-5)
    logger.info("Loading base Stable Diffusion model...")
    try:
        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            controlnet=controlnet,
            safety_checker=None,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
    except Exception as e:
        logger.error(f"Error loading SD 2.1: {e}")
        logger.info("Falling back to SD 1.5...")
        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            safety_checker=None,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
    
    # Use UniPC scheduler for faster and high-quality sampling
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    
    # Move to device
    pipeline = pipeline.to(device)
    
    # Load LoRA weights if available
    lora_path = Path(args.lora_weights)
    if lora_path.exists():
        logger.info(f"Loading LoRA weights from {lora_path}")
        try:
            pipeline.load_lora_weights(args.lora_weights)
            logger.info("LoRA weights loaded successfully")
        except Exception as e:
            logger.error(f"Error loading LoRA weights: {e}")
            logger.warning("Proceeding without LoRA - results may lack domain-specific appearance")
    else:
        logger.warning(f"LoRA weights not found at {lora_path}")
        logger.warning("Proceeding without LoRA - results may lack domain-specific appearance")
    
    # Enable optimizations
    if device == "cuda":
        logger.info("Enabling GPU memory optimizations...")
        pipeline.enable_attention_slicing()
        try:
            pipeline.enable_xformers_memory_efficient_attention()
            logger.info("xFormers optimization enabled")
        except Exception as e:
            logger.warning(f"Could not enable xFormers optimization: {e}")
    
    return pipeline

def extract_tumor_location(prompt):
    """Extract tumor location from the prompt for mask generation."""
    prompt = prompt.lower()
    # Brain regions to look for in the prompt
    regions = {
        "frontal": (150, 120),
        "temporal": (150, 180),
        "left temporal": (100, 180),
        "right temporal": (200, 180),
        "left frontal": (100, 120),
        "right frontal": (200, 120),
        "parietal": (150, 150),
        "left parietal": (100, 150),
        "right parietal": (200, 150), 
        "occipital": (150, 200),
        "left occipital": (100, 200),
        "right occipital": (200, 200),
        "cerebellum": (150, 230),
        "left cerebellum": (100, 230),
        "right cerebellum": (200, 230),
        "brain stem": (150, 250),
        "thalamus": (150, 170),
        "left thalamus": (130, 170),
        "right thalamus": (170, 170),
        "basal ganglia": (150, 160),
        "left basal ganglia": (130, 160),
        "right basal ganglia": (170, 160),
    }
    
    # Default location (center of image)
    location = (150, 150)
    radius = 30  # Default tumor size
    
    # Check for size hints
    if "small" in prompt:
        radius = 15
    elif "large" in prompt:
        radius = 45
    
    # Find region match
    for region, coords in regions.items():
        if region in prompt:
            location = coords
            logger.info(f"Detected tumor in {region} region at coordinates {coords}")
            break
    
    return location, radius

def create_tumor_mask(prompt, image_size=512):
    """Create a synthetic tumor mask based on the tumor location in the prompt."""
    logger.info(f"Creating tumor mask based on prompt: {prompt}")
    
    # Create blank mask
    mask = Image.new("L", (image_size, image_size), 0)
    draw = ImageDraw.Draw(mask)
    
    # Extract location and size from prompt
    location, radius = extract_tumor_location(prompt)
    
    # Scale coordinates to image size
    x, y = location
    x = int(x * image_size / 300)
    y = int(y * image_size / 300)
    radius = int(radius * image_size / 300)
    
    # Draw tumor as white circle (you could use more complex shapes here)
    draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=255)
    
    # Optional: add some noise/irregularity to the tumor edge 
    # to make it look more realistic
    mask_np = np.array(mask)
    mask_np = cv2.GaussianBlur(mask_np, (15, 15), 5)
    
    # Optional: Add some random variations
    # noise = np.random.normal(0, 10, mask_np.shape).astype(np.uint8)
    # mask_np = np.clip(mask_np + noise, 0, 255)
    
    return Image.fromarray(mask_np)

def prepare_condition_image(mask_path=None, prompt=None, create_mask=False):
    """Prepare the condition image for ControlNet."""
    if mask_path:
        logger.info(f"Using provided mask: {mask_path}")
        mask = Image.open(mask_path).convert("L")
        # Resize to 512x512 if needed
        if mask.width != 512 or mask.height != 512:
            mask = mask.resize((512, 512), Image.LANCZOS)
    elif create_mask and prompt:
        logger.info("Creating mask from prompt text")
        mask = create_tumor_mask(prompt)
    else:
        logger.warning("No mask provided or created. Using an empty mask.")
        mask = Image.new("L", (512, 512), 0)  # Blank mask
    
    # Convert to RGB for ControlNet input
    condition_image = Image.new("RGB", (512, 512), (0, 0, 0))
    # Set tumor area to green (for segmentation controlnet)
    mask_np = np.array(mask)
    condition_np = np.array(condition_image)
    condition_np[mask_np > 127, 1] = 255  # Green channel for tumor
    
    # Add a light gray outline for the brain
    # This helps guide the controlnet to form a brain shape
    brain_outline = Image.new("L", (512, 512), 0)
    draw = ImageDraw.Draw(brain_outline)
    draw.ellipse((50, 50, 462, 462), outline=128, width=3)  # Brain outline
    
    # Add the brain outline to red channel
    brain_np = np.array(brain_outline)
    condition_np[brain_np > 0, 0] = 128  # Light red for brain outline
    
    return Image.fromarray(condition_np)

def generate_mri(pipeline, args, condition_image):
    """Generate the MRI image using the pipeline."""
    logger.info("Generating MRI image...")
    
    # Prepare guidance scale
    guidance_scale = args.guidance_scale
    
    # Set seed for reproducibility
    generator = torch.manual_seed(args.seed)
    
    # Enhance the prompt with MRI-specific language
    prompt = args.prompt
    if "MRI" not in prompt and "mri" not in prompt:
        prompt = f"MRI scan: {prompt}"
        
    # Add detail enhancement to the prompt if not present
    detail_terms = ["high-resolution", "high-detail", "clear", "medical"]
    if not any(term in prompt.lower() for term in detail_terms):
        prompt = f"High-resolution {prompt}, clear medical imaging"
    
    # Negative prompt to avoid common artifacts
    negative_prompt = "blurry, distorted, low quality, low resolution, noise, grainy, text, watermark, signature, deformed anatomy, extra structures"
    
    # Start generation timer
    start_time = time.time()
    
    # Generate image
    result = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=condition_image,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        generator=generator,
    )
    
    # End timer
    generation_time = time.time() - start_time
    logger.info(f"Generation completed in {generation_time:.2f} seconds")
    
    return result.images[0], generation_time

def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory
    output_dir = create_output_dir(args.output_dir)
    logger.info(f"Output directory: {output_dir}")
    
    # Configure file handler for logging
    file_handler = logging.FileHandler(output_dir / "generation.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    
    # Setup pipeline
    pipeline = setup_pipeline(args)
    
    # Prepare condition image
    condition_image = prepare_condition_image(
        mask_path=args.mask,
        prompt=args.prompt,
        create_mask=args.create_mask
    )
    
    # Save condition image
    condition_image.save(output_dir / "condition_mask.png")
    
    # Generate MRI
    mri_image, generation_time = generate_mri(pipeline, args, condition_image)
    
    # Save generated image
    mri_image.save(output_dir / "generated_mri.png")
    
    # Save metadata
    metadata = {
        "prompt": args.prompt,
        "seed": args.seed,
        "inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "controlnet_conditioning_scale": args.controlnet_conditioning_scale,
        "generation_time": generation_time,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mask_provided": args.mask is not None,
        "mask_created": args.create_mask,
        "cuda_available": torch.cuda.is_available()
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    
    logger.info(f"Generation complete. Results saved to {output_dir}")
    print(f"\nGenerated MRI saved to: {output_dir / 'generated_mri.png'}")

if __name__ == "__main__":
    main() 