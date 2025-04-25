#!/usr/bin/env python
"""
Enhanced ControlNet generation script optimized for MedMNIST images
"""

import sys
import torch
import argparse
import logging
from PIL import Image
import traceback
from pathlib import Path
import numpy as np
import cv2
import json
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate medical images with ControlNet")
    parser.add_argument("--condition_image", type=str, required=True,
                        help="Path to conditioning image")
    parser.add_argument("--prompt", type=str, 
                        default="High-resolution histopathology slide showing colorectal tissue with clear cellular detail, H&E stain, medical scan",
                        help="Text prompt for generation")
    parser.add_argument("--negative_prompt", type=str,
                        default="blurry, low quality, low resolution, deformed, distorted, watermark, text, bad anatomy",
                        help="Negative prompt to guide generation")
    parser.add_argument("--output_dir", type=str, default="output/enhanced_controlnet",
                        help="Base output directory (will be auto-incremented)")
    parser.add_argument("--steps", type=int, default=50, 
                        help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=9.0,
                        help="Guidance scale")
    parser.add_argument("--input_size", type=int, default=64,
                        help="Original image size")
    parser.add_argument("--output_size", type=int, default=512,
                        help="Output image size")
    parser.add_argument("--low_threshold", type=int, default=50,
                        help="Canny edge detection low threshold")
    parser.add_argument("--high_threshold", type=int, default=150,
                        help="Canny edge detection high threshold")
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=0.8,
                        help="ControlNet conditioning scale")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--debug", action="store_true", 
                        help="Enable debug logging")
    return parser.parse_args()

def create_canny_edge_from_medical_image(image_path, low_threshold=50, high_threshold=150, target_size=512):
    """Process a medical image for ControlNet Canny conditioning"""
    # Load the image
    original_image = Image.open(image_path).convert("RGB")
    original_np = np.array(original_image)
    
    if original_np.shape[0] < target_size or original_np.shape[1] < target_size:
        # First upscale with better algorithm for small images
        resized_np = cv2.resize(original_np, (target_size, target_size), 
                               interpolation=cv2.INTER_LANCZOS4)
    else:
        resized_np = cv2.resize(original_np, (target_size, target_size))
    
    # Convert to grayscale for edge detection
    if len(resized_np.shape) == 3 and resized_np.shape[2] == 3:
        gray_np = cv2.cvtColor(resized_np, cv2.COLOR_RGB2GRAY)
    else:
        gray_np = resized_np
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray_np)
    
    # Apply multiple preprocessing techniques and combine results
    # 1. Standard Canny with moderate thresholds
    canny_standard = cv2.Canny(equalized, low_threshold, high_threshold)
    
    # 2. Gaussian blur followed by Canny with lower thresholds to catch more edges
    blurred = cv2.GaussianBlur(equalized, (3, 3), 0)
    canny_blurred = cv2.Canny(blurred, max(5, low_threshold//2), max(50, high_threshold//2))
    
    # 3. Sobel edge detection (captures more gradual transitions)
    sobelx = cv2.Sobel(equalized, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(equalized, cv2.CV_64F, 0, 1, ksize=3)
    sobel_edges = np.sqrt(sobelx**2 + sobely**2)
    sobel_edges = (sobel_edges > np.mean(sobel_edges) + 0.5 * np.std(sobel_edges)) * 255
    sobel_edges = sobel_edges.astype(np.uint8)
    
    # Combine edge maps (take maximum at each pixel)
    combined_edges = np.maximum(canny_standard, canny_blurred)
    combined_edges = np.maximum(combined_edges, sobel_edges)
    
    # Optional: Dilate edges slightly to enhance connectivity
    kernel = np.ones((2, 2), np.uint8)
    combined_edges = cv2.dilate(combined_edges, kernel, iterations=1)
    
    # Convert to RGB (required by ControlNet)
    canny_edges_rgb = np.stack([combined_edges] * 3, axis=2)
    
    # Create both the canny edge image and the resized original
    canny_image = Image.fromarray(canny_edges_rgb.astype(np.uint8))
    resized_image = Image.fromarray(resized_np.astype(np.uint8))
    
    return canny_image, resized_image

def setup_controlnet_pipeline():
    """Set up an optimized ControlNet pipeline for medical imaging"""
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load models
    base_model_id = "runwayml/stable-diffusion-v1-5"
    controlnet_model_id = "lllyasviel/sd-controlnet-canny"
    
    logger.info(f"Loading ControlNet model: {controlnet_model_id}")
    logger.info(f"Loading base model: {base_model_id}")
    
    # Set up parameters
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    
    # Load ControlNet
    controlnet = ControlNetModel.from_pretrained(
        controlnet_model_id, 
        torch_dtype=torch_dtype
    )
    
    # Load base model
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_id, 
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=torch_dtype
    )
    
    # Use faster scheduler (UniPC typically gives better results)
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    
    # Set up memory optimizations
    if device == "cuda":
        pipeline.enable_model_cpu_offload()
        
        try:
            pipeline.enable_attention_slicing()
            logger.info("Enabled attention slicing")
        except Exception as e:
            logger.warning(f"Could not enable attention slicing: {e}")
        
        try:
            pipeline.enable_xformers_memory_efficient_attention()
            logger.info("Enabled xFormers memory efficient attention")
        except Exception as e:
            logger.warning(f"Could not enable xFormers: {e}")
    else:
        pipeline = pipeline.to(device)
    
    logger.info("Pipeline set up complete")
    return pipeline

def enhance_medical_prompt(base_prompt, image_type="histopathology"):
    """Enhance a basic prompt with medical imaging specific details"""
    # Check if the prompt already has detailed medical terms
    medical_terms = ["histology", "pathology", "H&E", "stain", "microscopy", "histopathology", 
                   "medical scan", "biopsy", "specimen"]
    
    has_medical_terms = any(term.lower() in base_prompt.lower() for term in medical_terms)
    
    # Define specialized enhancements based on image type
    enhancements = {
        "histopathology": ", H&E stain, high-resolution microscopy, cellular detail, professional pathology scan, tissue section",
        "xray": ", high-contrast radiograph, professional medical imaging, clear bone detail, radiological scan",
        "ct": ", computed tomography, axial slice, clear tissue differentiation, professional medical scan",
        "mri": ", magnetic resonance imaging, T1-weighted, detailed anatomical scan, high tissue contrast",
        "ultrasound": ", sonogram, professional ultrasound imaging, clear tissue boundaries, diagnostic quality"
    }
    
    # Only enhance if the prompt doesn't already have specific medical terms
    if not has_medical_terms:
        enhancement = enhancements.get(image_type.lower(), ", medical quality, professional scan, high detail")
        return f"{base_prompt}{enhancement}"
    
    return base_prompt

def get_next_output_dir(base_dir, source_image_path):
    """
    Create a sequentially numbered output directory that includes source image info.
    
    Args:
        base_dir (str or Path): Base output directory path
        source_image_path (str or Path): Path to the source/conditioning image
        
    Returns:
        Path: Next available numbered directory
    """
    base_path = Path(base_dir)
    base_parent = base_path.parent
    base_name = base_path.name
    
    # Extract source image filename without extension
    source_filename = Path(source_image_path).stem
    
    # Create base name with source image info
    source_dir_name = f"{base_name}_{source_filename}"
    
    # Find all existing numbered directories for this source image
    existing_dirs = []
    for item in base_parent.glob(f"{source_dir_name}_*"):
        if item.is_dir():
            try:
                # Extract the number after the last underscore
                num = int(item.name.split('_')[-1])
                existing_dirs.append(num)
            except ValueError:
                # Skip directories that don't end with a number
                continue
    
    # Determine the next number
    next_num = 1
    if existing_dirs:
        next_num = max(existing_dirs) + 1
    
    # Create the new directory path with source image info and number
    next_dir = base_parent / f"{source_dir_name}_{next_num}"
    return next_dir

def main():
    args = parse_args()
    
    # Set up logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Get the next available numbered output directory with source image info
    output_dir = get_next_output_dir(args.output_dir, args.condition_image)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using output directory: {output_dir}")
    
    # Set up logging file
    log_file = output_dir / "generate.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(file_handler)
    
    logger.info(f"Logging to {log_file}")
    
    # Enhance the prompt with medical specific details
    prompt = enhance_medical_prompt(args.prompt, "histopathology")
    logger.info(f"Using enhanced prompt: {prompt}")
    
    try:
        # Process conditioning image
        logger.info(f"Processing conditioning image: {args.condition_image}")
        canny_image, resized_image = create_canny_edge_from_medical_image(
            args.condition_image, 
            low_threshold=args.low_threshold,
            high_threshold=args.high_threshold,
            target_size=args.output_size
        )
        
        # Save the processed images for reference
        canny_path = output_dir / "canny_edge.png"
        resized_path = output_dir / "resized_input.png"
        canny_image.save(canny_path)
        resized_image.save(resized_path)
        logger.info(f"Saved enhanced edge image to {canny_path}")
        logger.info(f"Saved resized input image to {resized_path}")
        
        # Set up pipeline
        pipeline = setup_controlnet_pipeline()
        
        # Generate image
        logger.info(f"Generating image with {args.steps} steps, guidance scale {args.guidance_scale}")
        
        # Set up generator for reproducibility
        if args.seed is not None:
            generator = torch.manual_seed(args.seed)
        else:
            generator = None
        
        controlnet_scale = args.controlnet_conditioning_scale
        
        # Use optimized parameters for medical images
        image = pipeline(
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            image=canny_image,
            num_inference_steps=args.steps,
            generator=generator,
            guidance_scale=args.guidance_scale,
            controlnet_conditioning_scale=controlnet_scale
        ).images[0]
        
        # Save generated image
        output_path = output_dir / "generated.png"
        image.save(output_path)
        logger.info(f"Image saved to {output_path}")
        
        # Save metadata
        metadata = {
            "prompt": prompt,
            "negative_prompt": args.negative_prompt,
            "conditioning_image": str(args.condition_image),
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "controlnet_conditioning_scale": controlnet_scale,
            "seed": args.seed,
            "low_threshold": args.low_threshold,
            "high_threshold": args.high_threshold,
            "output_size": args.output_size
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("Generation complete")
        return 0
    
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 