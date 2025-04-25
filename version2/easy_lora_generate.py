#!/usr/bin/env python
"""
Simple medical image generation with LoRA
A minimal script to generate medical images using ControlNet and LoRA
"""

import os
import torch
import time
from pathlib import Path
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image
import numpy as np
import cv2
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def print_cuda_info():
    """Print information about CUDA availability and GPU"""
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    return torch.cuda.is_available()

def create_canny_image(image_path, low_threshold=100, high_threshold=200, target_size=512):
    """Create a canny edge image for controlnet input"""
    # Load and resize image
    image = Image.open(image_path).convert("RGB")
    image = image.resize((target_size, target_size), Image.LANCZOS)
    image_np = np.array(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    
    # Convert to RGB
    edge_image = np.stack([edges] * 3, axis=2)
    
    return Image.fromarray(edge_image), Image.fromarray(image_np)

def load_lora_weights(pipeline, lora_path, scale):
    """Load LoRA weights into the pipeline"""
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA file not found: {lora_path}")
    
    logger.info(f"Loading LoRA weights from: {lora_path}")
    logger.info(f"File exists: {os.path.exists(lora_path)}")
    
    try:
        if lora_path.endswith('.safetensors'):
            # Create a directory structure for LoRA weights *inside the output dir*
            lora_dir = Path("output/lora_test/temp_lora_adapter")
            lora_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy the safetensors file to the temp directory
            import shutil
            target_path = lora_dir / "adapter_model.safetensors"
            shutil.copy(lora_path, target_path)
            
            # Create a simple config
            with open(lora_dir / "adapter_config.json", "w") as f:
                import json
                config = {
                    "base_model_name_or_path": "runwayml/stable-diffusion-v1-5",
                    "inference_scheduler_config": None,
                    "lora_alpha": scale,
                    "lora_dropout": 0.0,
                    "peft_type": "LORA",
                    "rank": 4,
                    "target_modules": ["to_k", "to_q", "to_v", "to_out.0"]
                }
                json.dump(config, f, indent=2)
            
            # Try loading with PEFT adapter directory approach
            logger.info(f"Loading LoRA with directory adapter approach")
            pipeline.load_lora_weights(lora_dir)
            
        else:
            # Fallback method for non-safetensors
            logger.info("Loading LoRA from non-safetensors format")
            # Try using the folder approach
            folder_path = os.path.dirname(lora_path)
            pipeline.load_lora_weights(folder_path)
        
        # Fuse LoRA weights if possible
        try:
            pipeline.fuse_lora(lora_scale=scale)
            logger.info(f"LoRA weights fused with scale {scale}")
        except Exception as e:
            logger.warning(f"Could not fuse LoRA weights: {e}")
            logger.info("Continuing without fusing weights")
            
        return pipeline
    
    except Exception as e:
        logger.error(f"Error loading LoRA weights: {e}")
        logger.warning("Proceeding without LoRA weights - still generating image")
        return pipeline

def main():
    # Print CUDA information
    cuda_available = print_cuda_info()
    device = "cuda" if cuda_available else "cpu"
    print(f"Using device: {device}")
    
    # --- Configuration ---
    seed = 123 # Define the seed here
    condition_image_path = Path("data") / "pathmnist_samples" / "sample_0000.png"
    lora_path = Path("version2") / "models" / "lora_histopathology" / "adapter_model.safetensors" # Custom Trained model
    base_output_dir = Path("output")
    output_dir = base_output_dir / f"lora_test_{seed}" # Create seed-specific path
    output_dir.mkdir(parents=True, exist_ok=True) # Create the directory
    # --- End Configuration ---

    # Check if input image exists
    if not condition_image_path.exists():
        logger.error(f"Input image not found: {condition_image_path}")
        return

    # Process condition image
    print("Processing condition image...")
    canny_image, original_image = create_canny_image(
        condition_image_path, 
        low_threshold=100, 
        high_threshold=200
    )
    
    # Save condition images
    canny_image.save(output_dir / "canny_edge.png")
    original_image.save(output_dir / "original.png")
    
    # Load models
    print("Loading models...")
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny", 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    
    # Set scheduler
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    
    # Move to device
    pipeline = pipeline.to(device)
    
    # Load LoRA weights
    try:
        lora_scale = 0.8
        pipeline = load_lora_weights(pipeline, lora_path, lora_scale)
    except FileNotFoundError as e:
        logger.error(f"LoRA file not found: {e}")
        logger.warning("Continuing generation without LoRA weights.")
    except Exception as e:
        logger.error(f"Error loading LoRA weights: {e}")
        logger.warning("Proceeding without LoRA weights")
    
    # Enable optimizations
    if device == "cuda":
        pipeline.enable_attention_slicing()
        try:
            pipeline.enable_xformers_memory_efficient_attention()
            logger.info("xFormers enabled")
        except Exception as e:
            logger.warning(f"Could not enable xFormers: {e}")
    
    # Generate image
    print("Generating image...")
    prompt = "High-resolution histopathology slide showing colorectal tissue with cellular detail, H&E stain"
    negative_prompt = "blurry, low quality, low resolution, deformed, distorted, watermark, text"
    
    # Set seed for reproducibility
    generator = torch.manual_seed(seed) # Use the seed variable
    
    start_time = time.time()
    image = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=canny_image,
        num_inference_steps=30,
        guidance_scale=7.5,
        generator=generator
    ).images[0]
    
    end_time = time.time()
    
    # Save output
    output_path = output_dir / "generated_lora.png"
    image.save(output_path)
    
    print(f"Generated image saved to {output_path}")
    print(f"Generation took {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main() 