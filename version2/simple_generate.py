#!/usr/bin/env python
"""
Simple standalone script to test image generation with diffusers and CUDA
"""
import os
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image
import numpy as np
import cv2
import time
from pathlib import Path

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

# Load input image
def load_image(image_path, target_size=512):
    # Load the original image
    original_image = Image.open(image_path).convert("RGB")
    # Resize the image to match the desired size
    if original_image.width != target_size or original_image.height != target_size:
        original_image = original_image.resize((target_size, target_size), Image.LANCZOS)
    return original_image

# Create a canny edge map
def create_canny_edge(image, low_threshold=100, high_threshold=200):
    # Convert PIL to numpy array
    np_image = np.array(image)
    # Convert to grayscale
    gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
    # Apply canny edge detection
    edge = cv2.Canny(gray, low_threshold, high_threshold)
    # Convert back to RGB
    edge_rgb = np.stack([edge] * 3, axis=2)
    # Convert to PIL
    return Image.fromarray(edge_rgb)

def main():
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Setup model paths
    controlnet_model_id = "lllyasviel/sd-controlnet-canny"
    base_model_id = "runwayml/stable-diffusion-v1-5"
    
    # Load controlnet model
    print("Loading ControlNet model...")
    controlnet = ControlNetModel.from_pretrained(controlnet_model_id, torch_dtype=torch.float32)
    
    # Setup pipeline
    print("Loading StableDiffusion pipeline...")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_id,
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=torch.float32
    )
    
    # Setup scheduler
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    
    # Move to device
    print(f"Moving model to {device}...")
    pipe = pipe.to(device)
    
    # Process input image
    print("Processing input image...")
    # Use Path for better cross-platform compatibility and join paths correctly
    input_image_path = Path("data") / "pathmnist_samples" / "sample_0000.png" 
    # Ensure output directory exists (moved from below)
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_image_path.exists():
        print(f"ERROR: Input image not found at {input_image_path}")
        return # Exit if input image is missing

    input_image = load_image(input_image_path)
    canny_image = create_canny_edge(input_image)
    
    # Save edge image
    canny_image.save(output_dir / "canny_edge_simple.png") # Changed filename to avoid conflict
    
    # Generate image
    print("Generating image...")
    start_time = time.time()
    prompt = "High-resolution histopathology slide showing colorectal tissue with cellular detail, H&E stain"
    negative_prompt = "blurry, low quality, low resolution, deformed, distorted, watermark, text"
    
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=canny_image,
        num_inference_steps=30,
        guidance_scale=7.5,
    ).images[0]
    
    end_time = time.time()
    
    # Save output
    # os.makedirs("output", exist_ok=True) # Moved up
    output_path = output_dir / "generated_simple.png"
    image.save(output_path)
    
    print(f"Generated image saved to {output_path}")
    print(f"Generation took {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main() 