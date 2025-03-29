#!/usr/bin/env python
"""
Generation script for ControlNet medical image synthesis.
"""

import os
import sys
import torch
import argparse
import logging
from PIL import Image
import matplotlib.pyplot as plt
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from src.utils.config import Config
from src.utils.visualization import plot_images
from src.preprocessing.transforms import MedicalImageTransforms


def parse_args():
    """
    Parse command line arguments for generation.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Generate medical images with ControlNet")
    
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to ControlNet checkpoint")
    parser.add_argument("--control_images", type=str, nargs="+", help="Paths to control images (masks)")
    parser.add_argument("--prompts", type=str, nargs="+", help="Text prompts for generation")
    parser.add_argument("--output_dir", type=str, default="generated", help="Output directory")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    
    return parser.parse_args()


def load_control_images(image_paths, img_size=512):
    """
    Load control images from paths.
    
    Args:
        image_paths (list): List of image paths
        img_size (int): Size to resize images to
        
    Returns:
        list: List of processed control images
    """
    if not image_paths:
        return None
    
    control_images = []
    transforms = MedicalImageTransforms()
    transform = transforms.get_mask_transforms(img_size=img_size)
    
    for path in image_paths:
        img = Image.open(path).convert("L")
        img_tensor = transform(img)
        control_images.append(img_tensor)
    
    return control_images


def generate_images(
    controlnet_path,
    pretrained_model_id,
    prompts,
    control_images=None,
    num_inference_steps=50,
    guidance_scale=7.5,
    seed=42,
    batch_size=1
):
    """
    Generate images using ControlNet.
    
    Args:
        controlnet_path (str): Path to ControlNet checkpoint
        pretrained_model_id (str): ID of the pretrained Stable Diffusion model
        prompts (list): List of text prompts
        control_images (list, optional): List of control images (tensors)
        num_inference_steps (int): Number of inference steps
        guidance_scale (float): Guidance scale
        seed (int): Random seed
        batch_size (int): Batch size
        
    Returns:
        list: List of generated images
    """
    # Set random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load ControlNet
    controlnet = ControlNetModel.from_pretrained(controlnet_path)
    
    # Create pipeline
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        pretrained_model_id,
        controlnet=controlnet,
        safety_checker=None,  # Remove safety checker for medical images
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
    ).to(device)
    
    # Enable memory optimizations
    pipeline.enable_attention_slicing()
    if hasattr(pipeline, "enable_xformers_memory_efficient_attention"):
        pipeline.enable_xformers_memory_efficient_attention()
    
    # Generate images
    generated_images = []
    
    # If no control images, just use text prompts
    if control_images is None:
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            
            with torch.no_grad():
                batch_output = pipeline(
                    batch_prompts,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale
                )
                
                generated_images.extend(batch_output.images)
    else:
        # Match prompts with control images
        for i in range(min(len(prompts), len(control_images))):
            prompt = prompts[i]
            control_image = control_images[i].unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = pipeline(
                    prompt,
                    control_image,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale
                )
                
                generated_images.extend(output.images)
    
    return generated_images


def main():
    """
    Main generation function.
    """
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(args.output_dir, "generate.log"))
        ]
    )
    
    # Load config
    config = Config.from_yaml(args.config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load control images if provided
    control_images = None
    if args.control_images:
        control_images = load_control_images(args.control_images, config.data.image_size)
    
    # Use provided prompts or from config
    prompts = args.prompts if args.prompts else config.inference.prompts
    
    # Generate images
    logging.info("Generating images...")
    generated_images = generate_images(
        controlnet_path=args.checkpoint,
        pretrained_model_id=config.model.pretrained_model_id,
        prompts=prompts,
        control_images=control_images,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        batch_size=args.batch_size
    )
    
    # Save images
    logging.info(f"Saving {len(generated_images)} generated images...")
    for i, img in enumerate(generated_images):
        img.save(os.path.join(args.output_dir, f"generated_{i}.png"))
    
    # Create a grid of images
    fig = plot_images(
        generated_images,
        titles=[f"Prompt: {p[:20]}..." for p in prompts[:len(generated_images)]],
        figsize=(20, 10),
        filename=os.path.join(args.output_dir, "generated_grid.png")
    )
    
    logging.info(f"Generation complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main() 