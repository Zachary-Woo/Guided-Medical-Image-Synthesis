#!/usr/bin/env python
"""
Simple text-to-image generation script as a fallback
"""

import os
import sys
import torch
import argparse
import logging
from PIL import Image
import traceback
from pathlib import Path
import json
from diffusers import StableDiffusionPipeline

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
    parser = argparse.ArgumentParser(description="Generate medical images with Stable Diffusion")
    parser.add_argument("--prompt", type=str, default="High-resolution histopathology slide showing colon tissue",
                        help="Text prompt for generation")
    parser.add_argument("--output_dir", type=str, default="output/simple_generation",
                        help="Output directory")
    parser.add_argument("--steps", type=int, default=30, 
                        help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set up logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging file
    log_file = output_dir / "generate.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(file_handler)
    
    logger.info(f"Logging to {log_file}")
    logger.info(f"Using prompt: {args.prompt}")
    
    # Set up device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    try:
        # Load model
        logger.info("Loading Stable Diffusion model")
        model_id = "runwayml/stable-diffusion-v1-5"
        
        # Set up parameters
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        
        # Load pipeline
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            safety_checker=None,
        )
        
        # Set up memory optimizations if on GPU
        if device == "cuda":
            pipeline = pipeline.to("cuda")
            pipeline.enable_attention_slicing()
            try:
                pipeline.enable_xformers_memory_efficient_attention()
                logger.info("Enabled xFormers memory efficient attention")
            except Exception as e:
                logger.warning(f"Could not enable xFormers: {e}")
        
        # Generate
        logger.info(f"Generating image with {args.steps} steps")
        generator = torch.Generator(device=device).manual_seed(args.seed)
        
        image = pipeline(
            prompt=args.prompt,
            num_inference_steps=args.steps,
            generator=generator,
        ).images[0]
        
        # Save image
        output_path = output_dir / "generated.png"
        image.save(output_path)
        logger.info(f"Image saved to {output_path}")
        
        # Save metadata
        metadata = {
            "prompt": args.prompt,
            "steps": args.steps,
            "seed": args.seed,
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