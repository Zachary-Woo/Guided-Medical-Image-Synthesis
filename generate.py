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
import cv2
import numpy as np
from src.utils.pipeline_utils import load_sd_controlnet_pipeline
from src.preprocessing.transforms import tensor_to_pil


def parse_args():
    """
    Parse command line arguments for generation.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Generate medical images with ControlNet")
    
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--conditioning_source_images", type=str, nargs="+",
                        help="Paths to images used for conditioning (e.g., masks for seg, real images for canny)")
    parser.add_argument("--prompts", type=str, nargs="+", help="Text prompts for generation")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (overrides config if set)")
    parser.add_argument("--steps", type=int, default=None, help="Number of inference steps (overrides config if set)")
    parser.add_argument("--guidance_scale", type=float, default=None, help="Guidance scale (overrides config if set)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides config if set)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation")
    
    return parser.parse_args()


def load_conditioning_sources(image_paths):
    """
    Load conditioning source images from paths as PIL Images.
    
    Args:
        image_paths (list): List of image paths
        
    Returns:
        list: List of PIL images, or None if image_paths is empty/None
    """
    if not image_paths:
        return None
    
    source_images = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            source_images.append(img)
        except Exception as e:
            logging.error(f"Failed to load conditioning source image: {path} - {e}")
            continue
    
    return source_images if source_images else None


def preprocess_conditioning_image(source_image_pil, conditioning_type, target_size):
    """
    Preprocess the source PIL image based on the conditioning type for ControlNet.

    Currently supports:
        - 'canny': Applies Canny edge detection.
        - 'seg': Resizes segmentation mask using NEAREST interpolation.
        - 'none': Returns None.
        - Other types: Resizes the original image (may not be suitable for ControlNet).

    Args:
        source_image_pil (PIL.Image): The source image (e.g., a real image for Canny, a mask for Seg).
        conditioning_type (str): Type of conditioning ('canny', 'seg', 'none', etc.).
        target_size (int): The target square size (height and width) for the output conditioning image.

    Returns:
        PIL.Image or None: The processed conditioning image ready for ControlNet input, or None if
                         conditioning_type is 'none' or an error occurs.
    """
    if source_image_pil is None:
         logging.warning("Source image for preprocessing is None.")
         return None
    if conditioning_type == 'none':
        return None

    logging.debug(f"Preprocessing image for conditioning type: {conditioning_type}")

    try:
        if conditioning_type == 'canny':
            source_image_pil = source_image_pil.resize((target_size, target_size))
            image_np = np.array(source_image_pil)
            low_threshold = 100
            high_threshold = 200
            canny_edges_np = cv2.Canny(image_np, low_threshold, high_threshold)
            canny_edges_np = canny_edges_np[:, :, None]
            canny_edges_np = np.concatenate([canny_edges_np, canny_edges_np, canny_edges_np], axis=2)
            conditioning_image_pil = Image.fromarray(canny_edges_np)

        elif conditioning_type == 'seg':
            conditioning_image_pil = source_image_pil.resize((target_size, target_size), Image.NEAREST)
            if conditioning_image_pil.mode != 'RGB':
                 conditioning_image_pil = conditioning_image_pil.convert('RGB')

        else:
            logging.warning(f"Unsupported conditioning type: {conditioning_type}. Returning unprocessed source.")
            conditioning_image_pil = source_image_pil.resize((target_size, target_size))

        logging.debug(f"Conditioning image size after processing: {conditioning_image_pil.size}")
        return conditioning_image_pil

    except Exception as e:
        logging.error(f"Error preprocessing conditioning image for type {conditioning_type}: {e}")
        return None


def generate_images(
    config: Config,
    prompts: list,
    conditioning_source_images: list = None,
    override_steps: int = None,
    override_guidance_scale: float = None,
    override_seed: int = None,
    batch_size: int = 1
):
    """
    Generate images using ControlNet with configuration.
    
    Args:
        config (Config): Project configuration object.
        prompts (list): List of text prompts.
        conditioning_source_images (list, optional): List of source PIL images for conditioning.
        override_steps (int, optional): Override config inference steps.
        override_guidance_scale (float, optional): Override config guidance scale.
        override_seed (int, optional): Override config seed.
        batch_size (int): Batch size (currently only 1 is robustly supported with conditioning).
        
    Returns:
        list: List of generated PIL images.
    """
    seed = override_seed if override_seed is not None else config.training.seed
    num_inference_steps = override_steps if override_steps is not None else config.inference.num_inference_steps
    guidance_scale = override_guidance_scale if override_guidance_scale is not None else config.inference.guidance_scale
    cond_type = config.inference.conditioning_type
    img_size = config.data.image_size

    generator = torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    logging.info(f"Using Base Model: {config.model.pretrained_model_id}")
    logging.info(f"Using ControlNet: {config.inference.controlnet_inference_id} ({cond_type} conditioning)")

    try:
        pipeline = load_sd_controlnet_pipeline(
            base_model_id=config.model.pretrained_model_id,
            controlnet_model_id=config.inference.controlnet_inference_id,
            device=device
        )
    except Exception as e:
        logging.error(f"Failed to load pipeline: {e}")
        return []

    generated_images_pil = []
    num_prompts = len(prompts)
    num_cond_images = len(conditioning_source_images) if conditioning_source_images else 0

    if batch_size > 1:
         logging.warning("Batch size > 1 is not fully tested with conditioning images, running with batch size 1.")
         batch_size = 1

    for i in range(num_prompts):
        prompt = prompts[i]
        source_img = conditioning_source_images[i] if i < num_cond_images else None

        logging.info(f"Generating image {i+1}/{num_prompts} for prompt: '{prompt[:50]}...'")

        conditioning_image_pil = preprocess_conditioning_image(
            source_image_pil=source_img,
            conditioning_type=cond_type,
            target_size=img_size
        )

        if source_img is not None and conditioning_image_pil is None:
            logging.warning(f"Conditioning source was provided but preprocessing failed for type '{cond_type}'. Generating without condition.")

        pipeline_args = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": generator
        }
        if conditioning_image_pil:
            pipeline_args["image"] = conditioning_image_pil
        elif cond_type != 'none':
             logging.warning(f"Conditioning type is '{cond_type}' but no valid conditioning image provided/processed for prompt {i}. Generating text-to-image only.")

        with torch.no_grad():
            try:
                output = pipeline(**pipeline_args)
                if hasattr(output, 'images') and output.images:
                     generated_images_pil.extend(output.images)
                else:
                    logging.error(f"Pipeline did not return images for prompt {i}.")
            except Exception as e:
                logging.error(f"Error during pipeline inference for prompt {i}: {e}")
                continue

    return generated_images_pil


def main():
    """
    Main generation function.
    """
    args = parse_args()
    
    try:
        config = Config.from_yaml(args.config)
    except Exception as e:
        logging.error(f"Error loading config file {args.config}: {e}")
        sys.exit(1)

    output_dir = args.output_dir if args.output_dir else config.training.output_dir
    if not output_dir:
        output_dir = "generated_images"
        logging.warning(f"No output directory specified in args or config, using default: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, "generate.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ]
    )
    logging.info(f"Logging to {log_file}")

    conditioning_source_images = None
    if args.conditioning_source_images:
        logging.info(f"Loading conditioning source images from: {args.conditioning_source_images}")
        conditioning_source_images = load_conditioning_sources(args.conditioning_source_images)
        if not conditioning_source_images:
             logging.warning("Conditioning source images specified but failed to load any.")

    prompts = args.prompts if args.prompts else config.inference.prompts
    if not prompts:
         logging.error("No prompts provided via arguments or found in config. Cannot generate.")
         sys.exit(1)

    logging.info("Starting image generation...")
    generated_images_pil = generate_images(
        config=config,
        prompts=prompts,
        conditioning_source_images=conditioning_source_images,
        override_steps=args.steps,
        override_guidance_scale=args.guidance_scale,
        override_seed=args.seed,
        batch_size=args.batch_size
    )

    if generated_images_pil:
        logging.info(f"Saving {len(generated_images_pil)} generated images to {output_dir}...")
        for i, img in enumerate(generated_images_pil):
            try:
                img.save(os.path.join(output_dir, f"generated_{i:04d}.png"))
            except Exception as e:
                logging.error(f"Failed to save image {i}: {e}")

        try:
            grid_filename = os.path.join(output_dir, "generated_grid.png")
            logging.info(f"Creating image grid: {grid_filename}")
            plot_images(
                generated_images_pil,
                titles=[f"Prompt: {p[:30]}..." for p in prompts[:len(generated_images_pil)]],
                figsize=(20, 10),
                filename=grid_filename
            )
        except Exception as e:
            logging.error(f"Failed to create image grid: {e}")

    else:
         logging.warning("No images were generated.")

    logging.info(f"Generation process complete. Results (if any) saved to {output_dir}")


if __name__ == "__main__":
    main() 