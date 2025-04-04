"""
Utilities for creating and loading Diffusers pipelines.
"""

import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import logging
import os
import gc
import traceback
from huggingface_hub import scan_cache_dir, HfFolder
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

logger = logging.getLogger(__name__)

def clear_gpu_memory():
    """
    Clear GPU memory to avoid out-of-memory errors.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("GPU memory cleared")

def check_model_in_cache(model_id):
    """
    Check if a model is already in the Hugging Face cache.
    
    Args:
        model_id (str): Hugging Face model ID
        
    Returns:
        bool: True if model is in cache, False otherwise
    """
    try:
        # Get Hugging Face cache directory
        if hasattr(HfFolder, 'exists_cache_dir') and HfFolder.exists_cache_dir():
            cached_repos = scan_cache_dir()
            for model in cached_repos.repos:
                for d in model.downstream_dirs:
                    if model_id in d.repo_id:
                        logger.info(f"Model {model_id} found in cache at {d.repo_type}/{d.repo_id}")
                        return True
    except Exception as e:
        logger.warning(f"Error checking cache for model {model_id}: {e}")
    
    return False

def load_sd_controlnet_pipeline(
    base_model_id: str,
    controlnet_model_id: str,
    device: torch.device,
    use_auth_token: bool = False,
    max_retries: int = 3
) -> StableDiffusionControlNetPipeline:
    """
    Loads a Stable Diffusion ControlNet pipeline with specified pre-trained models.

    Also configures the pipeline with a faster scheduler (UniPCMultistepScheduler)
    and enables memory optimizations (CPU offloading, xFormers if available).

    Args:
        base_model_id (str): Hugging Face ID or local path of the base Stable Diffusion model.
        controlnet_model_id (str): Hugging Face ID or local path of the pre-trained ControlNet model.
        device (torch.device): The target device ('cuda' or 'cpu') to load the pipeline onto.
        use_auth_token (bool): Whether to use the Hugging Face auth token for downloading models.
        max_retries (int): Maximum number of retries for downloading models.

    Returns:
        StableDiffusionControlNetPipeline: The loaded and configured pipeline.

    Raises:
        Exception: If model loading fails after max_retries.
    """
    # Check available VRAM
    if device.type == "cuda":
        torch_dtype = torch.float16
        free_vram = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        free_vram_gb = free_vram / (1024**3)
        logger.info(f"Available VRAM: {free_vram_gb:.2f} GB")
        
        # Warn if low on VRAM
        if free_vram_gb < 4:
            logger.warning("Low GPU memory. Model may fail to load or run slowly.")
            
        # For very low memory, fallback to CPU
        if free_vram_gb < 2:
            logger.warning("Extremely low GPU memory. Forcing CPU execution.")
            device = torch.device("cpu")
            torch_dtype = torch.float32
    else:
        torch_dtype = torch.float32
    
    # Clear GPU memory before loading
    clear_gpu_memory()
    
    for attempt in range(max_retries):
        try:
            # Load controlnet
            logger.info(f"Loading ControlNet model: {controlnet_model_id} (attempt {attempt+1}/{max_retries})")
            controlnet = ControlNetModel.from_pretrained(
                controlnet_model_id,
                torch_dtype=torch_dtype,
                use_auth_token=use_auth_token
            )
            
            # Load base model
            logger.info(f"Loading base Stable Diffusion model: {base_model_id} (attempt {attempt+1}/{max_retries})")
            pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                base_model_id,
                controlnet=controlnet,
                safety_checker=None,  # Disable safety checker for medical images
                torch_dtype=torch_dtype,
                use_auth_token=use_auth_token
            )
            
            # Set scheduler
            logger.info("Setting scheduler to UniPCMultistepScheduler")
            pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
            
            # Memory optimizations
            if device.type == "cuda":
                logger.info("Enabling model CPU offload for memory saving")
                pipeline.enable_model_cpu_offload()
                
                # Try to enable attention slicing as fallback if not enough memory
                try:
                    pipeline.enable_attention_slicing(1)
                    logger.info("Enabled attention slicing")
                except Exception as e:
                    logger.warning(f"Could not enable attention slicing: {e}")
                
                # Try to enable xformers if available
                try:
                    if hasattr(pipeline, "enable_xformers_memory_efficient_attention"):
                        pipeline.enable_xformers_memory_efficient_attention()
                        logger.info("Enabled xFormers memory efficient attention")
                except Exception as e:
                    logger.warning(f"Could not enable xFormers: {e}")
            else:
                # For CPU, we should manually move to CPU
                pipeline = pipeline.to(device)
                logger.info(f"Pipeline moved to {device}")
            
            logger.info("Pipeline loaded successfully")
            return pipeline
            
        except Exception as e:
            logger.error(f"Error loading pipeline (attempt {attempt+1}/{max_retries}): {e}")
            clear_gpu_memory()
            
            if attempt == max_retries - 1:
                logger.error("Maximum retries reached. Could not load pipeline.")
                raise
            else:
                logger.info("Retrying...")
    
    # This should not be reached due to the raise in the loop
    raise RuntimeError("Failed to load pipeline after maximum retries")


def preprocess_image_for_controlnet(image, conditioning_type, target_size):
    """
    Preprocess an image for use with ControlNet.
    
    Args:
        image (PIL.Image): The input image
        conditioning_type (str): Type of conditioning (e.g., 'canny', 'seg')
        target_size (int): Target size for the processed image
        
    Returns:
        PIL.Image: Processed image ready for ControlNet
    """
    import numpy as np
    import cv2
    from PIL import Image
    
    if image is None:
        logger.warning("No image to preprocess")
        return None
    
    if conditioning_type == 'none':
        return None
    
    try:
        # Convert to numpy array
        image_np = np.array(image)
        
        if conditioning_type == 'canny':
            # Resize image
            if image_np.shape[0] != target_size or image_np.shape[1] != target_size:
                image_np = cv2.resize(image_np, (target_size, target_size))
            
            # Convert to grayscale if needed
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            else:
                image_gray = image_np
            
            # Apply Canny edge detection
            low_threshold = 100
            high_threshold = 200
            canny_edges = cv2.Canny(image_gray, low_threshold, high_threshold)
            
            # Convert back to RGB format expected by ControlNet
            canny_edges = np.stack([canny_edges] * 3, axis=2)
            result = Image.fromarray(canny_edges.astype(np.uint8))
            
        elif conditioning_type == 'seg':
            # For segmentation masks, just resize
            result = image.resize((target_size, target_size), Image.NEAREST)
            if result.mode != 'RGB':
                result = result.convert('RGB')
                
        else:
            logger.warning(f"Unsupported conditioning type: {conditioning_type}")
            # Default: just resize
            result = image.resize((target_size, target_size))
            
        return result
        
    except Exception as e:
        logger.error(f"Error preprocessing image for {conditioning_type}: {e}")
        logger.error(traceback.format_exc())
        return None


def generate_with_controlnet(
    pipeline, 
    prompt, 
    condition_image=None,
    negative_prompt="blurry, low quality, low resolution, deformed, distorted",
    num_inference_steps=30,
    guidance_scale=7.5,
    seed=None,
    return_dict=True
):
    """
    Generate an image using the ControlNet pipeline.
    
    Args:
        pipeline: The loaded ControlNet pipeline
        prompt (str): Text prompt for generation
        condition_image (PIL.Image, optional): Conditioning image
        negative_prompt (str): Negative prompt for generation
        num_inference_steps (int): Number of inference steps
        guidance_scale (float): Guidance scale
        seed (int, optional): Random seed
        return_dict (bool): Whether to return a dict or just the image
        
    Returns:
        Image or dict: Generated image or dict with image and metadata
    """
    import traceback
    
    try:
        # Set up generator with seed if provided
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
        else:
            generator = None
        
        # Check if condition_image is None and handle accordingly
        kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": generator
        }
        
        # Only add image if it's not None
        if condition_image is not None:
            kwargs["image"] = condition_image
        
        # Generate image
        output = pipeline(**kwargs)
        
        # Return full output or just the image
        if return_dict:
            result = {
                "images": output.images,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "seed": seed,
                "steps": num_inference_steps,
                "guidance_scale": guidance_scale
            }
            return result
        else:
            return output.images[0]
            
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        logger.error(traceback.format_exc())
        return None 