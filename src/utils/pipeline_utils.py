"""
Utilities for creating and loading Diffusers pipelines.
"""

import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import logging # Import logging

def load_sd_controlnet_pipeline(
    base_model_id: str,
    controlnet_model_id: str,
    device: torch.device
) -> StableDiffusionControlNetPipeline:
    """
    Loads a Stable Diffusion ControlNet pipeline with specified pre-trained models.

    Also configures the pipeline with a faster scheduler (UniPCMultistepScheduler)
    and enables memory optimizations (CPU offloading, xFormers if available).

    Args:
        base_model_id (str): Hugging Face ID or local path of the base Stable Diffusion model.
        controlnet_model_id (str): Hugging Face ID or local path of the pre-trained ControlNet model.
        device (torch.device): The target device ('cuda' or 'cpu') to load the pipeline onto.

    Returns:
        StableDiffusionControlNetPipeline: The loaded and configured pipeline.

    Raises:
        Exception: If model loading fails.
    """
    try:
        # Determine torch dtype based on device
        torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
        logging.info(f"Loading ControlNet model: {controlnet_model_id} with dtype: {torch_dtype}")

        controlnet = ControlNetModel.from_pretrained(
            controlnet_model_id,
            torch_dtype=torch_dtype
        )

        logging.info(f"Loading base Stable Diffusion model: {base_model_id}")
        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            base_model_id,
            controlnet=controlnet,
            safety_checker=None, # Often disabled for non-photorealistic/medical images
            torch_dtype=torch_dtype
        )

        # Use a potentially faster scheduler
        logging.info("Setting scheduler to UniPCMultistepScheduler")
        pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)

        # Memory optimizations
        logging.info("Enabling model CPU offload for memory saving.")
        pipeline.enable_model_cpu_offload() # More aggressive than attention slicing

        # Commented out xformers section as it can cause installation issues
        # if hasattr(pipeline, "enable_xformers_memory_efficient_attention"):
        #     try:
        #         pipeline.enable_xformers_memory_efficient_attention()
        #         logging.info("Enabled xFormers memory efficient attention.")
        #     except ImportError:
        #         logging.warning("xFormers not installed. Cannot enable memory efficient attention.")
        #     except Exception as e:
        #          logging.warning(f"Could not enable xFormers: {e}")

        # Note: enable_model_cpu_offload() handles moving to device, so pipeline.to(device) might be redundant/harmful
        # pipeline.to(device)
        logging.info("Pipeline loaded successfully.")
        return pipeline

    except Exception as e:
        logging.error(f"Failed to load pipeline: {e}")
        raise # Re-raise the exception after logging 