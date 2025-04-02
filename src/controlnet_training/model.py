"""
ControlNet model configuration and setup for medical image synthesis.
"""

import torch
from torch import nn
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    DDPMScheduler
)
from transformers import CLIPTextModel, CLIPTokenizer


def setup_controlnet(pretrained_model_id="runwayml/stable-diffusion-v1-5", 
                    controlnet_id=None):
    """
    Set up a ControlNet model for training.
    
    Args:
        pretrained_model_id (str): ID of the pretrained Stable Diffusion model
        controlnet_id (str, optional): ID of a pretrained ControlNet model
        
    Returns:
        tuple: The components needed for ControlNet training
    """
    # Set up the tokenizer and text encoder
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_id,
        subfolder="tokenizer"
    )
    
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_id,
        subfolder="text_encoder"
    )
    
    # Load the UNet from the pretrained model
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_id,
        subfolder="unet"
    )
    
    # Load or initialize a ControlNet model
    if controlnet_id:
        # Load a pre-trained ControlNet
        controlnet = ControlNetModel.from_pretrained(controlnet_id)
    else:
        # Initialize a new ControlNet from the UNet
        controlnet = ControlNetModel.from_unet(unet)
    
    # Set up the noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        pretrained_model_id,
        subfolder="scheduler"
    )
    
    return {
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "unet": unet,
        "controlnet": controlnet,
        "noise_scheduler": noise_scheduler
    }


def freeze_model_components(model_components, controlnet_trainable_modules=None):
    """
    Freeze model components for efficient training.
    
    Args:
        model_components (dict): Dictionary of model components
        controlnet_trainable_modules (list, optional): List of ControlNet module names to train
    """
    # Freeze text encoder
    for param in model_components["text_encoder"].parameters():
        param.requires_grad = False
    
    # Freeze UNet
    for param in model_components["unet"].parameters():
        param.requires_grad = False
    
    # By default, freeze all ControlNet parameters
    for param in model_components["controlnet"].parameters():
        param.requires_grad = False
    
    # If specific ControlNet modules are specified, unfreeze them
    if controlnet_trainable_modules:
        for module_name in controlnet_trainable_modules:
            for name, param in model_components["controlnet"].named_parameters():
                if module_name in name:
                    param.requires_grad = True 