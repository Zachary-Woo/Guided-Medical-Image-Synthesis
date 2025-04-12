#!/usr/bin/env python
"""
ControlNet Training Script for Brain MRI Tumor Mask Conditioning

This script trains a ControlNet model that learns to generate MRI images
conditioned on tumor segmentation masks. The resulting model can be used
to generate anatomically accurate brain MRIs with tumors in specific locations.
"""

import os
import logging
import argparse
import json
import math
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from PIL import Image
from tqdm.auto import tqdm

# Will error if the minimal version of diffusers is not installed
check_min_version("0.19.0")

logger = get_logger(__name__)

class ControlNetTrainingDataset(Dataset):
    """
    Dataset for ControlNet training on medical images with masks.
    
    Expected format: A folder containing images, masks, and a JSONL file
    with entries that have paths to paired images and masks.
    """
    def __init__(
        self,
        data_root,
        tokenizer,
        image_size=512,
        center_crop=False,
        random_flip=False,
        max_token_length=77
    ):
        self.data_root = Path(data_root)
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.max_token_length = max_token_length
        
        # Load metadata
        self.metadata = []
        
        # Try to find controlnet_train.jsonl or fallback to metadata.jsonl
        if (self.data_root / "controlnet_train.jsonl").exists():
            metadata_file = self.data_root / "controlnet_train.jsonl"
        else:
            metadata_file = self.data_root / "metadata.jsonl"
        
        if not metadata_file.exists():
            raise ValueError(f"Metadata file not found: {metadata_file}")
        
        with open(metadata_file, "r") as f:
            for line in f:
                item = json.loads(line)
                # Ensure it has the required keys
                if "conditioning_image" in item and "image" in item:
                    self.metadata.append(item)
                elif "mask_path" in item and "image_path" in item:
                    # Convert to expected format
                    self.metadata.append({
                        "conditioning_image": item["mask_path"],
                        "image": item["image_path"],
                        "prompt": item.get("prompt", "Brain MRI scan")
                    })
        
        logger.info(f"Loaded {len(self.metadata)} examples from {data_root}")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        item = self.metadata[idx]
        text = item.get("prompt", "Brain MRI scan")
        image_path = self.data_root / item["image"]
        mask_path = self.data_root / item["conditioning_image"]
        
        # Load and process image
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")
        
        # Apply transformations
        if self.center_crop:
            # Calculate the center crop dimensions
            w, h = image.size
            min_dim = min(w, h)
            left = (w - min_dim) // 2
            top = (h - min_dim) // 2
            right = left + min_dim
            bottom = top + min_dim
            image = image.crop((left, top, right, bottom))
            mask = mask.crop((left, top, right, bottom))
        
        # Resize to target size
        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        
        # Apply random horizontal flip with 50% probability
        if self.random_flip and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Normalize to [0, 1] and convert to channel-first format (C, H, W)
        image_array = np.array(image).astype(np.float32) / 255.0
        mask_array = np.array(mask).astype(np.float32) / 255.0
        
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask_array).permute(2, 0, 1)
        
        # Tokenize text
        input_ids = self.tokenizer(
            text,
            max_length=self.max_token_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids[0]
        
        return {
            "input_ids": input_ids,
            "pixel_values": image_tensor,
            "conditioning_pixel_values": mask_tensor,
            "text": text
        }

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a ControlNet model for brain MRI generation conditioned on tumor masks"
    )
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to prepared BraTS data")
    parser.add_argument("--output_dir", type=str, default="version3/models/segmentation_controlnet",
                       help="Output directory for ControlNet weights")
    parser.add_argument("--base_model", type=str, default="stabilityai/stable-diffusion-2-1",
                       help="Base model")
    parser.add_argument("--controlnet_model", type=str, default="lllyasviel/sd-controlnet-seg",
                       help="Base ControlNet model to fine-tune")
    parser.add_argument("--resolution", type=int, default=512,
                       help="Training resolution")
    parser.add_argument("--train_batch_size", type=int, default=1,
                       help="Training batch size")
    parser.add_argument("--num_train_epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--max_train_steps", type=int, default=None,
                       help="Max training steps. If set, overrides num_train_epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="Learning rate")
    parser.add_argument("--scale_lr", action="store_true",
                       help="Scale learning rate by batch size, gradient accumulation, and mixed precision")
    parser.add_argument("--lr_scheduler", type=str, default="constant",
                       choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
                       help="Learning rate scheduler")
    parser.add_argument("--lr_warmup_steps", type=int, default=500,
                       help="Number of warmup steps for the learning rate scheduler")
    parser.add_argument("--use_8bit_adam", action="store_true",
                       help="Use 8-bit Adam optimizer")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="Number of gradient accumulation steps")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                       help="Enable gradient checkpointing")
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                       choices=["no", "fp16", "bf16"],
                       help="Mixed precision training")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--checkpointing_steps", type=int, default=500,
                       help="Save a checkpoint every N steps")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="Path to checkpoint to resume from")
    
    return parser.parse_args()

def main():
    """Main training function."""
    args = parse_args()
    
    # Initialize accelerator
    logging_dir = os.path.join(args.output_dir, "logs")
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_config=accelerator_project_config,
    )
    
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    # Create output directory
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(logging_dir, exist_ok=True)
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Load tokenizer and models
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        subfolder="tokenizer",
        use_fast=False,
    )
    
    # Load models
    noise_scheduler = DDPMScheduler.from_pretrained(args.base_model, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(args.base_model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.base_model, subfolder="unet")
    
    # Load ControlNet
    logger.info(f"Loading ControlNet from {args.controlnet_model}")
    controlnet = ControlNetModel.from_pretrained(args.controlnet_model)
    
    # Freeze VAE and UNet
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    
    # Make sure ControlNet is trainable
    controlnet.train()
    
    # Enable xformers if available
    if is_xformers_available():
        try:
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
            logger.info("Using xformers memory efficient attention")
        except Exception as e:
            logger.warning(f"Could not enable xformers: {e}")
    
    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()
        logger.info("Gradient checkpointing enabled for ControlNet")
    
    # Create optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
            logger.info("Using 8-bit AdamW")
        except ImportError:
            logger.warning("bitsandbytes not available, using standard AdamW")
            optimizer_cls = torch.optim.AdamW
    else:
        optimizer_cls = torch.optim.AdamW
    
    # Only optimize the ControlNet parameters
    optimizer = optimizer_cls(
        controlnet.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
    )
    
    # Create dataset and dataloader
    train_dataset = ControlNetTrainingDataset(
        data_root=args.data_path,
        tokenizer=tokenizer,
        image_size=args.resolution,
        center_crop=True,
        random_flip=True,
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=2,
    )
    
    # Determine number of training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
    # Create learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )
    
    # Prepare models for accelerator
    controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler
    )
    
    # Keep VAE and UNet on device but we don't want to optimize them
    unet = accelerator.prepare(unet)
    vae.to(accelerator.device, dtype=torch.float32)
    
    # Initialize trackers
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running ControlNet training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size = {args.train_batch_size}")
    logger.info(f"  Total batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    global_step = 0
    first_epoch = 0
    
    # Resume from checkpoint if specified
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            checkpoint_path = Path(args.resume_from_checkpoint)
        else:
            # Get the latest checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            
            if len(dirs) == 0:
                logger.warning(f"No checkpoints found in {args.output_dir}")
            else:
                checkpoint_path = Path(args.output_dir) / dirs[-1]
        
        if checkpoint_path:
            logger.info(f"Resuming from checkpoint {checkpoint_path}")
            accelerator.load_state(checkpoint_path)
            global_step = int(checkpoint_path.name.split("-")[1])
            
            resume_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_step % num_update_steps_per_epoch
    
    # Training loop
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")
    
    for epoch in range(first_epoch, args.num_train_epochs):
        controlnet.train()
        train_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resume step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            
            with accelerator.accumulate(controlnet):
                # Get input tensor dimensions
                pixel_values = batch["pixel_values"].to(dtype=torch.float32)
                conditioning_pixel_values = batch["conditioning_pixel_values"].to(dtype=torch.float32)
                
                # Encode images into latent space
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                
                # Sample noise
                noise = torch.randn_like(latents)
                
                # Sample a random timestep for each image
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                
                # Add noise to the latents according to the noise magnitude at the timestep
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Get the text embeddings
                encoder_hidden_states = batch["input_ids"]
                
                # Get the ControlNet conditioning
                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=conditioning_pixel_values,
                    return_dict=False,
                )
                
                # Use ControlNet conditioning in UNet
                # This is the forward pass with ControlNet residual connections
                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample
                
                # Calculate loss - the goal is for the conditioned UNet to predict the original noise
                loss = F.mse_loss(noise_pred, noise, reduction="mean")
                
                # Gather the losses across all processes for logging (if needed)
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                
                # Backpropagate
                accelerator.backward(loss)
                
                # Clip gradients
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(controlnet.parameters(), 1.0)
                
                # Update
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Advance progress bar
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                # Log metrics
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved checkpoint at {save_path}")
                
                logs = {"loss": train_loss, "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                train_loss = 0.0
            
            # Stop if we reach max steps
            if global_step >= args.max_train_steps:
                break
    
    # Save the final ControlNet weights
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        controlnet = accelerator.unwrap_model(controlnet)
        controlnet.save_pretrained(args.output_dir)
        
        # Create model card
        model_card = f"""
        # ControlNet for Brain MRI Tumor Mask Conditioning

        This ControlNet model was trained to generate realistic brain MRI images from tumor segmentation masks.
        It can be used to create synthetic MRI data with precise tumor locations.

        ## Training Parameters
        
        - Base model: {args.base_model}
        - Original ControlNet: {args.controlnet_model}
        - Resolution: {args.resolution}
        - Training steps: {global_step}
        - Learning rate: {args.learning_rate}
        - Batch size: {args.train_batch_size}
        - Seed: {args.seed}
        - Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        
        ## Usage

        ```python
        from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
        
        # Load the trained ControlNet
        controlnet = ControlNetModel.from_pretrained("{args.output_dir}")
        
        # Load the pipeline
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "{args.base_model}",
            controlnet=controlnet
        )
        
        # Prepare a tumor mask as conditioning
        conditioning_image = load_mask("path/to/tumor_mask.png")
        
        # Generate a conditional image
        prompt = "T1 weighted axial brain MRI scan with tumor"
        image = pipe(prompt, image=conditioning_image).images[0]
        ```
        """
        
        with open(os.path.join(args.output_dir, "README.md"), "w") as f:
            f.write(model_card)
        
        # Save a metadata file with training parameters
        metadata = {
            "base_model": args.base_model,
            "controlnet_base": args.controlnet_model,
            "resolution": args.resolution,
            "training_steps": global_step,
            "learning_rate": args.learning_rate,
            "batch_size": args.train_batch_size,
            "seed": args.seed,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"ControlNet model saved to {args.output_dir}")
    
    accelerator.end_training()

if __name__ == "__main__":
    main() 