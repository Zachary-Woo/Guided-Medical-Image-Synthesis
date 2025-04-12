#!/usr/bin/env python
"""
LoRA Training Script for Brain MRI Domain Adaptation

This script trains a LoRA adapter to adapt Stable Diffusion to generate
realistic brain MRI images. The adapter modifies the UNet and optionally
the text encoder while leaving most of the model frozen.
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
    DDPMScheduler,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from peft import LoraConfig, get_peft_model

from PIL import Image
from tqdm.auto import tqdm

# Will error if the minimal version of diffusers is not installed
check_min_version("0.19.0")

logger = get_logger(__name__)

class MedicalImageDataset(Dataset):
    """
    Dataset for LoRA training on medical images with text prompts.
    
    Expected format: A folder containing images and a metadata.jsonl file
    with prompts for each image.
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
        metadata_file = self.data_root / "metadata.jsonl"
        
        if not metadata_file.exists():
            raise ValueError(f"Metadata file not found: {metadata_file}")
        
        with open(metadata_file, "r") as f:
            for line in f:
                self.metadata.append(json.loads(line))
        
        logger.info(f"Loaded {len(self.metadata)} examples from {data_root}")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        item = self.metadata[idx]
        text = item["text"]
        image_path = self.data_root / item["file_name"]
        
        # Load and process image
        image = Image.open(image_path).convert("RGB")
        
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
        
        # Resize to target size
        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        
        # Apply random horizontal flip with 50% probability
        if self.random_flip and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Normalize to [0, 1] and convert to channel-first format (C, H, W)
        image_array = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        
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
            "text": text
        }

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a LoRA adapter for brain MRI generation"
    )
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to prepared BraTS data")
    parser.add_argument("--output_dir", type=str, default="version3/models/mri_lora",
                       help="Output directory for LoRA weights")
    parser.add_argument("--base_model", type=str, default="stabilityai/stable-diffusion-2-1",
                       help="Base model to adapt")
    parser.add_argument("--resolution", type=int, default=512,
                       help="Training resolution")
    parser.add_argument("--train_batch_size", type=int, default=1,
                       help="Training batch size")
    parser.add_argument("--num_train_epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--max_train_steps", type=int, default=None,
                       help="Max training steps. If set, overrides num_train_epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--rank", type=int, default=4,
                       help="LoRA rank")
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
    parser.add_argument("--validation_prompt", type=str, default="T1 weighted axial brain MRI scan with tumor",
                       help="Prompt for validation during training")
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
    
    # Freeze VAE
    vae.requires_grad_(False)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        target_modules=[
            "to_q", "to_k", "to_v", "to_out.0", 
            "ff.net.0.proj", "ff.net.2"
        ],
        lora_dropout=0.1,
        bias="none",
    )
    
    # Add LoRA to UNet
    unet = get_peft_model(unet, lora_config)
    
    # Enable xformers if available
    if is_xformers_available():
        try:
            unet.enable_xformers_memory_efficient_attention()
            logger.info("Using xformers memory efficient attention")
        except Exception as e:
            logger.warning(f"Could not enable xformers: {e}")
    
    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        logger.info("Gradient checkpointing enabled")
    
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
    
    # Only optimize the LoRA parameters
    params_to_optimize = [p for n, p in unet.named_parameters() if "lora" in n and p.requires_grad]
    
    # Create optimizer
    optimizer = optimizer_cls(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
    )
    
    # Create dataset and dataloader
    train_dataset = MedicalImageDataset(
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
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    
    # Keep VAE on CPU or GPU depending on batch size
    vae.to(accelerator.device, dtype=torch.float32)
    
    # Initialize trackers
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
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
        unet.train()
        train_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resume step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            
            with accelerator.accumulate(unet):
                # Get input tensor dimensions
                pixel_values = batch["pixel_values"].to(dtype=torch.float32)
                
                # Encode images into latent space
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
                
                # Predict the noise residual with the unet
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                # Calculate loss
                loss = F.mse_loss(noise_pred, noise, reduction="mean")
                
                # Gather the losses across all processes for logging (if needed)
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                
                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, 1.0)
                
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
    
    # Save the final LoRA adapter weights
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        # Save the PEFT adapter model
        unet = accelerator.unwrap_model(unet)
        unet.save_pretrained(args.output_dir)
        
        # Create model metadata card
        model_card = f"""
        # LoRA Adapter for Brain MRI Generation

        This LoRA adapter was trained on the BraTS dataset to adapt Stable Diffusion for generating realistic brain MRI images.

        ## Training Parameters
        
        - Base model: {args.base_model}
        - Resolution: {args.resolution}
        - LoRA rank: {args.rank}
        - Training steps: {global_step}
        - Batch size: {args.train_batch_size}
        - Learning rate: {args.learning_rate}
        - Seed: {args.seed}
        - Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        
        ## Usage

        ```python
        from diffusers import StableDiffusionPipeline
        
        # Load base model
        pipe = StableDiffusionPipeline.from_pretrained("{args.base_model}")
        
        # Load LoRA weights
        pipe.unet.load_attn_procs("{args.output_dir}")
        
        # Generate an image
        prompt = "T1 weighted axial brain MRI scan with tumor in left temporal lobe"
        image = pipe(prompt).images[0]
        ```
        """
        
        with open(os.path.join(args.output_dir, "README.md"), "w") as f:
            f.write(model_card)
        
        logger.info(f"LoRA adapter saved to {args.output_dir}")
        
        # Create a simplified config for use with our custom scripts
        simplified_config = {
            "base_model": args.base_model,
            "lora_rank": args.rank,
            "seed": args.seed,
            "resolution": args.resolution,
            "training_steps": global_step
        }
        
        with open(os.path.join(args.output_dir, "config.json"), "w") as f:
            json.dump(simplified_config, f, indent=2)
    
    accelerator.end_training()

if __name__ == "__main__":
    main() 