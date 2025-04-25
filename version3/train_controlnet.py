"""
Train a ControlNet model for MRI brain tumor generation.
This script trains a ControlNet conditioned on segmentation masks to generate realistic MRI brain images with tumors.
"""

import os
import logging
import random
import yaml
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.utils.checkpoint

from PIL import Image
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

import diffusers
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
    DDPMScheduler,
    AutoencoderKL,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from transformers import CLIPTextModel, CLIPTokenizer

from tqdm.auto import tqdm
import nibabel as nib

from utils.mri_utils import normalize_intensity, extract_brain, mri_to_pil

# Will error if the minimal version of diffusers is not installed
check_min_version("0.21.0")

logger = get_logger(__name__)


class BraTSControlNetDataset(Dataset):
    """Dataset for training ControlNet with BraTS data."""
    
    def __init__(
        self,
        data_root,
        tokenizer,
        size=512,
        prompt_prefix="MRI scan of brain with ",
        center_crop=True,
        use_augmentation=False,
        modality="t1ce",
        text_file=None,
    ):
        self.data_root = Path(data_root)
        self.tokenizer = tokenizer
        self.size = size
        self.center_crop = center_crop
        self.use_augmentation = use_augmentation
        self.prompt_prefix = prompt_prefix
        self.modality = modality
        
        # Find all nifti files for the specified modality
        self.mri_files = list(self.data_root.glob(f"**/*{modality}.nii.gz"))
        
        # If text file is provided, load prompts from it
        self.text_prompts = {}
        if text_file is not None and os.path.exists(text_file):
            with open(text_file, "r") as f:
                for line in f:
                    parts = line.strip().split(":")
                    if len(parts) >= 2:
                        case_id = parts[0].strip()
                        prompt = ":".join(parts[1:]).strip()
                        self.text_prompts[case_id] = prompt
        
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        
        self.mask_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.NEAREST),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.mri_files)
    
    def _get_slice_range(self, volume):
        """Get the range of slices that contain the brain."""
        # Find slices with actual brain content
        non_zero_slices = []
        for i in range(volume.shape[2]):
            if np.sum(volume[:, :, i]) > 100:  # Threshold
                non_zero_slices.append(i)
        
        if not non_zero_slices:
            return 0, volume.shape[2]-1
        
        start_slice = max(0, min(non_zero_slices) - 5)
        end_slice = min(volume.shape[2]-1, max(non_zero_slices) + 5)
        return start_slice, end_slice
        
    def _generate_prompt(self, case_id):
        """Generate a descriptive prompt for the MRI scan."""
        if case_id in self.text_prompts:
            return self.prompt_prefix + self.text_prompts[case_id]
        
        # Default generic prompts if no specific prompt is found
        tumor_types = ["glioblastoma", "meningioma", "astrocytoma", "oligodendroglioma", "metastatic tumor"]
        locations = ["frontal lobe", "temporal lobe", "parietal lobe", "occipital lobe", "cerebellum"]
        effects = ["with mass effect", "with midline shift", "with surrounding edema", "compressing ventricles"]
        
        prompt = self.prompt_prefix
        prompt += random.choice(tumor_types)
        prompt += f" in the {random.choice(locations)}"
        
        if random.random() > 0.5:
            prompt += f" {random.choice(effects)}"
            
        return prompt
    
    def __getitem__(self, idx):
        mri_path = self.mri_files[idx]
        case_id = mri_path.parent.name
        
        # Load MRI volume
        mri_img = nib.load(str(mri_path))
        mri_data = mri_img.get_fdata()
        
        # Load segmentation mask (should be in the same directory)
        seg_path = mri_path.parent / "seg.nii.gz"
        seg_data = nib.load(str(seg_path)).get_fdata() if seg_path.exists() else np.zeros_like(mri_data)
        
        # Process the MRI data
        mri_data = normalize_intensity(mri_data)
        brain_mask = extract_brain(mri_data)
        mri_data = mri_data * brain_mask  # Apply brain mask
        
        # Find a good slice with tumor
        start_slice, end_slice = self._get_slice_range(mri_data)
        
        # Create a slice index list with higher probability for slices containing tumor
        slice_indices = list(range(start_slice, end_slice+1))
        if np.max(seg_data) > 0:  # If there's a tumor
            tumor_slice_indices = [i for i in slice_indices if np.sum(seg_data[:, :, i]) > 0]
            if tumor_slice_indices:
                # 80% chance to select a slice with tumor if available
                if random.random() < 0.8 and tumor_slice_indices:
                    slice_idx = random.choice(tumor_slice_indices)
                else:
                    slice_idx = random.choice(slice_indices)
            else:
                slice_idx = random.choice(slice_indices)
        else:
            slice_idx = random.choice(slice_indices)
        
        # Extract the selected slice
        mri_slice = mri_data[:, :, slice_idx]
        seg_slice = seg_data[:, :, slice_idx]
        
        # Ensure all tumor classes are represented in a single channel
        condition_image = np.zeros_like(seg_slice)
        if np.max(seg_slice) > 0:
            condition_image[seg_slice > 0] = 1
        
        # Convert to PIL images
        mri_pil = mri_to_pil(mri_slice)
        condition_pil = mri_to_pil(condition_image)
        
        # Apply transforms
        mri_tensor = self.image_transforms(mri_pil)
        condition_tensor = self.mask_transforms(condition_pil)
        
        # Repeat the condition tensor to have 3 channels
        condition_tensor = condition_tensor.repeat(3, 1, 1)
        
        # Generate prompt
        prompt = self._generate_prompt(case_id)
        
        # Tokenize text
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "pixel_values": mri_tensor,
            "conditioning_pixel_values": condition_tensor,
            "input_ids": text_inputs.input_ids[0],
            "prompt": prompt,
            "case_id": case_id,
            "slice_idx": slice_idx,
        }


def collate_fn(examples):
    """Collate function for the DataLoader."""
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    input_ids = torch.stack([example["input_ids"] for example in examples])
    
    prompts = [example["prompt"] for example in examples]
    case_ids = [example["case_id"] for example in examples]
    slice_indices = [example["slice_idx"] for example in examples]
    
    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "input_ids": input_ids,
        "prompts": prompts,
        "case_ids": case_ids,
        "slice_indices": slice_indices,
    }


def create_controlnet_model(config, accelerator):
    """Create or load ControlNet model"""
    # Load the tokenizer and text encoder
    tokenizer = CLIPTokenizer.from_pretrained(
        config["model"]["pretrained_model_name_or_path"], subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        config["model"]["pretrained_model_name_or_path"], subfolder="text_encoder"
    )
    
    # Freeze the text encoder
    text_encoder.requires_grad_(False)
    
    # Load the VAE
    vae_path = config["model"]["vae_model_name_or_path"] or config["model"]["pretrained_model_name_or_path"]
    vae = AutoencoderKL.from_pretrained(
        vae_path, subfolder="vae" if vae_path == config["model"]["pretrained_model_name_or_path"] else None
    )
    
    # Freeze the VAE
    vae.requires_grad_(False)
    
    # Load the UNet
    unet = UNet2DConditionModel.from_pretrained(
        config["model"]["pretrained_model_name_or_path"], subfolder="unet"
    )
    
    # Freeze the UNet
    unet.requires_grad_(False)
    
    # Create or load the ControlNet
    if config["controlnet"]["init_weights_from_controlnet"]:
        logger.info(f"Loading ControlNet from {config['controlnet']['init_weights_from_controlnet']}")
        controlnet = ControlNetModel.from_pretrained(config["controlnet"]["init_weights_from_controlnet"])
    elif config["controlnet"]["controlnet_model_name_or_path"]:
        logger.info(f"Loading ControlNet from {config['controlnet']['controlnet_model_name_or_path']}")
        controlnet = ControlNetModel.from_pretrained(config["controlnet"]["controlnet_model_name_or_path"])
    else:
        logger.info("Creating new ControlNet model")
        controlnet = ControlNetModel.from_unet(unet)
    
    # Freeze encoder if specified
    if config["controlnet"]["freeze_encoder"]:
        logger.info("Freezing ControlNet encoder")
        for name, param in controlnet.named_parameters():
            if "controlnet_cond_embedding" in name:
                param.requires_grad = False
    
    # Freeze everything except the conditioning layers if encoder_only is True
    if config["controlnet"]["encoder_only"]:
        logger.info("Training only the encoder part of ControlNet")
        for name, param in controlnet.named_parameters():
            if "controlnet_cond_embedding" not in name:
                param.requires_grad = False
    
    # Load the noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(config["model"]["pretrained_model_name_or_path"], subfolder="scheduler")
    
    # Enable xformers memory efficient attention if available and enabled
    if is_xformers_available() and config["performance"]["enable_xformers_memory_efficient_attention"]:
        import xformers
        unet.enable_xformers_memory_efficient_attention()
        controlnet.enable_xformers_memory_efficient_attention()
        logger.info("Using xformers memory efficient attention")
    
    # Return all models and tokenizer
    return {
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "vae": vae,
        "unet": unet,
        "controlnet": controlnet,
        "noise_scheduler": noise_scheduler
    }


def get_models_from_checkpoint(pretrained_model_name, controlnet_checkpoint_path=None):
    """Load models with optional ControlNet checkpoint."""
    # Load the base model components
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name, subfolder="text_encoder"
    )
    vae = diffusers.AutoencoderKL.from_pretrained(
        pretrained_model_name, subfolder="vae"
    )
    unet = diffusers.UNet2DConditionModel.from_pretrained(
        pretrained_model_name, subfolder="unet"
    )
    
    # Load or create ControlNet
    if controlnet_checkpoint_path:
        logger.info(f"Loading ControlNet from checkpoint: {controlnet_checkpoint_path}")
        controlnet = ControlNetModel.from_pretrained(controlnet_checkpoint_path)
    else:
        logger.info("Creating new ControlNet model")
        controlnet = ControlNetModel.from_unet(unet)
    
    return {
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "vae": vae,
        "unet": unet,
        "controlnet": controlnet,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train ControlNet for MRI generation")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/controlnet_training.yaml", 
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--local_rank", 
        type=int, 
        default=-1, 
        help="For distributed training: local_rank"
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained ControlNet model if finetuning",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory containing the training data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save model checkpoints and logs",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="t1ce",
        choices=["t1", "t1ce", "t2", "flair"],
        help="MRI modality to use for training",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None, 
        help="Random seed for initialization"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Resolution for input images",
    )
    parser.add_argument(
        "--train_batch_size", 
        type=int, 
        default=None, 
        help="Batch size for training"
    )
    parser.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=None, 
        help="Number of training epochs"
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=None,
        help="Number of updates steps to accumulate before performing a backward/update pass",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Initial learning rate (after warmup period)",
    )
    parser.add_argument(
        "--text_prompt_file",
        type=str,
        default=None,
        help="File containing text prompts for each case ID",
    )
    parser.add_argument(
        "--use_8bit_adam", 
        action="store_true", 
        help="Whether to use 8-bit Adam optimizer"
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Whether to allow TF32 on Ampere GPUs",
    )
    parser.add_argument(
        "--enable_xformers",
        action="store_true",
        help="Whether to use xFormers memory efficient attention",
    )
    args = parser.parse_args()
    return args


def main():
    # Parse arguments
    args = parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.pretrained_model_name_or_path:
        base_model = args.pretrained_model_name_or_path
    else:
        base_model = config["models"]["controlnet"]["base_model"]
    
    if args.data_dir:
        data_dir = args.data_dir
    else:
        data_dir = config["dataset"]["brats"]["path"]
    
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(config["paths"]["output_dir"], f"controlnet_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seed
    seed = args.seed if args.seed is not None else config["models"]["controlnet"]["seed"]
    set_seed(seed)
    
    # Initialize accelerator
    project_config = ProjectConfiguration(
        project_dir=output_dir,
        logging_dir=os.path.join(output_dir, "logs"),
    )
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps or config["models"]["controlnet"]["gradient_accumulation_steps"],
        mixed_precision=config["models"]["controlnet"]["mixed_precision"],
        log_with=config["models"]["controlnet"]["tracker"],
        project_config=project_config,
    )
    
    # Make one log on every process with the configuration for debugging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    # Setup TF32 if requested
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    
    # Load or create models
    models = create_controlnet_model(config, accelerator)
    tokenizer = models["tokenizer"]
    text_encoder = models["text_encoder"]
    vae = models["vae"]
    unet = models["unet"]
    controlnet = models["controlnet"]
    
    # Enable xFormers if requested
    if args.enable_xformers or config["models"]["controlnet"]["enable_xformers"]:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            logger.warning("xFormers not available, skipping memory efficient attention")
    
    # Create noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")
    
    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    
    # Enable training for controlnet
    controlnet.train()
    
    # Set up optimizer
    if args.use_8bit_adam or config["models"]["controlnet"]["use_8bit_adam"]:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
        except ImportError:
            logger.warning("bitsandbytes not available, using standard AdamW")
            optimizer_cls = torch.optim.AdamW
    else:
        optimizer_cls = torch.optim.AdamW
    
    # Create dataset
    train_dataset = BraTSControlNetDataset(
        data_root=data_dir,
        tokenizer=tokenizer,
        size=args.resolution or config["models"]["controlnet"]["resolution"],
        modality=args.modality or "t1ce",
        text_file=args.text_prompt_file,
    )
    
    # Create dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size or config["models"]["controlnet"]["train_batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
    )
    
    # Get number of training steps
    if args.max_train_steps:
        num_train_steps = args.max_train_steps
    else:
        num_train_epochs = args.num_train_epochs or config["models"]["controlnet"]["num_train_epochs"]
        num_update_steps_per_epoch = len(train_dataloader)
        num_train_steps = num_train_epochs * num_update_steps_per_epoch
    
    # Create learning rate scheduler
    lr_scheduler = get_scheduler(
        name=config["models"]["controlnet"]["lr_scheduler"],
        optimizer=optimizer_cls(
            controlnet.parameters(),
            lr=args.learning_rate or config["models"]["controlnet"]["learning_rate"],
            betas=(0.9, 0.999),
            weight_decay=1e-2,
            eps=1e-8,
        ),
        num_warmup_steps=config["models"]["controlnet"]["lr_warmup_steps"],
        num_training_steps=num_train_steps,
    )
    
    # Prepare everything with accelerator
    controlnet, lr_scheduler = accelerator.prepare(controlnet, lr_scheduler)
    
    # Move text_encoder and vae to device
    text_encoder.to(accelerator.device)
    vae.to(accelerator.device)
    unet.to(accelerator.device)
    
    # Track training progress
    if accelerator.is_main_process:
        accelerator.init_trackers("controlnet_training")
    
    # Training loop
    total_batch_size = (
        args.train_batch_size or config["models"]["controlnet"]["train_batch_size"]
    ) * accelerator.num_processes * (
        args.gradient_accumulation_steps or config["models"]["controlnet"]["gradient_accumulation_steps"]
    )
    
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Batch size per device = {args.train_batch_size or config['models']['controlnet']['train_batch_size']}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps or config['models']['controlnet']['gradient_accumulation_steps']}")
    logger.info(f"  Total optimization steps = {num_train_steps}")
    
    global_step = 0
    first_epoch = 0
    
    progress_bar = tqdm(
        range(global_step, num_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")
    
    for epoch in range(first_epoch, int(args.num_train_epochs or config["models"]["controlnet"]["num_train_epochs"])):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet):
                # Convert images to latent space
                with torch.no_grad():
                    latents = vae.encode(batch["pixel_values"].to(dtype=vae.dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                
                # Sample noise
                noise = torch.randn_like(latents)
                batch_size = latents.shape[0]
                
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, 
                    noise_scheduler.config.num_train_timesteps, 
                    (batch_size,), 
                    device=latents.device
                )
                timesteps = timesteps.long()
                
                # Add noise to the latents according to the noise magnitude at each timestep
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Get the text embedding for conditioning
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                
                # Get the additional conditioning image (segmentation)
                controlnet_image = batch["conditioning_pixel_values"].to(dtype=controlnet.dtype)
                
                # Forward pass
                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                )
                
                # Calculate model's prediction of the noise as a baseline
                with torch.no_grad():
                    model_pred = unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                    ).sample
                
                # Calculate the loss
                loss = F.mse_loss(model_pred, noise, reduction="mean")
                
                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(controlnet.parameters(), 1.0)
                lr_scheduler.step()
                accelerator.clip_grad_norm_(text_encoder.parameters(), 1.0)
                accelerator.clip_grad_norm_(vae.parameters(), 1.0)
                optimizer_cls(text_encoder.parameters(), lr=lr_scheduler.get_last_lr()[0], weight_decay=0).step()
                optimizer_cls(vae.parameters(), lr=lr_scheduler.get_last_lr()[0], weight_decay=0).step()
            
            # Update progress bar
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                # Log metrics
                if global_step % config["models"]["controlnet"]["validation_steps"] == 0:
                    if accelerator.is_main_process:
                        log_validation(
                            config, 
                            models, 
                            accelerator, 
                            global_step, 
                            epoch
                        )
                
                # Save checkpoint
                if global_step % config["models"]["controlnet"]["checkpointing_steps"] == 0:
                    if accelerator.is_main_process:
                        save_checkpoint(
                            config, 
                            models, 
                            accelerator, 
                            global_step, 
                            epoch
                        )
            
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            
            if global_step >= num_train_steps:
                break
    
    # Create the pipeline using the trained controlnet
    if accelerator.is_main_process:
        logger.info("Training completed, saving final model...")
        accelerator.wait_for_everyone()
        
        # Extract controlnet weights
        unwrapped_controlnet = accelerator.unwrap_model(controlnet)
        
        # Save the trained controlnet
        final_path = os.path.join(output_dir, "controlnet_final")
        unwrapped_controlnet.save_pretrained(final_path)
        
        logger.info(f"ControlNet model saved to {final_path}")
        
        # Create and save a sample pipeline
        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            base_model,
            controlnet=unwrapped_controlnet,
            safety_checker=None,
        )
        pipeline.save_pretrained(os.path.join(output_dir, "pipeline"))
        
        # Generate and save a few samples
        pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
        
        logger.info("Training complete!")


def log_validation(config, models, accelerator, global_step, epoch):
    """Generate and log validation images."""
    logger.info("Running validation...")
    
    # Create validation pipeline
    validation_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        config["model"]["pretrained_model_name_or_path"],
        controlnet=accelerator.unwrap_model(models["controlnet"]),
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    validation_pipeline.scheduler = UniPCMultistepScheduler.from_config(validation_pipeline.scheduler.config)
    validation_pipeline.set_progress_bar_config(disable=True)
    validation_pipeline.to(accelerator.device)
    
    # Define test prompts and conditions
    test_prompts = [
        "MRI scan of brain with glioblastoma in the left frontal lobe with surrounding edema",
        "MRI scan of brain with meningioma in the right temporal lobe",
        "MRI scan of brain with metastatic tumor in the cerebellum",
        "MRI scan of brain with low-grade glioma and minimal mass effect",
    ]
    
    # Create a simple conditioning image (circular mask)
    validation_image = Image.new("RGB", (512, 512), color="black")
    # Draw a simple circle in the center
    from PIL import ImageDraw
    draw = ImageDraw.Draw(validation_image)
    draw.ellipse((200, 200, 300, 300), fill="white")
    
    # Generate images
    all_images = []
    for prompt in test_prompts:
        with torch.autocast("cuda"):
            images = validation_pipeline(
                prompt,
                validation_image,
                num_inference_steps=30,
                guidance_scale=7.5,
                generator=torch.manual_seed(42),
            ).images
            all_images.extend(images)
    
    # Save images
    validation_dir = os.path.join(config["training"]["output_dir"], "validation")
    os.makedirs(validation_dir, exist_ok=True)
    
    for i, image in enumerate(all_images):
        image.save(os.path.join(validation_dir, f"epoch_{epoch}_step_{global_step}_sample_{i}.png"))
    
    # Log images to wandb if available
    if is_wandb_available() and accelerator.is_main_process:
        import wandb
        if wandb.run is not None:
            wandb.log({"validation": [wandb.Image(img) for img in all_images]})


def save_checkpoint(config, models, accelerator, global_step, epoch):
    """Save checkpoint of the model"""
    # Wait for all processes to synchronize
    accelerator.wait_for_everyone()
    
    # Save only from the main process
    if accelerator.is_main_process:
        controlnet = accelerator.unwrap_model(models["controlnet"])
        
        # Get checkpoint name
        checkpoint_name = f"checkpoint-{global_step}"
        checkpoint_path = os.path.join(config["training"]["output_dir"], checkpoint_name)
        os.makedirs(checkpoint_path, exist_ok=True)
        
        logger.info(f"Saving checkpoint to {checkpoint_path}")
        
        # Save the controlnet
        controlnet.save_pretrained(checkpoint_path)
        
        # Save training args
        with open(os.path.join(checkpoint_path, "training_config.yaml"), "w") as f:
            yaml.dump(config, f)


if __name__ == "__main__":
    main() 