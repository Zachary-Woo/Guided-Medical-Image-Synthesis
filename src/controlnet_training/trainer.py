"""
Trainer for ControlNet medical image synthesis.
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator
import wandb


class ControlNetTrainer:
    """
    Trainer for ControlNet medical image synthesis.
    """
    
    def __init__(
        self,
        model_components,
        train_dataloader,
        val_dataloader=None,
        optimizer=None,
        scheduler=None,
        device="cuda",
        output_dir="./output",
        log_wandb=False,
        mixed_precision="fp16",
        gradient_accumulation_steps=1
    ):
        """
        Initialize the trainer.
        
        Args:
            model_components (dict): Dictionary of model components
            train_dataloader (DataLoader): Training data loader
            val_dataloader (DataLoader, optional): Validation data loader
            optimizer (torch.optim.Optimizer, optional): Optimizer
            scheduler (lr_scheduler, optional): Learning rate scheduler
            device (str): Device to train on
            output_dir (str): Directory to save outputs
            log_wandb (bool): Whether to log to Weights & Biases
            mixed_precision (str): Mixed precision training type
            gradient_accumulation_steps (int): Number of steps to accumulate gradients
        """
        self.tokenizer = model_components["tokenizer"]
        self.text_encoder = model_components["text_encoder"]
        self.unet = model_components["unet"]
        self.controlnet = model_components["controlnet"]
        self.noise_scheduler = model_components["noise_scheduler"]
        
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # Setup optimizer if not provided
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.controlnet.parameters(),
                lr=1e-5
            )
        else:
            self.optimizer = optimizer
            
        self.scheduler = scheduler
        self.device = device
        self.output_dir = output_dir
        self.log_wandb = log_wandb
        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Create accelerator for mixed precision and distributed training
        self.accelerator = Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=gradient_accumulation_steps
        )
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Init W&B logging
        if log_wandb:
            wandb.init(project="medical-controlnet", name="controlnet-training")
    
    def train(self, num_epochs=100, save_steps=500):
        """
        Train the ControlNet model.
        
        Args:
            num_epochs (int): Number of epochs to train for
            save_steps (int): Number of steps between saving checkpoints
        """
        # Prepare all components for accelerated training
        self.controlnet, self.optimizer, self.train_dataloader = self.accelerator.prepare(
            self.controlnet, self.optimizer, self.train_dataloader
        )
        
        # Move models to device
        self.text_encoder.to(self.accelerator.device)
        self.unet.to(self.accelerator.device)
        
        # Set models to eval mode as we don't want to train them
        self.text_encoder.eval()
        self.unet.eval()
        self.controlnet.train()
        
        global_step = 0
        for epoch in range(num_epochs):
            progress_bar = tqdm(total=len(self.train_dataloader), disable=not self.accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")
            
            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.controlnet):
                    # Get input images and conditioning images (masks or edge maps)
                    input_images = batch["images"].to(self.accelerator.device)
                    conditioning_images = batch["conditioning"].to(self.accelerator.device)
                    
                    # Encode text prompts
                    text_inputs = self.tokenizer(
                        batch["prompts"],
                        padding="max_length",
                        max_length=self.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt"
                    ).to(self.accelerator.device)
                    
                    with torch.no_grad():
                        text_embeddings = self.text_encoder(text_inputs.input_ids)[0]
                    
                    # Add noise to the input images
                    noise = torch.randn_like(input_images)
                    batch_size = input_images.shape[0]
                    timesteps = torch.randint(
                        0, self.noise_scheduler.config.num_train_timesteps, 
                        (batch_size,), device=self.accelerator.device
                    ).long()
                    
                    noisy_images = self.noise_scheduler.add_noise(
                        input_images, noise, timesteps
                    )
                    
                    # Get ControlNet output
                    controlnet_output = self.controlnet(
                        noisy_images,
                        timesteps,
                        encoder_hidden_states=text_embeddings,
                        controlnet_cond=conditioning_images,
                        return_dict=False
                    )
                    
                    # Predict the noise residual with the UNet
                    with torch.no_grad():
                        unet_output = self.unet(
                            noisy_images,
                            timesteps,
                            encoder_hidden_states=text_embeddings,
                            down_block_additional_residuals=controlnet_output[0],
                            mid_block_additional_residual=controlnet_output[1]
                        ).sample
                    
                    # Calculate the loss
                    loss = F.mse_loss(unet_output, noise)
                    
                    # Backpropagate
                    self.accelerator.backward(loss)
                    
                    # Update parameters
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.optimizer.zero_grad()
                
                # Update progress
                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), "step": global_step}
                progress_bar.set_postfix(**logs)
                global_step += 1
                
                # Log to wandb
                if self.log_wandb and global_step % 10 == 0:
                    wandb.log(logs)
                
                # Save checkpoint
                if global_step % save_steps == 0:
                    self.save_checkpoint(f"checkpoint-{global_step}")
            
            # Save checkpoint after each epoch
            self.save_checkpoint(f"checkpoint-epoch-{epoch}")
    
    def save_checkpoint(self, checkpoint_name):
        """
        Save a checkpoint of the model.
        
        Args:
            checkpoint_name (str): Name of the checkpoint
        """
        checkpoint_dir = os.path.join(self.output_dir, checkpoint_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save using accelerator to handle distributed training
        unwrapped_controlnet = self.accelerator.unwrap_model(self.controlnet)
        unwrapped_controlnet.save_pretrained(checkpoint_dir)
        
        print(f"Saved checkpoint to {checkpoint_dir}")
    
    def generate(self, prompt, control_image, num_inference_steps=50, guidance_scale=7.5):
        """
        Generate an image using the trained ControlNet.
        
        Args:
            prompt (str): Text prompt for generation
            control_image (torch.Tensor): Control image
            num_inference_steps (int): Number of inference steps
            guidance_scale (float): Guidance scale for classifier-free guidance
            
        Returns:
            PIL.Image: Generated image
        """
        from diffusers import StableDiffusionControlNetPipeline
        
        # Create pipeline for inference
        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=self.accelerator.unwrap_model(self.controlnet),
            torch_dtype=torch.float16
        ).to(self.device)
        
        # Enable attention slicing for lower memory usage
        pipeline.enable_attention_slicing()
        
        # Generate image
        image = pipeline(
            prompt,
            control_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        ).images[0]
        
        return image 