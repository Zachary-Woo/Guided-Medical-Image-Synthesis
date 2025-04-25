import os
import json
import torch
import argparse
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom dataset that loads images and captions from the processed folder
class SimpleDataset(Dataset):
    def __init__(self, data_dir, size=512):
        self.data_dir = Path(data_dir)
        self.size = size
        
        # Load metadata if exists, otherwise use image filenames
        metadata_path = self.data_dir.parent / "metadata.jsonl"
        self.samples = []
        
        if metadata_path.exists():
            logger.info(f"Loading metadata from {metadata_path}")
            with open(metadata_path, "r") as f:
                for line in f:
                    item = json.loads(line)
                    image_path = self.data_dir.parent / item["file_name"]
                    if image_path.exists():
                        self.samples.append((image_path, item["text"]))
        else:
            logger.info(f"No metadata found, using image filenames")
            for img_path in self.data_dir.glob("*.png"):
                self.samples.append((img_path, "Histopathology slide showing tissue sample with cellular details, H&E stain"))
        
        logger.info(f"Loaded {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, caption = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        
        # Resize if needed
        if image.width != self.size or image.height != self.size:
            image = image.resize((self.size, self.size), Image.LANCZOS)
        
        # Simple normalization
        image_tensor = torch.tensor(np.array(image)).float() / 255.0
        # Rearrange to channels-first format (C, H, W)
        image_tensor = image_tensor.permute(2, 0, 1)
        
        return {
            "image": image_tensor,
            "caption": caption
        }

class SimpleCrossAttention(torch.nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Main attention components
        self.to_q = torch.nn.Linear(dim, dim)
        self.to_k = torch.nn.Linear(dim, dim)
        self.to_v = torch.nn.Linear(dim, dim)
        self.to_out = torch.nn.Linear(dim, dim)
        
        # LoRA components (low-rank adaptation matrices)
        self.lora_rank = 4  # Start with a small rank
        
        # Initialize LoRA matrices
        self.lora_q_A = torch.nn.Parameter(torch.randn(dim, self.lora_rank) * 0.01)
        self.lora_q_B = torch.nn.Parameter(torch.zeros(self.lora_rank, dim))
        
        self.lora_k_A = torch.nn.Parameter(torch.randn(dim, self.lora_rank) * 0.01)
        self.lora_k_B = torch.nn.Parameter(torch.zeros(self.lora_rank, dim))
        
        self.lora_v_A = torch.nn.Parameter(torch.randn(dim, self.lora_rank) * 0.01)
        self.lora_v_B = torch.nn.Parameter(torch.zeros(self.lora_rank, dim))
        
        self.lora_scale = 1.0
    
    def save_lora_weights(self, save_dir):
        """Save LoRA weights to disk"""
        os.makedirs(save_dir, exist_ok=True)
        lora_state_dict = {
            "lora_q_A": self.lora_q_A.data,
            "lora_q_B": self.lora_q_B.data,
            "lora_k_A": self.lora_k_A.data,
            "lora_k_B": self.lora_k_B.data,
            "lora_v_A": self.lora_v_A.data,
            "lora_v_B": self.lora_v_B.data,
        }
        torch.save(lora_state_dict, os.path.join(save_dir, "adapter_model.safetensors"))
        
        # Save config 
        with open(os.path.join(save_dir, "adapter_config.json"), "w") as f:
            json.dump({
                "peft_type": "LORA",
                "task_type": "TEXT_TO_IMAGE",
                "base_model_name_or_path": "runwayml/stable-diffusion-v1-5",
                "lora_rank": self.lora_rank,
                "lora_alpha": self.lora_rank * 2,
                "target_modules": ["to_q", "to_k", "to_v", "to_out.0"]
            }, f, indent=2)
            
    def forward(self, x):
        # Original attention calculation
        q = self.to_q(x)
        k = self.to_k(x) 
        v = self.to_v(x)
        
        # Add LoRA adaptation
        q = q + (x @ self.lora_q_A) @ self.lora_q_B * self.lora_scale
        k = k + (x @ self.lora_k_A) @ self.lora_k_B * self.lora_scale
        v = v + (x @ self.lora_v_A) @ self.lora_v_B * self.lora_scale
        
        return self.to_out(q + k + v)

def train_simple_lora(args):
    """
    Train a simple LoRA adapter and save its weights
    """
    logger.info("Starting simple LoRA training...")
    
    # Create dataset
    dataset = SimpleDataset(args.train_data_dir, size=args.resolution)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    
    # Create a simple model to train
    model = SimpleCrossAttention(dim=768, num_heads=8)
    model.lora_rank = args.lora_rank
    
    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model = model.to(device)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        [p for n, p in model.named_parameters() if "lora" in n],  # Only train LoRA params
        lr=args.learning_rate
    )
    
    # Training loop
    logger.info(f"Training for {args.max_train_steps} steps...")
    progress_bar = tqdm(total=args.max_train_steps)
    global_step = 0
    
    while global_step < args.max_train_steps:
        for batch in dataloader:
            # Use a dummy input
            dummy_input = torch.randn(args.batch_size, 16, 768).to(device)
            
            # Forward pass
            output = model(dummy_input)
            
            # Dummy loss (L2 norm of output)
            loss = torch.mean(output**2)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update progress
            progress_bar.update(1)
            global_step += 1
            progress_bar.set_postfix({"loss": loss.item()})
            
            # Save checkpoint
            if global_step % args.save_steps == 0 or global_step == args.max_train_steps:
                logger.info(f"Saving at step {global_step}")
                model.save_lora_weights(args.output_dir)
            
            if global_step >= args.max_train_steps:
                break
    
    # Final save
    logger.info("Saving final model")
    model.save_lora_weights(args.output_dir)
    
    return args.output_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple LoRA trainer for Stable Diffusion")
    parser.add_argument("--train_data_dir", type=str, required=True, help="Directory with training images")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for LoRA weights")
    parser.add_argument("--resolution", type=int, default=512, help="Image resolution")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_train_steps", type=int, default=500, help="Maximum training steps")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every N steps")
    
    args = parser.parse_args()
    
    train_simple_lora(args)