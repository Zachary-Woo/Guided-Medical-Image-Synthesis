import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
import os
from pathlib import Path
import json
from datetime import datetime
import cv2
from PIL import Image
import wandb
from torch.utils.tensorboard import SummaryWriter

class Logger:
    """
    Flexible logger with support for console, file, and ML platforms
    (TensorBoard and Weights & Biases).
    """
    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: str = "mri-generation",
        wandb_entity: Optional[str] = None,
        log_level: int = logging.INFO,
        save_code: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of the experiment
            use_tensorboard: Whether to use TensorBoard
            use_wandb: Whether to use Weights & Biases
            wandb_project: W&B project name
            wandb_entity: W&B entity (team or username)
            log_level: Logging level
            save_code: Whether to save code for experiment reproducibility
            config: Configuration dictionary for the experiment
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb
        
        # Create timestamp for the experiment
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.log_dir / f"{self.experiment_name}_{self.timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging to file and console
        self.setup_logging(log_level)
        
        # Save configuration
        self.config = config if config else {}
        self.save_config()
        
        # Set up TensorBoard
        if self.use_tensorboard:
            self.tb_writer = SummaryWriter(log_dir=str(self.experiment_dir / "tensorboard"))
        
        # Set up Weights & Biases
        if self.use_wandb:
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=f"{self.experiment_name}_{self.timestamp}",
                config=self.config
            )
            
        # Save code for reproducibility if requested
        if save_code:
            self.save_source_code()
            
        logging.info(f"Initialized logger for experiment: {self.experiment_name}_{self.timestamp}")
        
    def setup_logging(self, log_level: int):
        """Set up logging to file and console."""
        # Create log file
        log_file = self.experiment_dir / "experiment.log"
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def save_config(self):
        """Save experiment configuration to JSON file."""
        config_file = self.experiment_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump(self.config, f, indent=2)
    
    def save_source_code(self):
        """Save source code files for reproducibility."""
        import shutil
        import glob
        
        # Create code directory
        code_dir = self.experiment_dir / "code"
        code_dir.mkdir(exist_ok=True)
        
        # Directories to save (adjust as needed)
        dirs_to_save = ["version3"]
        
        # Copy Python files
        for dir_name in dirs_to_save:
            for py_file in glob.glob(f"{dir_name}/**/*.py", recursive=True):
                # Create directory structure
                rel_path = os.path.relpath(py_file, os.path.dirname(os.path.dirname(py_file)))
                target_dir = code_dir / os.path.dirname(rel_path)
                target_dir.mkdir(exist_ok=True, parents=True)
                
                # Copy file
                shutil.copy2(py_file, target_dir)
    
    def log_scalar(self, name: str, value: float, step: int):
        """
        Log scalar value.
        
        Args:
            name: Name of the metric
            value: Scalar value
            step: Training step or epoch
        """
        # Log to console/file
        logging.info(f"{name}: {value} (step {step})")
        
        # Log to TensorBoard
        if self.use_tensorboard:
            self.tb_writer.add_scalar(name, value, step)
        
        # Log to Weights & Biases
        if self.use_wandb:
            wandb.log({name: value}, step=step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """
        Log multiple scalars under a common tag.
        
        Args:
            main_tag: Main tag for the group of scalars
            tag_scalar_dict: Dictionary of tag-value pairs
            step: Training step or epoch
        """
        # Log to console/file
        logging.info(f"{main_tag}: {tag_scalar_dict} (step {step})")
        
        # Log to TensorBoard
        if self.use_tensorboard:
            self.tb_writer.add_scalars(main_tag, tag_scalar_dict, step)
        
        # Log to Weights & Biases
        if self.use_wandb:
            wandb_dict = {f"{main_tag}/{k}": v for k, v in tag_scalar_dict.items()}
            wandb.log(wandb_dict, step=step)
    
    def log_images(
        self, 
        name: str, 
        images: Union[torch.Tensor, np.ndarray, List[np.ndarray]],
        step: int,
        caption: Optional[List[str]] = None
    ):
        """
        Log images.
        
        Args:
            name: Name of the image group
            images: Images to log (tensor, numpy array, or list of numpy arrays)
            step: Training step or epoch
            caption: Optional list of captions
        """
        # Convert tensor to numpy if necessary
        if isinstance(images, torch.Tensor):
            if images.dim() == 4:  # NCHW format
                images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
            else:  # CHW format
                images = images.detach().cpu().permute(1, 2, 0).numpy()
        
        # Ensure images are in list format
        if isinstance(images, np.ndarray) and images.ndim == 3:
            images = [images]
        
        # Log to TensorBoard
        if self.use_tensorboard:
            if isinstance(images, list):
                for i, img in enumerate(images):
                    img_caption = caption[i] if caption and i < len(caption) else f"{name}_{i}"
                    
                    # Normalize image if not in [0, 1] or [0, 255]
                    if img.max() > 1.0 and img.max() <= 255:
                        img = img / 255.0
                    
                    # Ensure image has proper channels
                    if img.ndim == 2:  # Add channel dimension for grayscale
                        img = np.expand_dims(img, axis=2)
                    
                    # Convert to CHW format
                    img = np.transpose(img, (2, 0, 1))
                    
                    self.tb_writer.add_image(f"{name}/{img_caption}", img, step)
            else:
                self.tb_writer.add_images(name, images, step)
        
        # Log to Weights & Biases
        if self.use_wandb:
            wandb_images = []
            
            for i, img in enumerate(images):
                # Normalize image if needed
                if img.max() > 1.0 and img.max() <= 255:
                    img = img / 255.0
                
                # Create wandb Image
                img_caption = caption[i] if caption and i < len(caption) else f"{name}_{i}"
                wandb_images.append(wandb.Image(img, caption=img_caption))
            
            wandb.log({name: wandb_images}, step=step)
        
        # Save images to disk
        image_dir = self.experiment_dir / "images" / f"step_{step}"
        image_dir.mkdir(parents=True, exist_ok=True)
        
        for i, img in enumerate(images):
            # Ensure values are in [0, 255]
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            
            # Save image
            img_caption = caption[i] if caption and i < len(caption) else f"{i}"
            img_path = image_dir / f"{name}_{img_caption}.png"
            
            # Convert to RGB if grayscale
            if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
            # Save using PIL to ensure compatibility
            Image.fromarray(img).save(img_path)
    
    def log_histogram(self, name: str, values: Union[torch.Tensor, np.ndarray], step: int):
        """
        Log histogram of values.
        
        Args:
            name: Name of the histogram
            values: Values to log
            step: Training step or epoch
        """
        # Convert tensor to numpy if necessary
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        
        # Log to TensorBoard
        if self.use_tensorboard:
            self.tb_writer.add_histogram(name, values, step)
        
        # Log to Weights & Biases
        if self.use_wandb:
            wandb.log({name: wandb.Histogram(values)}, step=step)
    
    def log_model(
        self, 
        model: torch.nn.Module, 
        input_size: Optional[Tuple[int, ...]] = None,
    ):
        """
        Log model architecture.
        
        Args:
            model: PyTorch model
            input_size: Input size for visualization (e.g., (1, 3, 224, 224))
            name: Name of the model
        """
        # Log to TensorBoard
        if self.use_tensorboard and input_size:
            self.tb_writer.add_graph(model, torch.zeros(input_size))
        
        # Log to Weights & Biases
        if self.use_wandb:
            wandb.watch(model)
    
    def save_checkpoint(
        self, 
        state_dict: Dict[str, Any], 
        is_best: bool = False, 
        filename: str = "checkpoint.pth"
    ):
        """
        Save model checkpoint.
        
        Args:
            state_dict: State dictionary to save
            is_best: Whether this is the best model so far
            filename: Filename for the checkpoint
        """
        checkpoint_dir = self.experiment_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        # Save checkpoint
        checkpoint_path = checkpoint_dir / filename
        torch.save(state_dict, checkpoint_path)
        
        # Save as best model if specified
        if is_best:
            best_path = checkpoint_dir / "best_model.pth"
            torch.save(state_dict, best_path)
            logging.info(f"Saved best model to {best_path}")
        
        logging.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Log to Weights & Biases
        if self.use_wandb:
            artifacts_dir = wandb.run.dir
            os.makedirs(artifacts_dir, exist_ok=True)
            
            # Save a copy in the W&B artifacts directory
            wandb_path = os.path.join(artifacts_dir, filename)
            torch.save(state_dict, wandb_path)
            
            # Log as artifact
            artifact = wandb.Artifact(
                name=f"{self.experiment_name}_checkpoint",
                type="model",
                description=f"Model checkpoint at step {state_dict.get('step', 'unknown')}"
            )
            artifact.add_file(wandb_path)
            wandb.log_artifact(artifact)
    
    def close(self):
        """Close logger and associated resources."""
        if self.use_tensorboard:
            self.tb_writer.close()
        
        if self.use_wandb:
            wandb.finish()
        
        logging.info(f"Closed logger for experiment: {self.experiment_name}_{self.timestamp}")

def visualize_batch(
    samples: Dict[str, torch.Tensor],
    max_images: int = 8,
    denormalize: bool = True,
    save_path: Optional[str] = None,
    show: bool = True,
    title: Optional[str] = None
) -> np.ndarray:
    """
    Visualize a batch of samples.
    
    Args:
        samples: Dictionary with keys like 'image', 'mask', 'generated', etc.
        max_images: Maximum number of images to display
        denormalize: Whether to denormalize images from [0,1] to [0,255]
        save_path: Path to save the visualization
        show: Whether to display the visualization
        title: Title for the figure
        
    Returns:
        Numpy array containing the visualization grid
    """
    # Get number of images to display
    batch_size = next(iter(samples.values())).shape[0]
    n_images = min(batch_size, max_images)
    
    # Get number of columns (one per sample key)
    n_cols = len(samples)
    
    # Create figure
    fig, axes = plt.subplots(n_images, n_cols, figsize=(3*n_cols, 3*n_images))
    
    # Set title if provided
    if title:
        fig.suptitle(title, fontsize=16)
    
    # Handle case with single image or single column
    if n_images == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_images == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Process each sample
    for i in range(n_images):
        for j, (key, tensor) in enumerate(samples.items()):
            # Get current axis
            ax = axes[i, j]
            
            # Convert tensor to numpy
            if isinstance(tensor, torch.Tensor):
                if tensor.dim() == 4:  # NCHW format
                    image = tensor[i].detach().cpu().permute(1, 2, 0).numpy()
                else:  # Already single image
                    image = tensor.detach().cpu().permute(1, 2, 0).numpy()
            else:
                image = tensor[i]
            
            # Denormalize if needed
            if denormalize and image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            
            # Handle different channel configurations
            if image.shape[-1] == 1:  # Grayscale
                image = image.squeeze(-1)
                ax.imshow(image, cmap='gray')
            else:  # RGB/RGBA
                ax.imshow(image)
            
            # Set title and turn off axis
            ax.set_title(key)
            ax.axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()
    
    # Convert to numpy array for return
    fig.canvas.draw()
    vis_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    vis_image = vis_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    return vis_image

def plot_training_curves(
    metrics: Dict[str, List[float]],
    save_path: Optional[str] = None,
    show: bool = True,
    title: Optional[str] = None
) -> None:
    """
    Plot training curves from metrics.
    
    Args:
        metrics: Dictionary with metric names as keys and lists of values
        save_path: Path to save the plot
        show: Whether to display the plot
        title: Title for the plot
    """
    # Get number of metrics
    n_metrics = len(metrics)
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 3*n_metrics))
    
    # Set title if provided
    if title:
        fig.suptitle(title, fontsize=16)
    
    # Handle case with single metric
    if n_metrics == 1:
        axes = [axes]
    
    # Plot each metric
    for i, (metric_name, values) in enumerate(metrics.items()):
        ax = axes[i]
        ax.plot(values)
        ax.set_title(metric_name)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name)
        ax.grid(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()

def create_comparison_grid(
    images: List[np.ndarray],
    captions: List[str],
    rows: int = 1,
    cols: Optional[int] = None,
    fig_size: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
    show: bool = True,
    title: Optional[str] = None
) -> np.ndarray:
    """
    Create a grid of images for comparison.
    
    Args:
        images: List of images (numpy arrays)
        captions: List of captions for each image
        rows: Number of rows in the grid
        cols: Number of columns in the grid (calculated if None)
        fig_size: Figure size (width, height)
        save_path: Path to save the visualization
        show: Whether to display the visualization
        title: Title for the figure
        
    Returns:
        Numpy array containing the visualization grid
    """
    # Determine number of columns if not provided
    if cols is None:
        cols = (len(images) + rows - 1) // rows
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=fig_size)
    
    # Set title if provided
    if title:
        fig.suptitle(title, fontsize=16)
    
    # Handle case with single image
    if rows * cols == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()
    
    # Plot images
    for i, (image, caption) in enumerate(zip(images, captions)):
        if i >= rows * cols:
            break
            
        ax = axes[i]
        
        # Normalize if needed
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # Handle different channel configurations
        if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
            if image.ndim == 3:
                image = image.squeeze(-1)
            ax.imshow(image, cmap='gray')
        else:
            ax.imshow(image)
        
        # Set title and turn off axis
        ax.set_title(caption)
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(len(images), rows * cols):
        axes[i].axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()
    
    # Convert to numpy array for return
    fig.canvas.draw()
    vis_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    vis_image = vis_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    return vis_image 