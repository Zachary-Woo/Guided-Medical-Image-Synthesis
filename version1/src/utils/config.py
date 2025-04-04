"""
Configuration utilities for the project.
"""

import os
import yaml
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class DataConfig:
    """
    Data configuration.
    """
    dataset_name: str
    data_dir: str
    image_size: int = 512
    batch_size: int = 4
    num_workers: int = 4
    mask_folder: Optional[str] = None
    augment_data: bool = True
    val_split: float = 0.1
    test_split: float = 0.1


@dataclass
class ModelConfig:
    """
    Model configuration.
    """
    pretrained_model_id: str = "runwayml/stable-diffusion-v1-5"
    controlnet_id: Optional[str] = None
    use_lora: bool = False
    lora_rank: int = 4
    trainable_modules: List[str] = field(default_factory=lambda: [
        "down_blocks", "mid_block", "up_blocks"
    ])
    mixed_precision: str = "fp16"


@dataclass
class TrainingConfig:
    """
    Training configuration (primarily for downstream evaluation now).
    """
    output_dir: str = "output"
    # learning_rate: float = 1e-5 # For ControlNet fine-tuning (unused)
    downstream_learning_rate: float = 1e-4 # LR for downstream task model
    # num_epochs: int = 100 # For ControlNet fine-tuning (unused)
    downstream_num_epochs: int = 20 # Default epochs for downstream task
    # save_steps: int = 500 # For ControlNet fine-tuning (unused)
    # eval_steps: int = 100 # For ControlNet fine-tuning (unused)
    # gradient_accumulation_steps: int = 1 # For ControlNet fine-tuning (unused)
    seed: int = 42
    log_wandb: bool = False # For downstream task? Or remove?
    wandb_project: str = "medical-controlnet" # Maybe rename to downstream_wandb_project?
    wandb_run_name: Optional[str] = None # Maybe rename to downstream_wandb_run_name?


@dataclass
class InferenceConfig:
    """
    Inference configuration.
    """
    controlnet_inference_id: str = "lllyasviel/sd-controlnet-canny"
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    num_images_per_prompt: int = 1
    conditioning_type: str = "canny"
    prompts: List[str] = field(default_factory=lambda: [
        "A high-quality MRI scan of the brain showing a tumor",
        "A clear histopathology image showing cancer cells",
        "A chest X-ray showing pneumonia"
    ])


@dataclass
class Config:
    """
    Main configuration class.
    """
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    inference: InferenceConfig
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """
        Create a Config object from a dictionary.
        
        Args:
            config_dict (Dict[str, Any]): Configuration dictionary
            
        Returns:
            Config: Configuration object
        """
        data_config = DataConfig(**config_dict["data"])
        model_config = ModelConfig(**config_dict["model"])
        training_config = TrainingConfig(**config_dict["training"])
        inference_config = InferenceConfig(**config_dict["inference"])
        
        return cls(
            data=data_config,
            model=model_config,
            training=training_config,
            inference=inference_config
        )
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """
        Create a Config object from a YAML file.
        
        Args:
            yaml_path (str): Path to YAML file
            
        Returns:
            Config: Configuration object
        """
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert Config object to a dictionary.
        
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        return {
            "data": self.data.__dict__,
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "inference": self.inference.__dict__
        }
    
    def save(self, yaml_path: str) -> None:
        """
        Save Config object to a YAML file.
        
        Args:
            yaml_path (str): Path to YAML file
        """
        config_dict = self.to_dict()
        
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)


def create_default_config(dataset_name: str, data_dir: str) -> Config:
    """
    Create a default configuration.
    
    Args:
        dataset_name (str): Name of the dataset
        data_dir (str): Directory containing the dataset
        
    Returns:
        Config: Default configuration
    """
    return Config(
        data=DataConfig(
            dataset_name=dataset_name,
            data_dir=data_dir
        ),
        model=ModelConfig(),
        training=TrainingConfig(),
        inference=InferenceConfig()
    )


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Medical ControlNet training")
    
    # Config arguments
    parser.add_argument("--config", type=str, help="Path to configuration file")
    
    # Data arguments
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset")
    parser.add_argument("--data_dir", type=str, help="Directory containing the dataset")
    parser.add_argument("--image_size", type=int, default=512, help="Image size")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    
    # Model arguments
    parser.add_argument("--pretrained_model_id", type=str, default="runwayml/stable-diffusion-v1-5", 
                        help="ID of the pretrained Stable Diffusion model")
    parser.add_argument("--controlnet_id", type=str, help="ID of a pretrained ControlNet model")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for fine-tuning")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log_wandb", action="store_true", help="Log to Weights & Biases")
    
    return parser.parse_args()


def get_config_from_args(args: argparse.Namespace) -> Config:
    """
    Get configuration from command line arguments.
    
    Args:
        args (argparse.Namespace): Parsed arguments
        
    Returns:
        Config: Configuration object
    """
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        if not args.dataset_name or not args.data_dir:
            raise ValueError("Either --config or --dataset_name and --data_dir must be provided")
        
        config = create_default_config(args.dataset_name, args.data_dir)
    
    # Override config with command line arguments
    if args.image_size:
        config.data.image_size = args.image_size
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.pretrained_model_id:
        config.model.pretrained_model_id = args.pretrained_model_id
    if args.controlnet_id:
        config.model.controlnet_id = args.controlnet_id
    if args.use_lora:
        config.model.use_lora = args.use_lora
    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.num_epochs:
        config.training.num_epochs = args.num_epochs
    if args.seed:
        config.training.seed = args.seed
    if args.log_wandb:
        config.training.log_wandb = args.log_wandb
    
    return config 