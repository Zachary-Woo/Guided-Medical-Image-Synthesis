import os
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any
import json
from pathlib import Path
import re
import yaml
from diffusers import (
    StableDiffusionPipeline,
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    DDPMScheduler,
    DDIMScheduler,
    PNDMScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
from peft import LoraConfig, get_peft_model


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from a YAML or JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if config_path.suffix.lower() == '.json':
        with open(config_path, 'r') as f:
            config = json.load(f)
    elif config_path.suffix.lower() in ['.yaml', '.yml']:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
    
    return config


def save_config(config: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """
    Save configuration to a YAML or JSON file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save the configuration
    """
    output_path = Path(output_path)
    
    # Create directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix.lower() == '.json':
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
    elif output_path.suffix.lower() in ['.yaml', '.yml']:
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    else:
        raise ValueError(f"Unsupported configuration file format: {output_path.suffix}")


def load_scheduler(
    scheduler_name: str,
    config: Optional[Dict[str, Any]] = None
) -> Any:
    """
    Load scheduler based on name and configuration.
    
    Args:
        scheduler_name: Name of the scheduler
        config: Optional configuration for the scheduler
        
    Returns:
        Scheduler instance
    """
    scheduler_config = config or {}
    
    scheduler_map = {
        "ddpm": DDPMScheduler,
        "ddim": DDIMScheduler,
        "pndm": PNDMScheduler,
        "euler": EulerDiscreteScheduler,
        "dpm_solver_multistep": DPMSolverMultistepScheduler,
    }
    
    if scheduler_name.lower() not in scheduler_map:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    
    return scheduler_map[scheduler_name.lower()](**scheduler_config)


def load_stable_diffusion_model(
    model_id_or_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    scheduler_name: Optional[str] = None,
    extract_ema: bool = False,
) -> StableDiffusionPipeline:
    """
    Load a Stable Diffusion model from HuggingFace Hub or local path.
    
    Args:
        model_id_or_path: Model ID on HuggingFace Hub or local path
        device: Device to load the model on
        dtype: Model precision
        scheduler_name: Optional name of the scheduler to use
        extract_ema: Whether to extract EMA weights
        
    Returns:
        StableDiffusionPipeline
    """
    kwargs = {
        "torch_dtype": dtype,
        "safety_checker": None,
        "requires_safety_checker": False,
    }
    
    if extract_ema:
        kwargs["extract_ema"] = True
    
    # Load the pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id_or_path,
        **kwargs
    )
    
    # Replace scheduler if specified
    if scheduler_name:
        pipe.scheduler = load_scheduler(scheduler_name)
    
    # Move to device
    pipe = pipe.to(device)
    
    return pipe


def load_controlnet_model(
    controlnet_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> ControlNetModel:
    """
    Load a ControlNet model from HuggingFace Hub or local path.
    
    Args:
        controlnet_path: Model ID on HuggingFace Hub or local path
        device: Device to load the model on
        dtype: Model precision
        
    Returns:
        ControlNetModel
    """
    controlnet = ControlNetModel.from_pretrained(
        controlnet_path,
        torch_dtype=dtype,
    )
    
    # Move to device
    controlnet = controlnet.to(device)
    
    return controlnet


def load_controlnet_pipeline(
    base_model_path: str,
    controlnet_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    scheduler_name: Optional[str] = None,
) -> StableDiffusionControlNetPipeline:
    """
    Load a StableDiffusionControlNetPipeline.
    
    Args:
        base_model_path: Base model ID on HuggingFace Hub or local path
        controlnet_path: ControlNet model ID or path
        device: Device to load the models on
        dtype: Model precision
        scheduler_name: Optional name of the scheduler to use
        
    Returns:
        StableDiffusionControlNetPipeline
    """
    # Load the controlnet
    controlnet = load_controlnet_model(controlnet_path, device, dtype)
    
    # Load the pipeline
    kwargs = {
        "torch_dtype": dtype,
        "safety_checker": None,
        "requires_safety_checker": False,
    }
    
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_path,
        controlnet=controlnet,
        **kwargs,
    )
    
    # Replace scheduler if specified
    if scheduler_name:
        pipe.scheduler = load_scheduler(scheduler_name)
    
    # Move to device
    pipe = pipe.to(device)
    
    return pipe


def apply_lora_adapter(
    model: Any,
    adapter_path: str,
    adapter_name: str = "default",
    scale: float = 1.0,
) -> Any:
    """
    Apply a LoRA adapter to a model.
    
    Args:
        model: The model to apply the LoRA adapter to
        adapter_path: Path to the LoRA adapter
        adapter_name: Name of the adapter
        scale: LoRA scale factor
        
    Returns:
        Model with LoRA adapter applied
    """
    # Load the adapter
    if hasattr(model, "load_lora_weights"):
        # Native diffusers LoRA loading
        model.load_lora_weights(adapter_path, adapter_name=adapter_name)
        
        # Set the LoRA scale
        if hasattr(model, "set_adapters_scale"):
            model.set_adapters_scale(scale)
    else:
        # Use PEFT library for LoRA loading
        lora_config = LoraConfig.from_pretrained(adapter_path)
        model = get_peft_model(model, lora_config)
        
        # Load the weights
        lora_state_dict = torch.load(
            os.path.join(adapter_path, "adapter_model.bin"),
            map_location="cpu"
        )
        model.load_state_dict(lora_state_dict, strict=False)
        
        # Set the LoRA scale
        for module in model.modules():
            if hasattr(module, "scaling"):
                module.scaling = scale
    
    return model


def get_device_and_dtype(
    device: Optional[str] = None,
    dtype: Optional[str] = None,
) -> Tuple[torch.device, torch.dtype]:
    """
    Get device and dtype based on specified strings and available hardware.
    
    Args:
        device: Device string ('cuda', 'cpu', 'mps', etc.)
        dtype: Data type string ('float32', 'float16', 'bfloat16')
        
    Returns:
        Tuple of (device, dtype)
    """
    # Determine device
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)
    
    # Determine dtype
    if dtype is None:
        # Default to float16 for GPU, float32 for CPU
        if device.type == "cuda":
            dtype = torch.float16
        else:
            dtype = torch.float32
    else:
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map.get(dtype, torch.float32)
    
    # Check compatibility
    if dtype == torch.bfloat16 and device.type not in ["cuda", "cpu"]:
        print(f"Warning: bfloat16 not fully supported on {device.type}. Using float16 instead.")
        dtype = torch.float16
    
    return device, dtype


def check_model_compatibility(
    base_model_path: str,
    lora_path: Optional[str] = None,
    controlnet_path: Optional[str] = None,
) -> bool:
    """
    Check compatibility between base model, LoRA, and ControlNet.
    
    Args:
        base_model_path: Path to the base model
        lora_path: Path to the LoRA adapter
        controlnet_path: Path to the ControlNet model
        
    Returns:
        True if compatible, False otherwise
    """
    # Function to extract model version from path
    def extract_version(path):
        # Try to find standard version patterns
        version_patterns = [
            r"v(\d+\.\d+)",
            r"sd-v(\d+\.\d+)",
            r"stable-diffusion-v(\d+\.\d+)",
        ]
        
        for pattern in version_patterns:
            match = re.search(pattern, path.lower())
            if match:
                return match.group(1)
        
        # If no version found in path, try to load config
        try:
            if os.path.isdir(path):
                config_path = os.path.join(path, "model_index.json")
                if os.path.exists(config_path):
                    with open(config_path, "r") as f:
                        config = json.load(f)
                    if "_name_or_path" in config:
                        return extract_version(config["_name_or_path"])
        except:
            pass
        
        return None
    
    # Extract versions
    base_version = extract_version(base_model_path)
    lora_version = extract_version(lora_path) if lora_path else None
    controlnet_version = extract_version(controlnet_path) if controlnet_path else None
    
    # Check compatibility
    if base_version:
        if lora_version and lora_version != base_version:
            print(f"Warning: LoRA version ({lora_version}) may not be compatible with base model ({base_version})")
            return False
        
        if controlnet_version and controlnet_version != base_version:
            print(f"Warning: ControlNet version ({controlnet_version}) may not be compatible with base model ({base_version})")
            return False
    
    return True


def extract_model_components(
    model: nn.Module,
    components: List[str],
) -> Dict[str, nn.Module]:
    """
    Extract specific components from a model.
    
    Args:
        model: PyTorch model
        components: List of component names to extract
        
    Returns:
        Dictionary mapping component names to modules
    """
    results = {}
    
    for component in components:
        if "." in component:
            # Handle nested components with dot notation
            parts = component.split(".")
            current = model
            for part in parts:
                if hasattr(current, part):
                    current = getattr(current, part)
                else:
                    current = None
                    break
            if current is not None:
                results[component] = current
        else:
            # Handle top-level components
            if hasattr(model, component):
                results[component] = getattr(model, component)
    
    return results


def calculate_model_size(model: nn.Module) -> Dict[str, Union[int, float]]:
    """
    Calculate the size and parameter count of a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model size statistics
    """
    # Get number of parameters
    param_size = 0
    param_count = 0
    
    for param in model.parameters():
        param_count += param.numel()
        param_size += param.numel() * param.element_size()
    
    # Calculate buffer size (for non-parameter tensors like running mean/var)
    buffer_size = 0
    buffer_count = 0
    
    for buffer in model.buffers():
        buffer_count += buffer.numel()
        buffer_size += buffer.numel() * buffer.element_size()
    
    # Total size in MB
    total_size_mb = (param_size + buffer_size) / (1024 * 1024)
    
    return {
        "parameter_count": param_count,
        "parameter_size_mb": param_size / (1024 * 1024),
        "buffer_count": buffer_count,
        "buffer_size_mb": buffer_size / (1024 * 1024),
        "total_size_mb": total_size_mb,
    }


def get_optimized_model_config(
    model_type: str,
    base_model_path: str,
    device: str = "cuda",
    optimization_level: str = "default"
) -> Dict[str, Any]:
    """
    Get optimized configuration for inference based on model type and hardware.
    
    Args:
        model_type: Type of model ("sd", "controlnet", etc.)
        base_model_path: Path to the base model
        device: Target device
        optimization_level: Optimization level ("default", "speed", "memory")
        
    Returns:
        Configuration dictionary for optimized inference
    """
    # Device-specific optimizations
    if device == "cuda":
        # CUDA-specific optimizations
        config = {
            "dtype": torch.float16,
            "enable_attention_slicing": True,
            "enable_xformers_memory_efficient_attention": True,
            "enable_model_cpu_offload": False,
            "enable_sequential_cpu_offload": False,
            "enable_vae_slicing": True,
        }
        
        # Check if ROCm (AMD GPU)
        if torch.version.hip is not None:
            # ROCm has issues with xformers
            config["enable_xformers_memory_efficient_attention"] = False
    
    elif device == "mps":
        # Apple Silicon optimizations
        config = {
            "dtype": torch.float16, 
            "enable_attention_slicing": True,
            "enable_xformers_memory_efficient_attention": False,
            "enable_model_cpu_offload": False,
            "enable_sequential_cpu_offload": False,
            "enable_vae_slicing": True,
        }
    
    else:
        # CPU optimizations
        config = {
            "dtype": torch.float32,
            "enable_attention_slicing": True, 
            "enable_xformers_memory_efficient_attention": False,
            "enable_model_cpu_offload": False,
            "enable_sequential_cpu_offload": False,
            "enable_vae_slicing": True,
        }
    
    # Optimization level adjustments
    if optimization_level == "speed":
        config.update({
            "enable_attention_slicing": False,
            "enable_vae_tiling": False,
            "enable_vae_slicing": False,
        })
    
    elif optimization_level == "memory":
        config.update({
            "enable_attention_slicing": True,
            "enable_model_cpu_offload": device == "cuda",
            "enable_vae_tiling": True,
            "enable_vae_slicing": True,
        })
    
    # Model type specific optimizations
    if model_type == "sd":
        pass  # Default SD optimizations
    elif model_type == "controlnet":
        # ControlNet specific optimizations
        pass
    
    return config


def apply_model_optimizations(model: Any, config: Dict[str, Any]) -> Any:
    """
    Apply optimizations to a model based on configuration.
    
    Args:
        model: Model to optimize
        config: Optimization configuration
        
    Returns:
        Optimized model
    """
    # Apply attention slicing
    if config.get("enable_attention_slicing", False):
        if hasattr(model, "enable_attention_slicing"):
            model.enable_attention_slicing()
    
    # Apply xformers memory efficient attention
    if config.get("enable_xformers_memory_efficient_attention", False):
        if hasattr(model, "enable_xformers_memory_efficient_attention"):
            try:
                model.enable_xformers_memory_efficient_attention()
            except ImportError:
                print("xformers not available, skipping memory efficient attention")
    
    # Apply model CPU offload
    if config.get("enable_model_cpu_offload", False):
        if hasattr(model, "enable_model_cpu_offload"):
            model.enable_model_cpu_offload()
    
    # Apply sequential CPU offload
    if config.get("enable_sequential_cpu_offload", False):
        if hasattr(model, "enable_sequential_cpu_offload"):
            model.enable_sequential_cpu_offload()
    
    # Apply VAE slicing
    if config.get("enable_vae_slicing", False):
        if hasattr(model, "enable_vae_slicing"):
            model.enable_vae_slicing()
    
    # Apply VAE tiling
    if config.get("enable_vae_tiling", False):
        if hasattr(model, "enable_vae_tiling"):
            model.enable_vae_tiling()
    
    return model


def get_lora_target_modules(model_type: str) -> List[str]:
    """
    Get target modules for LoRA training based on model type.
    
    Args:
        model_type: Type of model ("sd", "controlnet", etc.)
        
    Returns:
        List of target module names
    """
    if model_type.lower() == "sd" or model_type.lower() == "stable-diffusion":
        return ["q_proj", "k_proj", "v_proj", "out_proj", "to_q", "to_k", "to_v", "to_out.0"]
    elif model_type.lower() == "controlnet":
        return ["q_proj", "k_proj", "v_proj", "out_proj", "to_q", "to_k", "to_v", "to_out.0"]
    else:
        raise ValueError(f"Unsupported model type for LoRA: {model_type}")


def configure_lora_for_training(
    model_type: str,
    rank: int = 4,
    alpha: Optional[int] = None,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
) -> LoraConfig:
    """
    Configure LoRA for training.
    
    Args:
        model_type: Type of model ("sd", "controlnet", etc.)
        rank: LoRA rank
        alpha: LoRA scaling factor (if None, will be set to rank)
        dropout: LoRA dropout
        target_modules: List of target modules (if None, will be determined from model_type)
        
    Returns:
        LoraConfig
    """
    if target_modules is None:
        target_modules = get_lora_target_modules(model_type)
    
    if alpha is None:
        alpha = rank
    
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
    ) 