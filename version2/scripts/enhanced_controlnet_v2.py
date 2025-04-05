#!/usr/bin/env python
"""
Enhanced ControlNet v2 - Histopathology Image Generation
With stain normalization and LoRA support
"""

import os
import sys
import torch
import argparse
import logging
from PIL import Image
import traceback
from pathlib import Path
import numpy as np
import cv2
import json
import time
import matplotlib.pyplot as plt

# Check CUDA availability and installation status
def check_cuda_installation():
    """Check CUDA installation and provide helpful error messages"""
    cuda_available = torch.cuda.is_available()
    
    if not cuda_available:
        print("\n" + "="*80)
        print("WARNING: CUDA NOT DETECTED - GPU ACCELERATION UNAVAILABLE")
        print("="*80)
        print("Your system does not have proper CUDA setup for PyTorch.")
        print("This will result in significantly slower generation speeds.")
        print("\nPossible solutions:")
        print("1. Install CUDA-enabled version of PyTorch:")
        print("   pip uninstall torch torchvision torchaudio")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("2. Install GPU-enabled bitsandbytes:")
        print("   pip uninstall bitsandbytes")
        print("   pip install bitsandbytes-windows")
        print("3. Install proper CUDA version matching your PyTorch")
        print("   Download from: https://developer.nvidia.com/cuda-downloads")
        print("="*80 + "\n")
        return False
    
    print("\n" + "="*80)
    print(f"CUDA AVAILABLE: {torch.cuda.get_device_name(0)}")
    cuda_version = torch.version.cuda if hasattr(torch.version, 'cuda') else "Unknown"
    print(f"CUDA Version: {cuda_version}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Check for GPU-related packages
    # Wrap bitsandbytes import to catch potential runtime errors during its setup
    try:
        import bitsandbytes
        print("bitsandbytes: Installed and imported successfully")
    except ImportError:
        print("bitsandbytes: Not installed (optional performance library)")
        print("  Install with: pip install bitsandbytes-windows (or appropriate version)")
    except Exception as e:
        # Catch other errors (like the RuntimeError from CUDA setup failure)
        print(f"bitsandbytes: Failed to import or initialize ({type(e).__name__}). Will proceed without it.")
        print(f"  Error details: {e}")
        print("  This is often due to CUDA version mismatch or missing dependencies for bitsandbytes.")
        print("  Check bitsandbytes documentation for compatibility with your CUDA/PyTorch version.")
    
    try:
        import xformers
        print("xformers: Installed")
    except ImportError:
        print("xformers: Not installed (recommended for memory efficiency)")
        print("  Install with: pip install xformers")
    
    print("="*80 + "\n")
    return True

# Run CUDA check at import time
CUDA_AVAILABLE = check_cuda_installation()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

# Import diffusers with better error handling
try:
    from diffusers import (
        ControlNetModel, 
        StableDiffusionControlNetPipeline, 
        UniPCMultistepScheduler,
        DPMSolverMultistepScheduler,
        DDIMScheduler,
        EulerDiscreteScheduler
    )
    from diffusers.utils import load_image
    
    # Try to import MultiControlNetModel, but don't fail if it's not available
    try:
        MultiControlNetModel = None
        # Try different import paths for MultiControlNetModel
        try:
            from diffusers import MultiControlNetModel
        except (ImportError, ModuleNotFoundError):
            try:
                from diffusers.models import MultiControlNetModel
            except (ImportError, ModuleNotFoundError):
                try:
                    from diffusers.pipelines.stable_diffusion import MultiControlNetModel
                except (ImportError, ModuleNotFoundError):
                    logger.warning("MultiControlNetModel not available in your diffusers version. Multi-ControlNet features will be disabled.")
    except Exception as e:
        logger.warning(f"Error importing MultiControlNetModel: {e}")
        logger.warning("Multi-ControlNet features will be disabled.")
        MultiControlNetModel = None
except ImportError as e:
    print(f"Error importing diffusers: {e}")
    print("Please install diffusers with: pip install diffusers>=0.19.0 transformers")
    sys.exit(1)

# Add parent directory to path for importing project modules
parent_dir = str(Path(__file__).resolve().parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import project utilities
try:
    from version2.utils.stain_normalization import (
        normalize_histopathology_image,
        MacenkoNormalizer,
        ReinhardNormalizer,
        visualize_normalization
    )
    from version2.utils.output_helpers import get_next_output_dir, setup_output_logging
except ImportError:
    print("Project utilities not found. Please run from the project root directory.")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Enhanced Medical Image Generation with ControlNet and LoRA")
    
    # Input/Output arguments
    parser.add_argument("--condition_image", type=str, required=True,
                        help="Path to conditioning image")
    parser.add_argument("--reference_image", type=str, default=None,
                        help="Path to staining reference image (if not provided, default H&E colors will be used)")
    parser.add_argument("--output_dir", type=str, default="output/enhanced_controlnet_v2",
                        help="Base output directory (will be auto-incremented)")
    
    # ControlNet arguments
    parser.add_argument("--controlnet_model", type=str, default="lllyasviel/sd-controlnet-canny",
                        help="ControlNet model ID or path")
    parser.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-v1-5",
                        help="Base model ID or path")
    parser.add_argument("--lora_model", type=str, default=None,
                        help="Optional path to LoRA adapter weights (.safetensors or .bin)")
    parser.add_argument("--lora_scale", type=float, default=0.7,
                        help="Scale factor for LoRA adapter")
    
    # Conditioning arguments
    parser.add_argument("--prompt", type=str, 
                        default="High-resolution histopathology slide showing colorectal tissue with clear cellular detail, H&E stain, medical scan",
                        help="Text prompt for generation")
    parser.add_argument("--negative_prompt", type=str,
                        default="blurry, low quality, low resolution, deformed, distorted, watermark, text, bad anatomy, poor staining",
                        help="Negative prompt to guide generation")
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=1.0,
                        help="ControlNet conditioning scale")
    parser.add_argument("--low_threshold", type=int, default=30,
                        help="Canny edge detection low threshold")
    parser.add_argument("--high_threshold", type=int, default=120,
                        help="Canny edge detection high threshold")
    
    # Generation arguments
    parser.add_argument("--steps", type=int, default=50, 
                        help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=9.0,
                        help="Guidance scale")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    parser.add_argument("--num_images", type=int, default=1,
                        help="Number of images to generate")
    parser.add_argument("--scheduler", type=str, default="unipc", 
                        choices=["unipc", "dpm", "ddim", "euler"],
                        help="Scheduler to use for diffusion")
    
    # Image processing arguments
    parser.add_argument("--input_size", type=int, default=64,
                        help="Original image size")
    parser.add_argument("--output_size", type=int, default=512,
                        help="Output image size")
    parser.add_argument("--stain_norm", type=str, default="macenko", 
                        choices=["macenko", "reinhard", "none"],
                        help="Stain normalization method")
    
    # Misc arguments
    parser.add_argument("--debug", action="store_true", 
                        help="Enable debug logging")
    parser.add_argument("--save_intermediates", action="store_true",
                        help="Save intermediate processed images")
    parser.add_argument("--use_multi_controlnet", action="store_true",
                        help="Use multiple ControlNet models (requires reference image)")
    parser.add_argument("--no_metadata", action="store_true",
                        help="Skip saving metadata file")
    
    return parser.parse_args()

def get_next_output_dir(base_dir):
    """
    Create a sequentially numbered output directory.
    
    Args:
        base_dir (str or Path): Base output directory path
        
    Returns:
        Path: Next available numbered directory
    """
    base_path = Path(base_dir)
    base_parent = base_path.parent
    base_name = base_path.name
    
    # Find all existing numbered directories
    existing_dirs = []
    for item in base_parent.glob(f"{base_name}_*"):
        if item.is_dir():
            try:
                # Extract the number after the underscore
                num = int(item.name.split('_')[-1])
                existing_dirs.append(num)
            except ValueError:
                # Skip directories that don't end with a number
                continue
    
    # Determine the next number
    next_num = 1
    if existing_dirs:
        next_num = max(existing_dirs) + 1
    
    # Create the new directory path
    next_dir = base_parent / f"{base_name}_{next_num}"
    return next_dir

def create_enhanced_edge_map(image_path, low_threshold=30, high_threshold=120, target_size=512, use_stain_norm=True, reference_image=None, stain_method="macenko", save_intermediates=False, save_dir=None):
    """
    Process a medical image to create an enhanced edge map for ControlNet conditioning.
    
    Args:
        image_path: Path to the image
        low_threshold: Lower threshold for Canny edge detection
        high_threshold: Higher threshold for Canny edge detection
        target_size: Target size for output images
        use_stain_norm: Whether to apply stain normalization
        reference_image: Reference image for stain normalization
        stain_method: Stain normalization method ('macenko' or 'reinhard')
        save_intermediates: Whether to save intermediate processed images
        save_dir: Directory to save intermediate images
        
    Returns:
        tuple of (edge_image, resized_original, normalized_image)
    """
    logger.info(f"Processing image: {image_path}")
    
    # Load the original image
    original_image = Image.open(image_path).convert("RGB")
    original_np = np.array(original_image)
    
    # Resize to target size with proper interpolation
    if original_np.shape[0] < target_size or original_np.shape[1] < target_size:
        # First upscale with better algorithm for small images
        resized_np = cv2.resize(original_np, (target_size, target_size), 
                               interpolation=cv2.INTER_LANCZOS4)
    else:
        resized_np = cv2.resize(original_np, (target_size, target_size))
    
    # Apply stain normalization if requested
    if use_stain_norm and stain_method.lower() != "none":
        logger.info(f"Applying {stain_method} stain normalization")
        try:
            # Create a direct visualization of the stain normalization effect
            if save_intermediates and save_dir:
                try:
                    # Create a synthetic reference if none is provided
                    ref_img = reference_image
                    if ref_img is None:
                        # Create a synthetic H&E reference
                        ref_img = np.zeros((100, 100, 3), dtype=np.uint8)
                        ref_img[:50, :, 0] = 150  # Hematoxylin - purple/blue
                        ref_img[:50, :, 1] = 50
                        ref_img[:50, :, 2] = 150
                        ref_img[50:, :, 0] = 200  # Eosin - pink
                        ref_img[50:, :, 1] = 100
                        ref_img[50:, :, 2] = 100
                    
                    # First apply normalization to get the normalized image
                    norm_img = normalize_histopathology_image(
                        resized_np,
                        reference_image=ref_img,
                        method=stain_method
                    )
                    
                    # Then use the visualization function
                    vis_fig = visualize_normalization(
                        resized_np,  # original
                        norm_img,    # normalized
                        target=ref_img  # reference target
                    )
                    
                    # Save the visualization
                    vis_path = save_dir / "stain_normalization_comparison.png"
                    vis_fig.savefig(vis_path)
                    plt.close(vis_fig)
                    logger.info(f"Saved stain normalization comparison to {vis_path}")
                except Exception as e:
                    logger.warning(f"Failed to create normalization visualization: {e}")
            
            # Initialize the appropriate normalizer
            if stain_method.lower() == "macenko":
                normalizer = MacenkoNormalizer()
            elif stain_method.lower() == "reinhard":
                normalizer = ReinhardNormalizer()
            else:
                # Fall back to function-based normalization for other methods
                normalized_np = normalize_histopathology_image(
                    resized_np, 
                    reference_image=reference_image,
                    method=stain_method
                )
                
            # Use the normalizer directly if initialized
            if stain_method.lower() in ["macenko", "reinhard"]:
                # If no reference image is provided, create a synthetic one
                if reference_image is None:
                    logger.info("Creating synthetic reference image for stain normalization")
                    # Create a larger synthetic reference image 
                    # with colors better suited for H&E staining
                    synthetic_reference = np.zeros((target_size, target_size, 3), dtype=np.uint8)
                    
                    # First half - Hematoxylin-like (purple/blue)
                    synthetic_reference[:target_size//2, :, 0] = 145  # R - more blue
                    synthetic_reference[:target_size//2, :, 1] = 60   # G - less green
                    synthetic_reference[:target_size//2, :, 2] = 170  # B - more blue
                    
                    # Second half - Eosin-like (pink)
                    synthetic_reference[target_size//2:, :, 0] = 210  # R - more red
                    synthetic_reference[target_size//2:, :, 1] = 120  # G - medium green
                    synthetic_reference[target_size//2:, :, 2] = 130  # B - medium blue
                    
                    # Add some texture/variation to make it more realistic
                    noise = np.random.randint(-20, 20, synthetic_reference.shape, dtype=np.int16)
                    synthetic_reference = np.clip(synthetic_reference.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                    
                    # Save synthetic reference if intermediates are requested
                    if save_intermediates and save_dir:
                        ref_path = save_dir / "synthetic_reference.png"
                        Image.fromarray(synthetic_reference).save(ref_path)
                        logger.info(f"Saved synthetic reference to {ref_path}")
                    
                    # Fit the normalizer to the synthetic reference
                    try:
                        normalizer.fit(synthetic_reference)
                    except Exception as e:
                        logger.error(f"Failed to fit normalizer on synthetic reference: {e}")
                        raise
                else:
                    # Fit the normalizer to the reference image
                    normalizer.fit(reference_image)
                
                # Transform the target image
                try:
                    normalized_np = normalizer.transform(resized_np)
                except Exception as e:
                    logger.error(f"Failed to transform image with normalizer: {e}")
                    normalized_np = resized_np  # Fallback to unnormalized image
            
            # Save intermediate normalized image if requested
            if save_intermediates and save_dir:
                normalized_img = Image.fromarray(normalized_np)
                normalized_path = save_dir / "normalized_image.png"
                normalized_img.save(normalized_path)
                logger.info(f"Saved normalized image to {normalized_path}")
        except Exception as e:
            logger.error(f"Stain normalization failed: {e}")
            logger.warning("Proceeding with unnormalized image")
            normalized_np = resized_np
    else:
        normalized_np = resized_np
    
    # Convert to grayscale for edge detection
    if len(normalized_np.shape) == 3 and normalized_np.shape[2] == 3:
        gray_np = cv2.cvtColor(normalized_np, cv2.COLOR_RGB2GRAY)
    else:
        gray_np = normalized_np
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # This dramatically improves feature extraction in medical images
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray_np)
    
    # Apply multiple preprocessing techniques and combine results
    # 1. Standard Canny with moderate thresholds
    canny_standard = cv2.Canny(equalized, low_threshold, high_threshold)
    
    # 2. Gaussian blur followed by Canny with lower thresholds to catch more edges
    blurred = cv2.GaussianBlur(equalized, (3, 3), 0)
    canny_blurred = cv2.Canny(blurred, max(5, low_threshold//2), max(50, high_threshold//2))
    
    # 3. Sobel edge detection (captures more gradual transitions)
    sobelx = cv2.Sobel(equalized, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(equalized, cv2.CV_64F, 0, 1, ksize=3)
    sobel_edges = np.sqrt(sobelx**2 + sobely**2)
    sobel_edges = (sobel_edges > np.mean(sobel_edges) + 0.5 * np.std(sobel_edges)) * 255
    sobel_edges = sobel_edges.astype(np.uint8)
    
    # Combine edge maps (take maximum at each pixel)
    combined_edges = np.maximum(canny_standard, canny_blurred)
    combined_edges = np.maximum(combined_edges, sobel_edges)
    
    # Optional: Dilate edges slightly to enhance connectivity
    kernel = np.ones((2, 2), np.uint8)
    combined_edges = cv2.dilate(combined_edges, kernel, iterations=1)
    
    # Save intermediate edge maps if requested
    if save_intermediates and save_dir:
        # Save individual edge maps
        Image.fromarray(canny_standard).save(save_dir / "canny_standard.png")
        Image.fromarray(canny_blurred).save(save_dir / "canny_blurred.png")
        Image.fromarray(sobel_edges).save(save_dir / "sobel_edges.png")
        logger.info(f"Saved intermediate edge maps to {save_dir}")
    
    # Convert to RGB (required by ControlNet)
    canny_edges_rgb = np.stack([combined_edges] * 3, axis=2)
    
    # Create both the canny edge image and the resized original
    canny_image = Image.fromarray(canny_edges_rgb.astype(np.uint8))
    resized_image = Image.fromarray(resized_np.astype(np.uint8))
    normalized_image = Image.fromarray(normalized_np.astype(np.uint8))
    
    return canny_image, resized_image, normalized_image

def setup_multicontrolnet_pipeline(args):
    """Set up a pipeline with multiple ControlNet models"""
    # Force CUDA if available, regardless of detection
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = "cpu"
        logger.warning("CUDA not available. Using CPU - this will be slow!")
    
    # Set up parameters
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    
    # Load canny ControlNet model
    canny_controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        torch_dtype=torch_dtype
    )
    
    # If reference image is provided, load a second ControlNet
    if hasattr(args, 'use_multi_controlnet') and args.use_multi_controlnet:
        # Load depth ControlNet as secondary control
        depth_controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-depth",
            torch_dtype=torch_dtype
        )
        # Combine into MultiControlNet
        controlnet = MultiControlNetModel([canny_controlnet, depth_controlnet])
        logger.info("Using MultiControlNet with Canny and Depth models")
    else:
        # Use only one ControlNet model
        controlnet = canny_controlnet
        logger.info("Using single ControlNet (Canny)")
    
    # Load pipeline with ControlNet
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        args.base_model,
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=torch_dtype
    )
    
    # Load LoRA weights if specified
    if args.lora_model:
        lora_file_path = args.lora_model
        logger.info(f"Loading LoRA adapter from: {lora_file_path}")
        if not os.path.exists(lora_file_path):
            logger.error(f"LoRA file not found: {lora_file_path}")
            logger.warning("Proceeding without LoRA weights")
        else:
            logger.info(f"LoRA file exists: {os.path.exists(lora_file_path)}, size: {os.path.getsize(lora_file_path)} bytes")
            try:
                # Use the standard diffusers method for loading local LoRA weights
                # This handles both .safetensors and other formats if PEFT is installed
                # It expects the file path directly.
                pipeline.load_lora_weights(
                    pretrained_model_name_or_path_or_dict=os.path.dirname(lora_file_path), 
                    weight_name=os.path.basename(lora_file_path), 
                    # adapter_name="default" # Optionally provide an adapter name
                )
                
                # Fuse/apply LoRA weights with scale
                # fuse_lora might be deprecated or behaviour changed; unet.set_adapters might be preferred
                try:
                     pipeline.fuse_lora(lora_scale=args.lora_scale)
                     logger.info(f"LoRA adapter '{os.path.basename(lora_file_path)}' fused with scale {args.lora_scale}")
                except AttributeError:
                     # Newer diffusers might use set_adapters for scaling
                     # pipeline.unet.set_adapters(["default"], adapter_weights=[args.lora_scale])
                     # pipeline.set_adapters(["default"], adapter_weights={"default": args.lora_scale}) # Check correct syntax
                     logger.warning("pipeline.fuse_lora() not available. LoRA scale might not be applied directly during fusion.")
                     logger.info("You might need to adjust scaling during inference if using set_adapters.")
                     # For simplicity, we'll rely on the loading itself and log a warning.
                     # If scaling issues arise, revisit diffusers docs for the specific version.
                except Exception as fuse_error:
                    logger.error(f"Error during LoRA fusion: {fuse_error}")
                    logger.warning("Proceeding with potentially unscaled LoRA weights.")

            except ImportError as import_err:
                 logger.error(f"Import error during LoRA loading (possibly missing 'peft'): {import_err}")
                 logger.warning("Proceeding without LoRA weights. Install PEFT (`pip install peft`) for full LoRA support.")
            except Exception as e:
                logger.error(f"Failed to load LoRA weights from {lora_file_path}: {e}")
                logger.error(traceback.format_exc())
                logger.warning("Proceeding without LoRA weights")
    
    # Set up scheduler
    if args.scheduler == "unipc":
        pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
        logger.info("Using UniPC scheduler")
    elif args.scheduler == "dpm":
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        logger.info("Using DPM-Solver++ scheduler")
    elif args.scheduler == "ddim":
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        logger.info("Using DDIM scheduler")
    elif args.scheduler == "euler":
        pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
        logger.info("Using Euler Discrete scheduler")
    
    # Skip CPU offload if we're using CUDA
    if device == "cuda":
        pipeline = pipeline.to(device)
        
        try:
            pipeline.enable_attention_slicing()
            logger.info("Enabled attention slicing")
        except Exception as e:
            logger.warning(f"Could not enable attention slicing: {e}")
        
        try:
            # Attempt to enable xformers, but handle the case where it's not available
            # or fails due to incompatibility
            pipeline.enable_xformers_memory_efficient_attention()
            logger.info("Enabled xFormers memory efficient attention")
        except Exception as e: # Catch a broader range of exceptions
            logger.warning(f"Could not enable xFormers: {type(e).__name__} - {e}")
            logger.info("Continuing without xFormers optimization. This is often due to version incompatibility.")
            logger.info("Check xformers documentation for compatibility with your PyTorch/CUDA version or reinstall if necessary.")
            # Example: pip uninstall xformers && pip install xformers
    else:
        # Only if CUDA not available
        pipeline.enable_model_cpu_offload()
        logger.warning("Using CPU offload - generation will be very slow")
    
    logger.info("Pipeline set up complete")
    return pipeline

def enhance_medical_prompt(base_prompt, image_type="histopathology"):
    """
    Enhance a basic prompt with medical imaging specific details.
    
    Args:
        base_prompt: Base text prompt
        image_type: Type of medical image
        
    Returns:
        Enhanced prompt
    """
    # Check if the prompt already has detailed medical terms
    medical_terms = ["histology", "pathology", "H&E", "stain", "microscopy", "histopathology", 
                   "medical scan", "biopsy", "specimen"]
    
    has_medical_terms = any(term.lower() in base_prompt.lower() for term in medical_terms)
    
    # Define specialized enhancements based on image type
    enhancements = {
        "histopathology": ", H&E stain, high-resolution microscopy, cellular detail, professional pathology scan, tissue section, medical grade",
        "xray": ", high-contrast radiograph, professional medical imaging, clear bone detail, radiological scan, diagnostic quality",
        "ct": ", computed tomography, axial slice, clear tissue differentiation, professional medical scan, diagnostic quality",
        "mri": ", magnetic resonance imaging, T1-weighted, detailed anatomical scan, high tissue contrast, medical grade",
        "ultrasound": ", sonogram, professional ultrasound imaging, clear tissue boundaries, diagnostic quality"
    }
    
    # Only enhance if the prompt doesn't already have specific medical terms
    if not has_medical_terms:
        enhancement = enhancements.get(image_type.lower(), ", medical quality, professional scan, high detail")
        return f"{base_prompt}{enhancement}"
    
    return base_prompt

def main():
    args = parse_args()
    start_time = time.time()
    
    # Set up logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create output directory using new helper
    output_dir = get_next_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    log_file = output_dir / "generate.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(file_handler)
    
    logger.info(f"Using output directory: {output_dir}")
    logger.info(f"Logging to {log_file}")
    
    # Create subdirectory for intermediate files if needed
    intermediates_dir = None
    if args.save_intermediates:
        intermediates_dir = output_dir / "intermediates"
        intermediates_dir.mkdir(exist_ok=True)
    
    # Enhance the prompt with medical specific details
    prompt = enhance_medical_prompt(args.prompt, "histopathology")
    logger.info(f"Using enhanced prompt: {prompt}")
    
    # Load reference image for stain normalization if provided
    reference_image = None
    if args.reference_image:
        try:
            logger.info(f"Loading stain reference image: {args.reference_image}")
            reference_image = np.array(Image.open(args.reference_image).convert('RGB'))
            
            # Enable multi-controlnet if reference image is provided and not explicitly disabled
            if not hasattr(args, 'use_multi_controlnet'):
                args.use_multi_controlnet = True
                logger.info("Automatically enabling multi-controlnet with reference image")
        except Exception as e:
            logger.error(f"Failed to load reference image: {e}")
            logger.warning("Proceeding without stain normalization reference")
            args.use_multi_controlnet = False
    else:
        if args.use_multi_controlnet:
            logger.warning("Multi-controlnet requested but no reference image provided. Disabling multi-controlnet.")
            args.use_multi_controlnet = False
    
    try:
        # Process conditioning image with stain normalization
        use_stain_norm = args.stain_norm.lower() != "none"
        canny_image, resized_image, normalized_image = create_enhanced_edge_map(
            args.condition_image, 
            low_threshold=args.low_threshold,
            high_threshold=args.high_threshold,
            target_size=args.output_size,
            use_stain_norm=use_stain_norm,
            reference_image=reference_image,
            stain_method=args.stain_norm,
            save_intermediates=args.save_intermediates,
            save_dir=intermediates_dir
        )
        
        # Save the processed images for reference
        canny_path = output_dir / "canny_edge.png"
        resized_path = output_dir / "original_input.png"
        normalized_path = output_dir / "normalized_input.png"
        
        canny_image.save(canny_path)
        resized_image.save(resized_path)
        normalized_image.save(normalized_path)
        
        logger.info(f"Saved canny edge image to {canny_path}")
        logger.info(f"Saved original input image to {resized_path}")
        if use_stain_norm:
            logger.info(f"Saved normalized input image to {normalized_path}")
        
        # Set up pipeline
        pipeline = setup_multicontrolnet_pipeline(args)
        
        # Generate images
        logger.info(f"Generating {args.num_images} image(s) with {args.steps} steps, guidance scale {args.guidance_scale}")
        
        # Set up generator for reproducibility
        if args.seed is not None:
            generator = torch.manual_seed(args.seed)
            seeds = [args.seed + i for i in range(args.num_images)]
        else:
            generator = None
            # Create random seeds for each image
            seeds = [int(torch.randint(0, 2147483647, (1,)).item()) for _ in range(args.num_images)]
        
        # Use optimized parameters for medical images
        controlnet_scale = args.controlnet_conditioning_scale
        
        all_images = []
        for i in range(args.num_images):
            if generator is None:
                # Use a different seed for each image if no seed was specified
                generator = torch.manual_seed(seeds[i])
            
            logger.info(f"Generating image {i+1}/{args.num_images} with seed {seeds[i]}")
            
            # Only use control images that are not None
            control_images = [canny_image]
            
            # Set up ControlNet conditioning scale
            if args.use_multi_controlnet and reference_image is not None:
                # Convert numpy array to PIL Image for the pipeline
                reference_pil = Image.fromarray(reference_image)
                control_images.append(reference_pil)
                # Use a list of scales for multi-controlnet
                controlnet_scales = [controlnet_scale, controlnet_scale * 0.5]  # Lower scale for second control
            else:
                # Single scale for single controlnet
                controlnet_scales = controlnet_scale
                
            image = pipeline(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                image=control_images,
                num_inference_steps=args.steps,
                generator=generator,
                guidance_scale=args.guidance_scale,
                controlnet_conditioning_scale=controlnet_scales
            ).images[0]
            
            all_images.append(image)
            
            # Save generated image
            if args.num_images > 1:
                output_path = output_dir / f"generated_{i+1}.png"
            else:
                output_path = output_dir / "generated.png"
                
            image.save(output_path)
            logger.info(f"Image saved to {output_path}")
        
        # Save metadata
        metadata = {
            "prompt": prompt,
            "negative_prompt": args.negative_prompt,
            "conditioning_image": str(args.condition_image),
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "controlnet_conditioning_scale": controlnet_scale,
            "seeds": seeds,
            "base_model": args.base_model,
            "controlnet_model": args.controlnet_model,
            "lora_model": args.lora_model,
            "lora_scale": args.lora_scale if args.lora_model else None,
            "scheduler": args.scheduler,
            "low_threshold": args.low_threshold,
            "high_threshold": args.high_threshold,
            "output_size": args.output_size,
            "stain_normalization": args.stain_norm if use_stain_norm else "none",
            "reference_image": str(args.reference_image) if args.reference_image else None,
            "generation_time_seconds": round(time.time() - start_time, 2)
        }
        
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Generation complete in {metadata['generation_time_seconds']} seconds")
        return 0
    
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 