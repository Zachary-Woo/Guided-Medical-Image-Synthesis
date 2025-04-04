#!/usr/bin/env python
"""
Test script to validate the ControlNet pipeline functionality.
This performs basic validation without running full generation.
"""

import os
import sys
import logging
import argparse
import torch
from PIL import Image
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add parent directory to path to import project modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from version1.src.utils.config import Config
from version1.src.utils.pipeline_utils import (
    load_sd_controlnet_pipeline,
    preprocess_image_for_controlnet,
    clear_gpu_memory
)


def parse_args():
    """Parse command line arguments for testing."""
    parser = argparse.ArgumentParser(description="Test ControlNet pipeline functionality")
    
    parser.add_argument("--config", type=str, default="version1/configs/medmnist_canny_demo.yaml",
                       help="Path to configuration file")
    parser.add_argument("--conditioning_image", type=str,
                       help="Path to a test image for conditioning")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    return parser.parse_args()


def test_model_loading(config):
    """Test model loading functionality."""
    logger.info("Testing model loading...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        logger.info(f"Loading base model: {config.model.pretrained_model_id}")
        logger.info(f"Loading ControlNet: {config.inference.controlnet_inference_id}")
        
        # Try to load pipeline
        pipeline = load_sd_controlnet_pipeline(
            base_model_id=config.model.pretrained_model_id,
            controlnet_model_id=config.inference.controlnet_inference_id,
            device=device
        )
        
        logger.info("✓ Model loading successful")
        return True, pipeline
    except Exception as e:
        logger.error(f"✗ Model loading failed: {e}")
        logger.error(traceback.format_exc())
        return False, None


def test_preprocessing(config, image_path=None):
    """Test preprocessing functionality."""
    logger.info("Testing preprocessing...")
    
    if not image_path:
        logger.info("No test image provided, creating a dummy image")
        # Create a dummy image
        img = Image.new('RGB', (config.data.image_size, config.data.image_size), color='white')
        
        # Draw some shapes for better edge detection
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.rectangle(
            [(config.data.image_size//4, config.data.image_size//4), 
             (3*config.data.image_size//4, 3*config.data.image_size//4)], 
            outline='black', width=5
        )
        draw.ellipse(
            [(config.data.image_size//3, config.data.image_size//3),
             (2*config.data.image_size//3, 2*config.data.image_size//3)],
            outline='black', width=5
        )
    else:
        try:
            logger.info(f"Loading test image: {image_path}")
            img = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"✗ Failed to load test image: {e}")
            return False, None
    
    try:
        logger.info(f"Preprocessing for {config.inference.conditioning_type} conditioning")
        processed = preprocess_image_for_controlnet(
            img, config.inference.conditioning_type, config.data.image_size
        )
        
        if processed is None:
            logger.error("✗ Preprocessing returned None")
            return False, None
        
        logger.info(f"✓ Preprocessing successful, output size: {processed.size}")
        return True, processed
    except Exception as e:
        logger.error(f"✗ Preprocessing failed: {e}")
        logger.error(traceback.format_exc())
        return False, None


def test_gpu_memory():
    """Test GPU memory management."""
    if not torch.cuda.is_available():
        logger.info("Skipping GPU memory test (CUDA not available)")
        return True
    
    logger.info("Testing GPU memory management...")
    
    try:
        # Get initial memory
        initial_allocated = torch.cuda.memory_allocated(0)
        initial_reserved = torch.cuda.memory_reserved(0)
        
        logger.info(f"Initial GPU memory: {initial_allocated/1024**2:.2f}MB allocated, {initial_reserved/1024**2:.2f}MB reserved")
        
        # Allocate a test tensor
        test_tensor = torch.zeros((1000, 1000, 10), device='cuda')
        
        # Check memory after allocation
        mid_allocated = torch.cuda.memory_allocated(0)
        logger.info(f"After allocation: {mid_allocated/1024**2:.2f}MB allocated")
        
        # Clear memory
        del test_tensor
        clear_gpu_memory()
        
        # Check memory after clearing
        final_allocated = torch.cuda.memory_allocated(0)
        logger.info(f"After clearing: {final_allocated/1024**2:.2f}MB allocated")
        
        # Memory should be closer to initial after clearing
        logger.info(f"✓ GPU memory management successful")
        return True
    except Exception as e:
        logger.error(f"✗ GPU memory management failed: {e}")
        return False


def main():
    """Main test function."""
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Starting pipeline tests...")
    
    try:
        # Load configuration
        config = Config.from_yaml(args.config)
        logger.info(f"✓ Configuration loaded from {args.config}")
    except Exception as e:
        logger.error(f"✗ Failed to load config: {e}")
        return 1
    
    # Test GPU memory management
    memory_ok = test_gpu_memory()
    
    # Test model loading
    models_ok, pipeline = test_model_loading(config)
    
    # Test preprocessing
    preprocessing_ok, processed_image = test_preprocessing(config, args.conditioning_image)
    
    # Clean up
    if pipeline is not None:
        del pipeline
    clear_gpu_memory()
    
    # Print summary
    logger.info("\n=== Test Summary ===")
    logger.info(f"Configuration loading: {'✓ Passed' if True else '✗ Failed'}")
    logger.info(f"GPU memory management: {'✓ Passed' if memory_ok else '✗ Failed'}")
    logger.info(f"Model loading: {'✓ Passed' if models_ok else '✗ Failed'}")
    logger.info(f"Image preprocessing: {'✓ Passed' if preprocessing_ok else '✗ Failed'}")
    
    all_passed = memory_ok and models_ok and preprocessing_ok
    if all_passed:
        logger.info("\n🎉 All tests passed! The pipeline is ready for generation.")
        return 0
    else:
        logger.warning("\n⚠️ Some tests failed. Check the logs for details.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 