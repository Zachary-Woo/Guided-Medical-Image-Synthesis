#!/usr/bin/env python
"""
CUDA Test Script for Version 3 Medical Image Synthesis

This script tests CUDA availability and performs basic operations with the
diffusion pipeline and segmentation models to verify everything is working correctly.
"""

import time
import torch
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def test_cuda_availability():
    """Test basic CUDA availability."""
    logger.info("Testing CUDA availability...")
    
    if not torch.cuda.is_available():
        logger.error("❌ CUDA is NOT available")
        logger.info("Please make sure you have a CUDA-capable GPU and proper drivers installed.")
        return False
    
    logger.info("✅ CUDA is available")
    logger.info(f"    - GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"    - CUDA version: {torch.version.cuda}")
    logger.info(f"    - Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Test a small tensor operation
    try:
        logger.info("Running a small tensor operation on GPU...")
        start_time = time.time()
        x = torch.rand(5000, 5000, device="cuda")
        y = torch.matmul(x, x)
        torch.cuda.synchronize()
        del x, y
        torch.cuda.empty_cache()
        logger.info(f"✅ Tensor operation successful ({time.time() - start_time:.2f}s)")
    except Exception as e:
        logger.error(f"❌ Error during tensor operation: {e}")
        return False
    
    return True

def test_diffusers_pipeline():
    """Test loading diffusion model pipeline."""
    try:
        from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
        
        logger.info("Testing diffusion pipeline loading...")
        
        # Test loading ControlNet (small model)
        start_time = time.time()
        try:
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-canny",
                torch_dtype=torch.float16
            )
            logger.info(f"✅ ControlNet loaded successfully ({time.time() - start_time:.2f}s)")
        except Exception as e:
            logger.error(f"❌ Error loading ControlNet: {e}")
            return False
        
        # Test loading base pipeline (don't download full model to save time)
        # Just check if the pipeline class works
        try:
            logger.info("Testing pipeline class creation...")
            pipeline = StableDiffusionControlNetPipeline(
                vae=None,
                text_encoder=None,
                tokenizer=None,
                unet=None,
                scheduler=None,
                safety_checker=None,
                feature_extractor=None,
                controlnet=controlnet
            )
            logger.info("✅ Pipeline class initialized successfully")
        except Exception as e:
            logger.error(f"❌ Error initializing pipeline: {e}")
            return False
        
        return True
    
    except ImportError as e:
        logger.error(f"❌ Missing diffusers dependency: {e}")
        logger.info("Please install diffusers: pip install diffusers transformers")
        return False

def test_sam_availability():
    """Test SAM model integration."""
    try:
        # Try loading transformers SAM
        try:
            from transformers import SamModel, SamProcessor
            
            logger.info("Testing SAM model loading (transformers)...")
            logger.info("(This is a lightweight test - not downloading full model)")
            
            # Don't actually download the model weights to save time
            # Just check if the module is available
            logger.info("✅ Transformers SAM module is available")
            
            # Check SAM2 presence
            sam_models = SamModel.pretrained_model_archive_map if hasattr(SamModel, "pretrained_model_archive_map") else {}
            if "facebook/sam2" in sam_models or "facebook/sam2-l" in sam_models:
                logger.info("✅ SAM2 appears to be available in transformers")
            else:
                logger.info("⚠️ SAM2 models not found in transformers, may need to specify full path or use official repo")
            
            return True
            
        except ImportError:
            logger.warning("⚠️ Transformers SAM not available. Trying official implementation...")
            
            # Try official SAM implementation
            try:
                import segment_anything
                logger.info("✅ Official segment-anything package is available")
                return True
            except ImportError:
                logger.error("❌ SAM not available via official implementation")
                logger.info("Please install segment-anything: pip install git+https://github.com/facebookresearch/segment-anything.git")
                return False
    
    except Exception as e:
        logger.error(f"❌ Unexpected error testing SAM availability: {e}")
        return False

def main():
    """Main test function."""
    logger.info("=" * 50)
    logger.info("CUDA and Model Availability Test")
    logger.info("=" * 50)
    logger.info("")
    
    # Test CUDA
    cuda_ok = test_cuda_availability()
    
    if cuda_ok:
        # Test Diffusers
        logger.info("\n" + "=" * 50)
        logger.info("Testing Diffusion Model Pipeline")
        logger.info("=" * 50)
        diffusion_ok = test_diffusers_pipeline()
        
        # Test SAM
        logger.info("\n" + "=" * 50)
        logger.info("Testing SAM Model Availability")
        logger.info("=" * 50)
        sam_ok = test_sam_availability()
        
        # Overall status
        logger.info("\n" + "=" * 50)
        logger.info("Overall Test Results")
        logger.info("=" * 50)
        logger.info(f"CUDA: {'✅ OK' if cuda_ok else '❌ FAILED'}")
        logger.info(f"Diffusion Pipeline: {'✅ OK' if diffusion_ok else '❌ FAILED'}")
        logger.info(f"SAM Availability: {'✅ OK' if sam_ok else '❌ FAILED'}")
        
        if cuda_ok and diffusion_ok and sam_ok:
            logger.info("\n✅ All tests passed! The system is ready for use.")
        else:
            logger.warning("\n⚠️ Some tests failed. See logs above for details.")
    else:
        logger.error("\n❌ CUDA not available. Other tests skipped.")
    
if __name__ == "__main__":
    main() 