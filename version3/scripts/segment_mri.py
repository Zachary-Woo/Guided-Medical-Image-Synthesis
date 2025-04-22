#!/usr/bin/env python
"""
SAM2 Integration for Brain MRI Segmentation

This script provides a wrapper around Meta's Segment Anything 2 (SAM2) model
for segmenting brain MRI images. It supports text prompts, point prompts, and box prompts.
"""

import os
import time
import logging
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Segment brain MRI images using SAM2"
    )
    parser.add_argument("--input_image", type=str, required=True,
                      help="Input MRI image path")
    parser.add_argument("--output_dir", type=str, default="output/version3/segments",
                      help="Output directory for segmentation results")
    parser.add_argument("--sam_model", type=str, default="facebook/sam2",
                      help="SAM2 model identifier")
    parser.add_argument("--prompt_type", type=str, default="text",
                      choices=["text", "point", "box"],
                      help="Type of prompt to provide to SAM2")
    parser.add_argument("--prompt", type=str,
                      help="Text prompt for text-guided segmentation")
    parser.add_argument("--point", type=str,
                      help="Point prompt as 'x,y' for point-guided segmentation")
    parser.add_argument("--box", type=str,
                      help="Box prompt as 'x1,y1,x2,y2' for box-guided segmentation")
    
    return parser.parse_args()

def create_output_dir(base_dir):
    """Create a uniquely timestamped output directory."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = Path(base_dir) / f"segment_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def load_sam2_model(model_id=None, sam_checkpoint=None, device="cuda"):
    """
    Load SAM2 model for inference
    
    Tries to load the model in this order:
    1. Load SAM2 from GitHub repository (segment-anything-2)
    2. Load original SAM from GitHub repository (segment-anything)
    3. Load SAM/SAM2 from HuggingFace transformers
    
    Args:
        model_id: Model ID on HuggingFace (used as fallback)
        sam_checkpoint: Path to SAM checkpoint or model type for GitHub repositories
        device: Device to load model on
    
    Returns:
        dict: Dictionary containing model, processor and other model details
    """
    supports_text = False
    supports_boxes = True
    model_type = "unknown"
    
    # First try to load SAM2 from GitHub repository
    try:
        import segment_anything_2
        from segment_anything_2.build_sam2 import sam2_model_registry
        from segment_anything_2.predictor import SamPredictor
        
        logging.info("Loading SAM2 from GitHub repository")
        
        # Use checkpoint to determine model type if provided, otherwise default to "vit_h"
        model_type = sam_checkpoint if sam_checkpoint else "vit_h"
        model = sam2_model_registry[model_type](checkpoint=sam_checkpoint, device=device)
        predictor = SamPredictor(model)
        
        return {
            "model": model,
            "predictor": predictor,
            "processor": None,  # No processor for GitHub implementation
            "supports_text": supports_text,
            "supports_boxes": supports_boxes,
            "model_type": "sam2_github"
        }
    except (ImportError, KeyError, FileNotFoundError) as e:
        logging.warning(f"Failed to load SAM2 from GitHub repository: {e}")
    
    # Next try to load original SAM from GitHub repository
    try:
        import segment_anything
        from segment_anything import sam_model_registry, SamPredictor
        
        logging.info("Loading SAM from GitHub repository")
        
        # Use checkpoint to determine model type if provided, otherwise default to "vit_h"
        model_type = sam_checkpoint if sam_checkpoint else "vit_h"
        model = sam_model_registry[model_type](checkpoint=sam_checkpoint, device=device)
        predictor = SamPredictor(model)
        
        return {
            "model": model,
            "predictor": predictor,
            "processor": None,  # No processor for GitHub implementation
            "supports_text": supports_text,
            "supports_boxes": supports_boxes,
            "model_type": "sam_github"
        }
    except (ImportError, KeyError, FileNotFoundError) as e:
        logging.warning(f"Failed to load SAM from GitHub repository: {e}")
    
    # Fall back to HuggingFace transformers
    if model_id is None:
        model_id = "facebook/sam2-l"
        logging.info(f"No model_id provided, using default: {model_id}")
    
    logging.info(f"Loading SAM/SAM2 from HuggingFace: {model_id}")
    
    # Check if model_id is for SAM or SAM2
    if "sam2" in model_id:
        from transformers import Sam2Model, Sam2Processor
        model = Sam2Model.from_pretrained(model_id).to(device)
        processor = Sam2Processor.from_pretrained(model_id)
        supports_text = True  # SAM2 supports text prompts
        model_type = "sam2_hf"
    else:
        from transformers import SamModel, SamProcessor
        model = SamModel.from_pretrained(model_id).to(device)
        processor = SamProcessor.from_pretrained(model_id)
        model_type = "sam_hf"
    
    return {
        "model": model,
        "processor": processor,
        "predictor": None,  # No predictor for HuggingFace implementation
        "supports_text": supports_text,
        "supports_boxes": supports_boxes,
        "model_type": model_type
    }

def load_image(image_path):
    """Load and prepare the image for segmentation."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load image with PIL
    image = Image.open(image_path)
    
    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Resize if too large to preserve memory
    max_size = 1024
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.LANCZOS)
        logger.info(f"Resized image to {new_size}")
    
    # Convert to numpy array for processing
    image_np = np.array(image)
    
    return image, image_np

def parse_point_prompt(point_str):
    """Parse the point prompt string into coordinates."""
    try:
        x, y = map(int, point_str.split(','))
        return [(x, y)]
    except Exception as e:
        logger.error(f"Error parsing point prompt: {e}")
        logger.info("Using center of the image as a fallback")
        return None  # Will be handled by creating a center point in the main function

def parse_box_prompt(box_str, image_size):
    """Parse the box prompt string into coordinates."""
    try:
        x1, y1, x2, y2 = map(int, box_str.split(','))
        return [x1, y1, x2, y2]
    except Exception as e:
        logger.error(f"Error parsing box prompt: {e}")
        
        # Create a default box in the center (1/3 of the image size)
        h, w = image_size
        center_x, center_y = w // 2, h // 2
        box_size = min(w, h) // 3
        
        x1 = center_x - box_size // 2
        y1 = center_y - box_size // 2
        x2 = center_x + box_size // 2
        y2 = center_y + box_size // 2
        
        logger.info(f"Using default box: [{x1}, {y1}, {x2}, {y2}]")
        return [x1, y1, x2, y2]

def segment_with_transformers(sam_dict, image, image_np, args):
    """Segment using HuggingFace Transformers implementation."""
    model = sam_dict["model"]
    processor = sam_dict["processor"]
    device = sam_dict["device"]
    supports_text = sam_dict["supports_text"]
    
    # Process based on prompt type
    if args.prompt_type == "text" and supports_text:
        logger.info(f"Using text prompt: '{args.prompt}'")
        
        # Process text and image
        inputs = processor(
            text=[args.prompt],
            images=image_np, 
            return_tensors="pt"
        ).to(device)
        
        # Generate masks
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Process masks
        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )
        
        # Get best mask (first batch, first text prompt, best mask)
        mask = masks[0][0][0].numpy()
    
    elif args.prompt_type == "point":
        # Parse point prompt or use center
        if args.point:
            points = parse_point_prompt(args.point)
        else:
            # Use center of image
            h, w = image_np.shape[:2]
            points = [(w // 2, h // 2)]
        
        logger.info(f"Using point prompt: {points}")
        
        # Process point and image
        inputs = processor(
            images=image_np,
            input_points=[points],  # List of points per image
            return_tensors="pt"
        ).to(device)
        
        # Generate masks
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Process masks
        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )
        
        # Get best mask (first batch, best mask)
        mask = masks[0][0].numpy()
    
    elif args.prompt_type == "box":
        # Parse box prompt or use default
        if args.box:
            box = parse_box_prompt(args.box, image_np.shape[:2])
        else:
            # Use center box 
            h, w = image_np.shape[:2]
            box_size = min(w, h) // 3
            box = [
                w // 2 - box_size // 2,
                h // 2 - box_size // 2,
                w // 2 + box_size // 2,
                h // 2 + box_size // 2
            ]
            
        logger.info(f"Using box prompt: {box}")
        
        # Process box and image
        inputs = processor(
            images=image_np,
            input_boxes=[[box]],  # List of boxes per image
            return_tensors="pt"
        ).to(device)
        
        # Generate masks
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Process masks
        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )
        
        # Get best mask (first batch, best mask)
        mask = masks[0][0].numpy()
    
    else:
        # Text prompt was requested but not supported, fallback to point
        logger.warning("Text prompts not supported by this model. Falling back to center point prompt.")
        
        # Use center of image
        h, w = image_np.shape[:2]
        points = [(w // 2, h // 2)]
        
        # Process point and image
        inputs = processor(
            images=image_np,
            input_points=[points],
            return_tensors="pt"
        ).to(device)
        
        # Generate masks
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Process masks
        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )
        
        # Get best mask (first batch, best mask)
        mask = masks[0][0].numpy()
    
    return mask

def segment_with_official_api(sam_dict, image, image_np, args):
    """Segment using the official SAM API."""
    predictor = sam_dict["predictor"]
    
    # Set the image in the predictor
    predictor.set_image(image_np)
    
    # Process based on prompt type
    if args.prompt_type == "point":
        # Parse point prompt or use center
        if args.point:
            points = parse_point_prompt(args.point)
        else:
            # Use center of image
            h, w = image_np.shape[:2]
            points = [(w // 2, h // 2)]
        
        logger.info(f"Using point prompt: {points}")
        
        # Convert points to numpy arrays
        input_point = np.array([p for p in points])
        input_label = np.array([1] * len(points))  # 1 for foreground
        
        # Generate masks
        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True
        )
        
        # Get best mask (highest score)
        best_idx = np.argmax(scores)
        mask = masks[best_idx]
    
    elif args.prompt_type == "box":
        # Parse box prompt or use default
        if args.box:
            box = parse_box_prompt(args.box, image_np.shape[:2])
        else:
            # Use center box 
            h, w = image_np.shape[:2]
            box_size = min(w, h) // 3
            box = [
                w // 2 - box_size // 2,
                h // 2 - box_size // 2,
                w // 2 + box_size // 2,
                h // 2 + box_size // 2
            ]
            
        logger.info(f"Using box prompt: {box}")
        
        # Convert box to numpy array
        input_box = np.array(box)
        
        # Generate masks
        masks, scores, _ = predictor.predict(
            box=input_box,
            multimask_output=True
        )
        
        # Get best mask (highest score)
        best_idx = np.argmax(scores)
        mask = masks[best_idx]
    
    else:
        # Text prompt not supported, fallback to point
        logger.warning("Text prompts not supported by native SAM API. Falling back to center point prompt.")
        
        # Use center of image
        h, w = image_np.shape[:2]
        input_point = np.array([[w // 2, h // 2]])
        input_label = np.array([1])  # 1 for foreground
        
        # Generate masks
        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True
        )
        
        # Get best mask (highest score)
        best_idx = np.argmax(scores)
        mask = masks[best_idx]
    
    return mask

def visualize_results(image_np, mask, output_path):
    """Create a visualization of the segmentation results."""
    # Create transparent overlay
    overlay = np.zeros_like(image_np, dtype=np.uint8)
    overlay[mask] = [0, 255, 0]  # Green for tumor
    
    # Create blended image
    alpha = 0.5
    blended = cv2.addWeighted(image_np, 1, overlay, alpha, 0)
    
    # Draw contour
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(blended, contours, -1, (255, 255, 0), 2)  # Yellow contour
    
    # Save visualization
    plt.figure(figsize=(10, 8))
    plt.imshow(blended)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory
    output_dir = create_output_dir(args.output_dir)
    logger.info(f"Output directory: {output_dir}")
    
    # Configure file handler for logging
    file_handler = logging.FileHandler(output_dir / "segmentation.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    
    # Load image
    logger.info(f"Loading image: {args.input_image}")
    image, image_np = load_image(args.input_image)
    
    # Save original image
    image.save(output_dir / "original.png")
    
    # Set default text prompt if needed
    if args.prompt_type == "text" and not args.prompt:
        args.prompt = "tumor in brain MRI"
        logger.info(f"Using default text prompt: '{args.prompt}'")
    
    # Load SAM2 model
    sam_dict = load_sam2_model(args.sam_model)
    
    # Measure segmentation time
    start_time = time.time()
    
    # Segment based on the implementation type
    if sam_dict["type"] == "transformers" or sam_dict["type"] == "transformers_fallback":
        mask = segment_with_transformers(sam_dict, image, image_np, args)
    else:
        mask = segment_with_official_api(sam_dict, image, image_np, args)
    
    # End timer
    segmentation_time = time.time() - start_time
    logger.info(f"Segmentation completed in {segmentation_time:.2f} seconds")
    
    # Save mask
    mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
    mask_image.save(output_dir / "mask.png")
    
    # Create visualization
    visualize_results(image_np, mask, output_dir / "visualization.png")
    
    # Save metadata
    metadata = {
        "input_image": args.input_image,
        "prompt_type": args.prompt_type,
        "prompt": args.prompt if args.prompt_type == "text" else None,
        "point": args.point if args.prompt_type == "point" else None,
        "box": args.box if args.prompt_type == "box" else None,
        "segmentation_time": segmentation_time,
        "sam_model": args.sam_model,
        "sam_implementation": sam_dict["type"],
        "supports_text": sam_dict["supports_text"],
        "cuda_available": torch.cuda.is_available(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    
    logger.info(f"Segmentation complete. Results saved to {output_dir}")
    print(f"\nSegmentation results saved to: {output_dir}")
    print(f"Mask saved to: {output_dir / 'mask.png'}")
    print(f"Visualization saved to: {output_dir / 'visualization.png'}")

if __name__ == "__main__":
    main() 