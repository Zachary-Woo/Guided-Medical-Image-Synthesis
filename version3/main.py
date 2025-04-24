#!/usr/bin/env python
"""
Version 3: Medical Brain MRI Synthesis System
Main entry point for brain MRI generation with anatomical control

This script provides a CLI for all project tasks:
- Data preparation from BraTS dataset
- LoRA fine-tuning for domain adaptation 
- ControlNet training for structural control
- SAM2 fine-tuning for tumor segmentation
- Inference with prompt-based MRI generation
"""

import sys
import argparse
import subprocess
import logging
import shlex
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Brain MRI Synthesis v3 - Generate anatomically accurate brain MRIs"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run", required=True)
    
    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate MRI from text prompt and/or mask")
    generate_parser.add_argument("--prompt", type=str, required=True,
                               help="Text prompt, e.g. 'T1 weighted axial brain MRI with tumor in left temporal lobe'")
    generate_parser.add_argument("--mask", type=str,
                               help="Optional path to binary tumor mask (white=tumor)")
    generate_parser.add_argument("--output_dir", type=str, default="output/version3/generated",
                               help="Output directory")
    generate_parser.add_argument("--lora_weights", type=str, default="version3/models/mri_lora",
                               help="Path to LoRA weights directory")
    generate_parser.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-v1-5",
                               help="Base model ID")
    generate_parser.add_argument("--seed", type=int, default=42,
                               help="Random seed for reproducibility")
    generate_parser.add_argument("--num_inference_steps", type=int, default=30,
                               help="Number of diffusion steps")
    generate_parser.add_argument("--guidance_scale", type=float, default=7.5,
                               help="Classifier-free guidance scale")
    generate_parser.add_argument("--controlnet_conditioning_scale", type=float, default=1.0,
                               help="ControlNet conditioning scale")
    generate_parser.add_argument("--create_mask", action="store_true",
                               help="Automatically create a tumor mask based on prompt")
    generate_parser.add_argument("--slice_level", type=str, default="mid-axial",
                               choices=["superior", "mid-axial", "inferior", "ventricles", "basal-ganglia", "cerebellum"],
                               help="Specify the axial slice level of the brain to generate")
    # Visualization options for generate command
    generate_parser.add_argument("--visualize", action="store_true",
                               help="Run visualization after generation to compare with real MRI data")
    generate_parser.add_argument("--brats_dir", type=str, default=None,
                               help="BraTS patient directory for comparison visualization")
    generate_parser.add_argument("--compare_modality", type=str, default="t1",
                               choices=["t1", "t2", "flair", "t1ce"],
                               help="MRI modality to use from BraTS dataset for comparison")
    generate_parser.add_argument("--show_visualization", action="store_true",
                               help="Display the visualization in addition to saving it")
    
    # Visualize command
    visualize_parser = subparsers.add_parser("visualize", help="Visualize and compare generated MRIs with real BraTS data")
    visualize_parser.add_argument("--generated_dir", type=str, required=True,
                                help="Directory containing generated MRI results")
    visualize_parser.add_argument("--brats_dir", type=str, required=True,
                                help="BraTS patient directory for comparison")
    visualize_parser.add_argument("--modality", type=str, default="t1",
                                choices=["t1", "t2", "flair", "t1ce"],
                                help="MRI modality to use from BraTS dataset")
    visualize_parser.add_argument("--slice_level", type=str, default=None,
                                choices=["superior", "mid-axial", "inferior", "ventricles", "basal-ganglia", "cerebellum"],
                                help="Axial slice level (if not provided, will be read from generation metadata)")
    visualize_parser.add_argument("--output_path", type=str, default=None,
                                help="Output path for visualization (default: generated_dir/comparison.png)")
    visualize_parser.add_argument("--show", action="store_true",
                                help="Display the visualization in addition to saving it")
    
    # Prepare data command
    prepare_parser = subparsers.add_parser("prepare", help="Prepare BraTS data for training")
    prepare_parser.add_argument("--brats_path", type=str, required=True,
                              help="Path to BraTS dataset root")
    prepare_parser.add_argument("--output_dir", type=str, default="data/processed_brats",
                              help="Output directory for processed data")
    prepare_parser.add_argument("--sample_count", type=int, default=1000,
                              help="Number of slices to extract")
    prepare_parser.add_argument("--modality", type=str, default="t1",
                              choices=["t1", "t2", "flair", "t1ce"],
                              help="MRI modality to extract")
    prepare_parser.add_argument("--image_size", type=int, default=512,
                              help="Target image size")
    prepare_parser.add_argument("--include_healthy", action="store_true",
                              help="Include healthy (no tumor) slices")
    
    # Train LoRA command
    train_lora_parser = subparsers.add_parser("train-lora", help="Train LoRA adapter for brain MRI")
    train_lora_parser.add_argument("--data_path", type=str, required=True,
                                 help="Path to prepared BraTS data")
    train_lora_parser.add_argument("--output_dir", type=str, default="version3/models/mri_lora",
                                 help="Output directory for LoRA weights")
    train_lora_parser.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-v1-5",
                                 help="Base model to adapt")
    train_lora_parser.add_argument("--resolution", type=int, default=512,
                                 help="Training resolution")
    train_lora_parser.add_argument("--train_batch_size", type=int, default=1,
                                 help="Training batch size")
    train_lora_parser.add_argument("--num_train_epochs", type=int, default=100,
                                 help="Number of training epochs")
    train_lora_parser.add_argument("--max_train_steps", type=int, default=None,
                               help="Max training steps. If set, overrides num_train_epochs")
    train_lora_parser.add_argument("--learning_rate", type=float, default=1e-4,
                                 help="Learning rate")
    train_lora_parser.add_argument("--rank", type=int, default=4,
                                 help="LoRA rank")
    train_lora_parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                                 help="Number of gradient accumulation steps")
    train_lora_parser.add_argument("--mixed_precision", type=str, default="fp16",
                                 choices=["no", "fp16", "bf16"],
                                 help="Mixed precision training")
    train_lora_parser.add_argument("--use_8bit_adam", action="store_true",
                                 help="Use 8-bit Adam optimizer")
    train_lora_parser.add_argument("--gradient_checkpointing", action="store_true",
                                 help="Enable gradient checkpointing")
    
    # Train ControlNet command
    train_controlnet_parser = subparsers.add_parser("train-controlnet", help="Train ControlNet for structure conditioning")
    train_controlnet_parser.add_argument("--data_path", type=str, required=True,
                                      help="Path to prepared BraTS data")
    train_controlnet_parser.add_argument("--output_dir", type=str, default="version3/models/segmentation_controlnet",
                                      help="Output directory for ControlNet weights")
    train_controlnet_parser.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-v1-5",
                                      help="Base model")
    train_controlnet_parser.add_argument("--controlnet_model", type=str, default="lllyasviel/sd-controlnet-seg",
                                      help="Base ControlNet model to fine-tune")
    train_controlnet_parser.add_argument("--resolution", type=int, default=512,
                                      help="Training resolution")
    train_controlnet_parser.add_argument("--train_batch_size", type=int, default=1,
                                      help="Training batch size")
    train_controlnet_parser.add_argument("--num_train_epochs", type=int, default=100,
                                      help="Number of training epochs")
    train_controlnet_parser.add_argument("--learning_rate", type=float, default=1e-5,
                                      help="Learning rate")
    train_controlnet_parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                                      help="Number of gradient accumulation steps")
    train_controlnet_parser.add_argument("--mixed_precision", type=str, default="fp16",
                                      choices=["no", "fp16", "bf16"],
                                      help="Mixed precision training")
    train_controlnet_parser.add_argument("--use_8bit_adam", action="store_true",
                                      help="Use 8-bit Adam optimizer")
    
    # Train SAM2 command
    train_sam2_parser = subparsers.add_parser("train-sam2", help="Fine-tune SAM2 for tumor segmentation")
    train_sam2_parser.add_argument("--brats_path", type=str, required=True,
                                help="Path to BraTS dataset with MRI and segmentation files")
    train_sam2_parser.add_argument("--output_dir", type=str, default="version3/models/sam2_finetuned",
                                help="Directory to save fine-tuned model")
    train_sam2_parser.add_argument("--sam_checkpoint", type=str, default="sam2_hiera_small.pt",
                                help="Path to SAM2 checkpoint (for local implementation)")
    train_sam2_parser.add_argument("--sam_config", type=str, default="sam2_hiera_s.yaml",
                                help="Path to SAM2 config file (for local implementation)")
    train_sam2_parser.add_argument("--sam_model_id", type=str, default="facebook/sam2",
                                help="SAM2 model ID (for transformers implementation)")
    train_sam2_parser.add_argument("--train_image_encoder", action="store_true",
                                help="Whether to train the image encoder (requires more GPU memory)")
    train_sam2_parser.add_argument("--learning_rate", type=float, default=1e-5,
                                help="Learning rate for training")
    train_sam2_parser.add_argument("--max_train_steps", type=int, default=5000,
                                help="Maximum number of training steps")
    train_sam2_parser.add_argument("--augmentation", action="store_true",
                                help="Apply data augmentation during training")
    train_sam2_parser.add_argument("--slice_axis", type=int, default=2,
                                help="Axis for slicing 3D volumes (0, 1, or 2)")
    
    # Run SAM2 command
    sam_parser = subparsers.add_parser("segment", help="Segment an MRI image with SAM2")
    sam_parser.add_argument("--input_image", type=str, required=True,
                         help="Input MRI image path")
    sam_parser.add_argument("--output_dir", type=str, default="output/version3/segments",
                         help="Output directory for segmentation results")
    sam_parser.add_argument("--sam_model", type=str, default="facebook/sam2",
                         help="SAM2 model identifier")
    sam_parser.add_argument("--prompt_type", type=str, default="text",
                         choices=["text", "point", "box"],
                         help="Type of prompt to provide to SAM2")
    sam_parser.add_argument("--prompt", type=str,
                         help="Text prompt for text-guided segmentation")
    sam_parser.add_argument("--point", type=str,
                         help="Point prompt as 'x,y' for point-guided segmentation")
    sam_parser.add_argument("--box", type=str,
                         help="Box prompt as 'x1,y1,x2,y2' for box-guided segmentation")
    
    # Test CUDA command 
    test_cuda_parser = subparsers.add_parser("test-cuda", help="Test CUDA availability")
    
    return parser.parse_args()

def build_command(script_name, args_dict):
    """Build a command list for subprocess, handling None values."""
    cmd = [sys.executable, script_name]
    for key, value in args_dict.items():
        # Skip the 'command' argument itself
        if key == 'command' or value is None:
            continue
        arg_name = f"--{key}"
        if isinstance(value, bool):
            if value:
                cmd.append(arg_name)
        elif isinstance(value, list):
            if value:
                cmd.append(arg_name)
                cmd.extend(map(str, value))
        else:
            cmd.append(arg_name)
            cmd.append(str(value))
    return cmd

def run_generate(args):
    """Run MRI generation script"""
    cmd = [
        "python",
        "version3/scripts/generate_mri.py",
        "--prompt", args.prompt,
        "--output_dir", args.output_dir,
        "--seed", str(args.seed),
        "--num_inference_steps", str(args.num_inference_steps),
        "--guidance_scale", str(args.guidance_scale)
    ]

    if args.mask:
        cmd.extend(["--mask", args.mask])
    else:
        cmd.extend(["--create_mask", "--slice_level", args.slice_level])

    if args.lora_weights:
        cmd.extend(["--lora_weights", args.lora_weights])

    if args.base_model:
        cmd.extend(["--base_model", args.base_model])

    # Add visualization flag if requested
    if args.visualize:
        cmd.extend(["--visualize"])
        if args.brats_dir:
            cmd.extend(["--brats_dir", args.brats_dir])
        if args.compare_modality:
            cmd.extend(["--compare_modality", args.compare_modality])
        if args.show_visualization:
            cmd.extend(["--show_visualization"])

    # Run generation script
    subprocess.run(cmd, check=True)

def run_prepare(args):
    """Run data preparation script."""
    args_dict = vars(args)
    cmd = build_command("version3/scripts/prepare_brats.py", args_dict)
    print(f"Running command: {' '.join(shlex.quote(c) for c in cmd)}")
    subprocess.run(cmd, check=True)

def run_train_lora(args):
    """Run LoRA training script."""
    args_dict = vars(args)
    cmd = build_command("version3/scripts/train_lora.py", args_dict)
    print(f"Running command: {' '.join(shlex.quote(c) for c in cmd)}")
    subprocess.run(cmd, check=True)

def run_train_controlnet(args):
    """Run ControlNet training script."""
    args_dict = vars(args)
    cmd = build_command("version3/scripts/train_controlnet.py", args_dict)
    print(f"Running command: {' '.join(shlex.quote(c) for c in cmd)}")
    subprocess.run(cmd, check=True)

def run_train_sam2(args):
    """Run SAM2 fine-tuning script."""
    args_dict = vars(args)
    cmd = build_command("version3/scripts/train_sam2.py", args_dict)
    print(f"Running command: {' '.join(shlex.quote(c) for c in cmd)}")
    subprocess.run(cmd, check=True)

def run_segment(args):
    """Run SAM2 segmentation script."""
    args_dict = vars(args)
    cmd = build_command("version3/scripts/segment_mri.py", args_dict)
    print(f"Running command: {' '.join(shlex.quote(c) for c in cmd)}")
    subprocess.run(cmd, check=True)

def run_visualize(args):
    """Run visualization script."""
    args_dict = vars(args)
    cmd = build_command("version3/scripts/visualize_results.py", args_dict)
    print(f"Running command: {' '.join(shlex.quote(c) for c in cmd)}")
    subprocess.run(cmd, check=True)

def test_cuda():
    """Test CUDA availability."""
    print("\n" + "="*80)
    print(" CUDA TEST ".center(80, "="))
    print("="*80 + "\n")
    
    if not torch.cuda.is_available():
        print("❌ CUDA is NOT available")
        print("Please make sure you have a CUDA-capable GPU and proper drivers installed.")
        return False
    
    print("✅ CUDA is available")
    print(f"    - GPU: {torch.cuda.get_device_name(0)}")
    print(f"    - CUDA version: {torch.version.cuda}")
    print(f"    - Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Test a small tensor operation
    try:
        print("\nRunning a small tensor operation on GPU...")
        x = torch.rand(1000, 1000).cuda()
        y = torch.matmul(x, x)
        del x, y
        torch.cuda.empty_cache()
        print("✅ Tensor operation successful")
    except Exception as e:
        print(f"❌ Error during tensor operation: {e}")
        return False
    
    return True

def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        if args.command == "generate":
            run_generate(args)
        elif args.command == "prepare":
            run_prepare(args)
        elif args.command == "train-lora":
            run_train_lora(args)
        elif args.command == "train-controlnet":
            run_train_controlnet(args)
        elif args.command == "train-sam2":
            run_train_sam2(args)
        elif args.command == "segment":
            run_segment(args)
        elif args.command == "test-cuda":
            test_cuda()
        elif args.command == "visualize":
            run_visualize(args)
    except subprocess.CalledProcessError as e:
        print(f"Error running command '{args.command}': {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 