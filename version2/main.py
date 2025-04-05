#!/usr/bin/env python
"""
Main entry point for the enhanced medical image synthesis project (Version 2).
This script provides a convenient CLI for all project tasks with support for
advanced features like stain normalization and LoRA fine-tuning.
"""

import os
import argparse
import subprocess
import sys
import shlex


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Advanced Medical Image Synthesis with ControlNet v2"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run", required=True)
    
    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate medical images with enhanced ControlNet")
    generate_parser.add_argument("--condition_image", type=str, required=True,
                               help="Path to conditioning image")
    generate_parser.add_argument("--reference_image", type=str,
                               help="Path to staining reference image (if not provided, default H&E colors will be used)")
    generate_parser.add_argument("--prompt", type=str,
                               help="Text prompt for generation")
    generate_parser.add_argument("--output_dir", type=str, default="output/enhanced_controlnet_v2",
                               help="Base output directory (will be auto-incremented)")
    generate_parser.add_argument("--stain_norm", type=str, default="macenko",
                               choices=["macenko", "reinhard", "none"],
                               help="Stain normalization method")
    generate_parser.add_argument("--steps", type=int, default=50,
                               help="Number of inference steps")
    generate_parser.add_argument("--guidance_scale", type=float, default=9.0,
                               help="Guidance scale")
    generate_parser.add_argument("--lora_model", type=str,
                               help="Optional path to LoRA adapter weights")
    generate_parser.add_argument("--seed", type=int,
                               help="Random seed")
    generate_parser.add_argument("--num_images", type=int, default=1,
                               help="Number of images to generate")
    generate_parser.add_argument("--debug", action="store_true", 
                               help="Enable debug logging")
    
    # Extract command for samples
    extract_parser = subparsers.add_parser("extract", help="Extract samples from MedMNIST dataset for conditioning")
    extract_parser.add_argument("--npz_file", type=str, default="./data/pathmnist_128.npz",
                              help="Path to the NPZ file to extract samples from")
    extract_parser.add_argument("--output_dir", type=str, default="./data/pathmnist_samples",
                              help="Output directory for extracted samples")
    extract_parser.add_argument("--num_samples", type=int, default=8,
                              help="Number of samples to extract")
    extract_parser.add_argument("--image_key", type=str, default="train_images",
                              help="Key for images in NPZ file")
    
    # Prepare command for LoRA training
    prepare_parser = subparsers.add_parser("prepare", help="Prepare dataset for LoRA fine-tuning")
    prepare_parser.add_argument("--dataset", type=str, default="kather_texture",
                               choices=["kather_texture", "breakhis", "pcam", "camelyon", "local"],
                               help="Dataset to prepare")
    prepare_parser.add_argument("--local_dataset_path", type=str,
                               help="Path to local dataset (if 'local' is selected)")
    prepare_parser.add_argument("--output_dir", type=str, default="output/processed_data",
                               help="Output directory for processed dataset")
    prepare_parser.add_argument("--config_output", type=str, default="version2/configs/lora_config.yaml",
                               help="Path to output configuration file")
    prepare_parser.add_argument("--stain_norm", type=str, default="macenko",
                               choices=["macenko", "reinhard", "none"],
                               help="Stain normalization method")
    prepare_parser.add_argument("--reference_image", type=str,
                               help="Path to reference image for stain normalization")
    prepare_parser.add_argument("--num_samples", type=int, default=1000,
                               help="Number of samples to prepare")
    prepare_parser.add_argument("--target_size", type=int, default=512,
                               help="Target image size")
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate downstream task performance with/without synthetic data")
    evaluate_parser.add_argument("--config", type=str, default="configs/medmnist_evaluation.yaml",
                               help="Path to config file (specifies dataset name, etc.)")
    evaluate_parser.add_argument("--synthetic_data_dir", type=str, required=True,
                               help="Directory containing the generated synthetic images")
    evaluate_parser.add_argument("--output_dir", type=str, default="output/evaluation_v2",
                               help="Directory to save evaluation results (plots, logs, models)")
    evaluate_parser.add_argument("--medmnist_download_dir", type=str, default="./data",
                               help="Root directory to download MedMNIST data")
    evaluate_parser.add_argument("--task", type=str, default="classification",
                               choices=["segmentation", "classification"],
                               help="Downstream task to evaluate")
    evaluate_parser.add_argument("--batch_size", type=int,
                               help="Batch size for evaluation")
    evaluate_parser.add_argument("--num_epochs", type=int,
                               help="Number of epochs for downstream training")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test stain normalization and pipeline components")
    test_parser.add_argument("--input_image", type=str, required=True,
                           help="Path to input image for testing stain normalization")
    test_parser.add_argument("--reference_image", type=str,
                           help="Path to reference image for stain normalization")
    test_parser.add_argument("--method", type=str, default="macenko",
                           choices=["macenko", "reinhard"],
                           help="Stain normalization method to test")
    test_parser.add_argument("--output_dir", type=str, default="output/stain_normalization_test",
                           help="Directory to save test outputs (will be auto-incremented)")
    test_parser.add_argument("--save_visualization", action="store_true",
                           help="Save visualization of test results")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Install requirements")
    
    # Setup output command
    setup_output_parser = subparsers.add_parser("setup-output", 
                                               help="Set up output directory structure")
    setup_output_parser.add_argument("--subdirs", nargs="+", default=[],
                                    help="Additional subdirectories to create in output")
    
    # Setup CUDA command
    setup_cuda_parser = subparsers.add_parser("setup-cuda",
                                            help="Install CUDA dependencies for GPU acceleration")
    setup_cuda_parser.add_argument("--force", action="store_true",
                                  help="Force reinstallation even if CUDA is available")
    
    return parser.parse_args()


def build_command(script_name: str, args_dict: dict) -> list:
    """Builds a command list for subprocess, handling None values and lists."""
    cmd = [sys.executable, script_name]
    for key, value in args_dict.items():
        if value is None:  # Skip None values
            continue
        # Keep underscores for argument names to match argparse definitions
        arg_name = "--" + key
        if isinstance(value, bool):
            if value:
                cmd.append(arg_name)
        elif isinstance(value, list):
            if value:  # Only add if list is not empty
                cmd.append(arg_name)
                cmd.extend(map(str, value))
        else:
            cmd.append(arg_name)
            cmd.append(str(value))
    return cmd


def run_generate(args):
    """
    Run enhanced generation script.
    
    Args:
        args (argparse.Namespace): Parsed arguments
    """
    args_dict = {
        "condition_image": args.condition_image,
        "reference_image": args.reference_image,
        "prompt": args.prompt,
        "output_dir": args.output_dir,
        "stain_norm": args.stain_norm,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "lora_model": args.lora_model,
        "seed": args.seed,
        "num_images": args.num_images,
        "debug": args.debug
    }
    cmd = build_command("version2/scripts/enhanced_controlnet_v2.py", args_dict)
    print(f"Running command: {' '.join(shlex.quote(c) for c in cmd)}")
    subprocess.run(cmd, check=True)


def run_extract(args):
    """
    Run sample extraction script.
    
    Args:
        args (argparse.Namespace): Parsed arguments
    """
    args_dict = {
        "npz_file": args.npz_file,
        "output_dir": args.output_dir,
        "num_samples": args.num_samples,
        "image_key": args.image_key
    }
    cmd = build_command("version1/extract_medmnist_samples.py", args_dict)
    print(f"Running command: {' '.join(shlex.quote(c) for c in cmd)}")
    subprocess.run(cmd, check=True)


def run_prepare(args):
    """
    Run LoRA dataset preparation script.
    
    Args:
        args (argparse.Namespace): Parsed arguments
    """
    args_dict = {
        "dataset": args.dataset,
        "local_dataset_path": args.local_dataset_path,
        "output_dir": args.output_dir,
        "config_output": args.config_output,
        "stain_norm": args.stain_norm,
        "reference_image": args.reference_image,
        "num_samples": args.num_samples,
        "target_size": args.target_size
    }
    cmd = build_command("version2/scripts/prepare_lora_training.py", args_dict)
    print(f"Running command: {' '.join(shlex.quote(c) for c in cmd)}")
    subprocess.run(cmd, check=True)


def run_evaluate(args):
    """
    Run evaluation script.
    
    Args:
        args (argparse.Namespace): Parsed arguments
    """
    args_dict = {
        "config": args.config,
        "synthetic_data_dir": args.synthetic_data_dir,
        "output_dir": args.output_dir,
        "medmnist_download_dir": args.medmnist_download_dir,
        "task": args.task,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs
    }
    cmd = build_command("version2/scripts/evaluate_v2.py", args_dict)
    print(f"Running command: {' '.join(shlex.quote(c) for c in cmd)}")
    subprocess.run(cmd, check=True)


def run_test(args):
    """
    Run stain normalization test script.
    
    Args:
        args (argparse.Namespace): Parsed arguments
    """
    args_dict = {
        "input_image": args.input_image,
        "reference_image": args.reference_image,
        "method": args.method,
        "output_dir": args.output_dir,
        "save_visualization": args.save_visualization
    }
    cmd = build_command("version2/scripts/test_stain_normalization.py", args_dict)
    print(f"Running command: {' '.join(shlex.quote(c) for c in cmd)}")
    # Use capture_output=False to ensure output is displayed in the console
    result = subprocess.run(cmd, check=True, text=True, capture_output=False)
    
    # Display a success message after completion
    if result.returncode == 0:
        print(f"\nStain normalization test completed successfully!")
        print(f"Results saved to the output directory")


def run_setup():
    """
    Install requirements using pip.
    """
    req_file = "requirements.txt"
    if not os.path.exists(req_file):
        print(f"Error: {req_file} not found.")
        sys.exit(1)

    print(f"Installing requirements from {req_file}...")
    cmd = [sys.executable, "-m", "pip", "install", "-r", req_file]
    try:
        subprocess.run(cmd, check=True)
        print("Requirements installed successfully!")

        # Install additional dependencies for version2
        print("Installing additional dependencies for version2...")
        subprocess.run([sys.executable, "-m", "pip", "install", "diffusers[training]", "transformers", "accelerate", "datasets"], check=True)
        print("Additional dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install requirements: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: '{sys.executable}' or 'pip' command not found. Ensure Python and pip are installed and in your PATH.")
        sys.exit(1)


def run_setup_output(args):
    """
    Run setup output directory script.
    
    Args:
        args (argparse.Namespace): Parsed arguments
    """
    args_dict = {
        "subdirs": args.subdirs
    }
    cmd = build_command("version2/scripts/setup_output_dir.py", args_dict)
    print(f"Running command: {' '.join(shlex.quote(c) for c in cmd)}")
    subprocess.run(cmd, check=True)


def run_setup_cuda(args):
    """
    Install CUDA dependencies for GPU acceleration
    
    Args:
        args (argparse.Namespace): Parsed arguments
    """
    print("\n" + "="*80)
    print(" INSTALLING CUDA DEPENDENCIES ".center(80, "="))
    print("="*80 + "\n")
    
    # Check if CUDA is already available
    has_cuda = False
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        
        if has_cuda and not args.force:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ CUDA already available: {gpu_name}")
            print("To reinstall anyway, use the --force flag")
            return
    except:
        print("× PyTorch not found or CUDA not available")
    
    # Install CUDA-enabled PyTorch
    print("Installing CUDA-enabled PyTorch...")
    try:
        # Uninstall existing PyTorch
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"], check=False)
        
        # Install CUDA version
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio", 
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ], check=True)
        print("✓ PyTorch with CUDA installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"× Failed to install PyTorch with CUDA: {e}")
    
    # Install GPU-enabled bitsandbytes
    print("\nInstalling GPU-enabled bitsandbytes...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "bitsandbytes"], check=False)
        subprocess.run([sys.executable, "-m", "pip", "install", "bitsandbytes-windows"], check=True)
        print("✓ bitsandbytes-windows installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"× Failed to install bitsandbytes-windows: {e}")
    
    # Install xformers for memory efficiency
    print("\nInstalling xformers for memory efficiency...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "xformers"], check=True)
        print("✓ xformers installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"× Failed to install xformers: {e}")
    
    print("\nReinstalling all required packages...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✓ All packages reinstalled successfully")
    except subprocess.CalledProcessError as e:
        print(f"× Failed to reinstall packages: {e}")
    
    print("\n" + "="*80)
    print(" CUDA SETUP COMPLETE ".center(80, "="))
    print("="*80)
    print("\nPlease restart your Python environment to apply changes!")
    print("You may need to restart your terminal/IDE as well.")


def main():
    """
    Main function.
    """
    args = parse_args()
    
    try:
        if args.command == "generate":
            run_generate(args)
        elif args.command == "extract":
            run_extract(args)
        elif args.command == "prepare":
            run_prepare(args)
        elif args.command == "evaluate":
            run_evaluate(args)
        elif args.command == "test":
            run_test(args)
        elif args.command == "setup":
            run_setup()
        elif args.command == "setup-output":
            run_setup_output(args)
        elif args.command == "setup-cuda":
            run_setup_cuda(args)
    except subprocess.CalledProcessError as e:
        print(f"Error running command '{args.command}': {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main() 