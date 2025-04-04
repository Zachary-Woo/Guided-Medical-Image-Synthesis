#!/usr/bin/env python
"""
Main entry point for the medical image synthesis project.
This script provides a convenient CLI for all project tasks.
"""

import os
import argparse
import subprocess
import sys
import shlex # Import shlex for safer command splitting


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Medical Image Synthesis using ControlNet"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run", required=True)
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train ControlNet model (Currently less relevant)")
    train_parser.add_argument("--config", type=str, default="version1/configs/default_config.yaml", 
                             help="Path to configuration file")
    # Add any other train-specific args if needed, mirroring train.py
    train_parser.add_argument("--output_dir", type=str, help="Override output directory in config")
    train_parser.add_argument("--learning_rate", type=float, help="Override learning rate in config")
    train_parser.add_argument("--num_epochs", type=int, help="Override num epochs in config")
    train_parser.add_argument("--log_wandb", action="store_true", help="Force log to Weights & Biases")
    
    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate medical images using a pre-trained ControlNet")
    generate_parser.add_argument("--config", type=str, default="version1/configs/default_config.yaml", 
                               help="Path to configuration file")
    # Removed checkpoint, uses config now
    # generate_parser.add_argument("--checkpoint", type=str, required=True, 
    #                            help="Path to ControlNet checkpoint")
    generate_parser.add_argument("--conditioning_source_images", type=str, nargs="+", default=[],
                               help="Paths to source images for conditioning (e.g., masks, real images)")
    generate_parser.add_argument("--prompts", type=str, nargs="+", default=[],
                               help="Text prompts for generation (overrides config)")
    generate_parser.add_argument("--output_dir", type=str, help="Override output directory in config")
    generate_parser.add_argument("--steps", type=int, help="Override inference steps in config")
    generate_parser.add_argument("--guidance_scale", type=float, help="Override guidance scale in config")
    generate_parser.add_argument("--seed", type=int, help="Override random seed in config")
    generate_parser.add_argument("--batch_size", type=int, help="Override batch size (usually 1 for controlled generation)")
    generate_parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate downstream task performance with/without synthetic data")
    evaluate_parser.add_argument("--config", type=str, default="version1/configs/medmnist_canny_demo.yaml",
                               help="Path to config file (specifies dataset name, etc.)")
    evaluate_parser.add_argument("--synthetic_data_dir", type=str, required=True,
                               help="Directory containing the generated synthetic images")
    evaluate_parser.add_argument("--output_dir", type=str, # Default now comes from config/logic in evaluate.py
                               help="Directory to save evaluation results (plots, logs, models)")
    evaluate_parser.add_argument("--medmnist_download_dir", type=str, # Added arg to specify download location
                               help="Root directory to download MedMNIST data (overrides default ./data in evaluate.py)")
    evaluate_parser.add_argument("--task", type=str, # Default in evaluate.py
                               choices=["segmentation", "classification"],
                               help="Downstream task to evaluate (overrides config/default)")
    evaluate_parser.add_argument("--batch_size", type=int, help="Override evaluation batch size")
    evaluate_parser.add_argument("--num_epochs", type=int, help="Override evaluation training epochs")
    evaluate_parser.add_argument("--image_size", type=int, help="Override evaluation image size")
    evaluate_parser.add_argument("--learning_rate", type=float, help="Override downstream learning rate")
    evaluate_parser.add_argument("--synthetic_mask_folder", type=str, help="Subfolder for synthetic masks (segmentation)")
    evaluate_parser.add_argument("--seed", type=int, help="Override evaluation random seed")
    
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
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test pipeline components without performing full generation")
    test_parser.add_argument("--config", type=str, default="version1/configs/medmnist_canny_demo.yaml",
                           help="Path to configuration file")
    test_parser.add_argument("--conditioning_image", type=str,
                           help="Path to a test image for conditioning")
    test_parser.add_argument("--verbose", action="store_true",
                           help="Enable verbose logging")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Install requirements (optional)")
    # No arguments needed for setup usually
    
    return parser.parse_args()


def build_command(script_name: str, args_dict: dict) -> list:
    """Builds a command list for subprocess, handling None values and lists."""
    cmd = [sys.executable, script_name]
    for key, value in args_dict.items():
        if value is None: # Skip None values
            continue
        # Keep underscores for argument names to match argparse definitions
        arg_name = "--" + key
        if isinstance(value, bool):
            if value:
                cmd.append(arg_name)
        elif isinstance(value, list):
            if value: # Only add if list is not empty
                cmd.append(arg_name)
                cmd.extend(map(str, value))
        else:
            cmd.append(arg_name)
            cmd.append(str(value))
    return cmd


def run_train(args):
    """
    Run training script.
    
    Args:
        args (argparse.Namespace): Parsed arguments
    """
    print("Note: Training is currently less relevant with the focus on pre-trained models.")
    args_dict = {
        "config": args.config,
        "output_dir": args.output_dir,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "log_wandb": args.log_wandb
    }
    cmd = build_command("version1/train.py", args_dict)
    print(f"Running command: {' '.join(shlex.quote(c) for c in cmd)}")
    subprocess.run(cmd, check=True)


def run_generate(args):
    """
    Run generation script.
    
    Args:
        args (argparse.Namespace): Parsed arguments
    """
    args_dict = {
        "config": args.config,
        "conditioning_source_images": args.conditioning_source_images,
        "prompts": args.prompts,
        "output_dir": args.output_dir,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "debug": args.debug
    }
    cmd = build_command("version1/generate.py", args_dict)
    print(f"Running command: {' '.join(shlex.quote(c) for c in cmd)}")
    subprocess.run(cmd, check=True)


def run_evaluate(args):
    """
    Run evaluation script.

    Args:
        args (argparse.Namespace): Parsed arguments
    """
    args_dict = {
        # Pass only the relevant args to evaluate.py
        "config": args.config,
        "synthetic_data_dir": args.synthetic_data_dir,
        "output_dir": args.output_dir,
        "medmnist_download_dir": args.medmnist_download_dir,
        "task": args.task,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "image_size": args.image_size,
        "learning_rate": args.learning_rate,
        "synthetic_mask_folder": args.synthetic_mask_folder,
        "seed": args.seed
    }
    cmd = build_command("version1/evaluate.py", args_dict)
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


def run_test(args):
    """
    Run pipeline testing script.
    
    Args:
        args (argparse.Namespace): Parsed arguments
    """
    args_dict = {
        "config": args.config,
        "conditioning_image": args.conditioning_image,
        "verbose": args.verbose
    }
    cmd = build_command("version1/test_pipeline.py", args_dict)
    print(f"Running command: {' '.join(shlex.quote(c) for c in cmd)}")
    subprocess.run(cmd, check=True)


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
    except subprocess.CalledProcessError as e:
        print(f"Failed to install requirements: {e}")
        sys.exit(1)
    except FileNotFoundError:
         print(f"Error: '{sys.executable}' or 'pip' command not found. Ensure Python and pip are installed and in your PATH.")
         sys.exit(1)


def main():
    """
    Main function.
    """
    args = parse_args()
    
    try:
        if args.command == "train":
            run_train(args)
        elif args.command == "generate":
            run_generate(args)
        elif args.command == "evaluate":
            run_evaluate(args)
        elif args.command == "extract":
            run_extract(args)
        elif args.command == "test":
            run_test(args)
        elif args.command == "setup":
            run_setup()
        # No else needed because subparsers are required
    except subprocess.CalledProcessError as e:
        print(f"Error running command '{args.command}': {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main() 