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
    train_parser.add_argument("--config", type=str, default="configs/default_config.yaml", 
                             help="Path to configuration file")
    # Add any other train-specific args if needed, mirroring train.py
    train_parser.add_argument("--output_dir", type=str, help="Override output directory in config")
    train_parser.add_argument("--learning_rate", type=float, help="Override learning rate in config")
    train_parser.add_argument("--num_epochs", type=int, help="Override num epochs in config")
    train_parser.add_argument("--log_wandb", action="store_true", help="Force log to Weights & Biases")
    
    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate medical images using a pre-trained ControlNet")
    generate_parser.add_argument("--config", type=str, default="configs/default_config.yaml", 
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
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate downstream task performance with/without synthetic data")
    evaluate_parser.add_argument("--real_data_dir", type=str, required=True, 
                               help="Directory with real training/test data (ImageFolder or custom structure)")
    evaluate_parser.add_argument("--synthetic_data_dir", type=str, required=True, 
                               help="Directory containing the generated synthetic images")
    evaluate_parser.add_argument("--output_dir", type=str, default="evaluation_results",
                               help="Directory to save evaluation results (plots, logs, models)")
    evaluate_parser.add_argument("--task", type=str, default="classification",
                               choices=["segmentation", "classification"], 
                               help="Downstream task to evaluate")
    evaluate_parser.add_argument("--batch_size", type=int, help="Override evaluation batch size")
    evaluate_parser.add_argument("--num_epochs", type=int, help="Override evaluation training epochs")
    evaluate_parser.add_argument("--image_size", type=int, help="Override evaluation image size")
    evaluate_parser.add_argument("--mask_folder", type=str, help="Subfolder for real masks (segmentation)")
    evaluate_parser.add_argument("--synthetic_mask_folder", type=str, help="Subfolder for synthetic masks (segmentation)")
    evaluate_parser.add_argument("--test_split", type=float, help="Override test split fraction for real data")
    evaluate_parser.add_argument("--val_split", type=float, help="Override validation split fraction for real data")
    evaluate_parser.add_argument("--seed", type=int, help="Override evaluation random seed")
    
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
        # Convert snake_case to --kebab-case
        arg_name = "--" + key.replace("_", "-")
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
    cmd = build_command("train.py", args_dict)
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
        "batch_size": args.batch_size
    }
    cmd = build_command("generate.py", args_dict)
    print(f"Running command: {' '.join(shlex.quote(c) for c in cmd)}")
    subprocess.run(cmd, check=True)


def run_evaluate(args):
    """
    Run evaluation script.
    
    Args:
        args (argparse.Namespace): Parsed arguments
    """
    args_dict = {
        "real_data_dir": args.real_data_dir,
        "synthetic_data_dir": args.synthetic_data_dir,
        "output_dir": args.output_dir,
        "task": args.task,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "image_size": args.image_size,
        "mask_folder": args.mask_folder,
        "synthetic_mask_folder": args.synthetic_mask_folder,
        "test_split": args.test_split,
        "val_split": args.val_split,
        "seed": args.seed
    }
    cmd = build_command("evaluate.py", args_dict)
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