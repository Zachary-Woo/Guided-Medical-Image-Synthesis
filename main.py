#!/usr/bin/env python
"""
Main entry point for the medical image synthesis project.
This script provides a convenient CLI for all project tasks.
"""

import os
import argparse
import subprocess
import sys


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Medical Image Synthesis using ControlNet"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train ControlNet model")
    train_parser.add_argument("--config", type=str, default="configs/default_config.yaml", 
                             help="Path to configuration file")
    
    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate medical images")
    generate_parser.add_argument("--config", type=str, default="configs/default_config.yaml", 
                               help="Path to configuration file")
    generate_parser.add_argument("--checkpoint", type=str, required=True, 
                               help="Path to ControlNet checkpoint")
    generate_parser.add_argument("--control_images", type=str, nargs="+", 
                               help="Paths to control images (masks)")
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate downstream tasks")
    evaluate_parser.add_argument("--real_data_dir", type=str, required=True, 
                               help="Directory with real data")
    evaluate_parser.add_argument("--synthetic_data_dir", type=str, required=True, 
                               help="Directory with synthetic data")
    evaluate_parser.add_argument("--task", type=str, default="segmentation", 
                               choices=["segmentation", "classification"], 
                               help="Downstream task to evaluate")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Setup project environment")
    
    return parser.parse_args()


def run_train(args):
    """
    Run training script.
    
    Args:
        args (argparse.Namespace): Parsed arguments
    """
    cmd = [sys.executable, "train.py"]
    
    if args.config:
        cmd.extend(["--config", args.config])
    
    subprocess.run(cmd)


def run_generate(args):
    """
    Run generation script.
    
    Args:
        args (argparse.Namespace): Parsed arguments
    """
    cmd = [sys.executable, "generate.py", "--config", args.config, "--checkpoint", args.checkpoint]
    
    if args.control_images:
        cmd.extend(["--control_images"] + args.control_images)
    
    subprocess.run(cmd)


def run_evaluate(args):
    """
    Run evaluation script.
    
    Args:
        args (argparse.Namespace): Parsed arguments
    """
    cmd = [
        sys.executable, "evaluate.py",
        "--real_data_dir", args.real_data_dir,
        "--synthetic_data_dir", args.synthetic_data_dir,
        "--task", args.task
    ]
    
    subprocess.run(cmd)


def run_setup():
    """
    Setup project environment.
    """
    # Create necessary directories
    os.makedirs("data/bci_dataset", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    os.makedirs("experiments/baseline_generation", exist_ok=True)
    os.makedirs("experiments/mask_conditioned_generation", exist_ok=True)
    os.makedirs("experiments/data_augmentation", exist_ok=True)
    
    # Install requirements
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    print("Project environment setup complete!")


def main():
    """
    Main function.
    """
    args = parse_args()
    
    if args.command == "train":
        run_train(args)
    elif args.command == "generate":
        run_generate(args)
    elif args.command == "evaluate":
        run_evaluate(args)
    elif args.command == "setup":
        run_setup()
    else:
        print("Please specify a command. Use --help for more information.")


if __name__ == "__main__":
    main() 