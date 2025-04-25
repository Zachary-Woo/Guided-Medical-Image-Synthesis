#!/usr/bin/env python
"""
Setup Output Directory Script

This script ensures that the output directory structure is correctly set up.
"""

import sys
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

# Add parent directory to path for importing project modules
parent_dir = str(Path(__file__).resolve().parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from version2.utils.output_helpers import ensure_output_dir_exists
except ImportError:
    logger.error("Output helper utilities not found. Please run from the project root directory.")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Set up output directory structure")
    
    parser.add_argument("--subdirs", nargs="+", default=[],
                        help="Additional subdirectories to create in output")
    
    return parser.parse_args()

def main():
    """Create the output directory structure."""
    args = parse_args()
    
    try:
        # Ensure the main output directory exists
        output_dir = ensure_output_dir_exists()
        logger.info(f"Main output directory: {output_dir}")
        
        # Create additional subdirectories if specified
        for subdir in args.subdirs:
            subdir_path = output_dir / subdir
            subdir_path.mkdir(exist_ok=True)
            logger.info(f"Created subdirectory: {subdir_path}")
        
        print("\nOutput directory setup complete!")
        print("All scripts will now use the 'output/' directory by default.")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error setting up output directory: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 