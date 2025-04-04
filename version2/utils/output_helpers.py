"""
Helper module for standardizing output directories across the project.
Provides functions to ensure all scripts use consistent output paths.
"""

import os
import logging
from pathlib import Path
import warnings

logger = logging.getLogger(__name__)

def standardize_output_path(path):
    """
    Standardize an output path to use 'output/' instead of 'outputs/'.
    Ensures consistent directory structure across the project.
    
    Args:
        path (str or Path): The path to standardize
        
    Returns:
        Path: Standardized path using 'output/' prefix
    """
    path_str = str(path)
    
    # If the path starts with 'outputs/', replace with 'output/'
    if path_str.startswith('outputs/'):
        new_path = 'output/' + path_str[8:]
        warnings.warn(
            f"Deprecated output path detected: '{path_str}'. "
            f"Using '{new_path}' instead. Please update your code.",
            DeprecationWarning, stacklevel=2
        )
        return Path(new_path)
    
    return Path(path)

def ensure_output_dir_exists(output_dir=None):
    """
    Ensure the output directory exists, creating it if necessary.
    
    Args:
        output_dir (str or Path, optional): Output directory to ensure exists
        
    Returns:
        Path: Path to the output directory
    """
    # Ensure the main output directory exists
    main_output = Path('output')
    main_output.mkdir(exist_ok=True)
    
    # If output_dir is specified, ensure it exists
    if output_dir:
        dir_path = standardize_output_path(output_dir)
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path
    
    return main_output

def get_next_output_dir(base_dir):
    """
    Create a sequentially numbered output directory.
    
    Args:
        base_dir (str or Path): Base output directory path
        
    Returns:
        Path: Next available numbered directory
    """
    # Standardize path
    base_dir = standardize_output_path(base_dir)
    
    base_path = Path(base_dir)
    base_parent = base_path.parent
    base_name = base_path.name
    
    # Ensure the parent directory exists
    base_parent.mkdir(parents=True, exist_ok=True)
    
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

def setup_output_logging(output_dir, log_filename="process.log"):
    """
    Set up logging for a script, with output to both console and file.
    
    Args:
        output_dir (str or Path): Directory to save log file
        log_filename (str): Name of the log file
        
    Returns:
        tuple: (output_dir_path, log_file_path)
    """
    # Standardize and create output dir
    output_dir = ensure_output_dir_exists(output_dir)
    
    # Set up logging file
    log_file = output_dir / log_filename
    
    # Check if the root logger already has handlers
    root_logger = logging.getLogger()
    
    # Add file handler if not already present
    has_file_handler = any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file) 
                         for h in root_logger.handlers)
    
    if not has_file_handler:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        root_logger.addHandler(file_handler)
        logger.info(f"Logging to {log_file}")
    
    return output_dir, log_file 