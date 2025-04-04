# Medical Image Synthesis Version 1

This directory contains the original implementation of the medical image synthesis pipeline using ControlNet.

## Key Files

1. **simple_controlnet.py**: Simplified ControlNet implementation for direct image generation.
2. **generate.py**: Basic image generation script using Stable Diffusion.
3. **main.py**: Original main pipeline script with command-line interface.
4. **test_pipeline.py**: Script for testing the pipeline functionality.
5. **extract_medmnist_samples.py**: Utility to extract sample images from MedMNIST datasets.
6. **evaluate.py**: Evaluation metrics for generated images.
7. **requirements.txt**: Required packages for the implementation.

## Usage

### Basic Generation (Stable Diffusion)

```bash
python version1/generate.py
```

### ControlNet Generation

```bash
python version1/simple_controlnet.py --condition_image ./data/pathmnist_samples/sample_0000.png
```

### Full Pipeline

```bash
python version1/main.py generate --input_dir ./data/pathmnist_samples --output_dir ./output/pipeline
```

### Using from Project Root

After moving this code to the version1 folder, make sure to:

1. Run scripts from the project root (not from within the version1 directory)
2. Use paths that include the version1 prefix, e.g.:
   - Config files: `version1/configs/medmnist_canny_demo.yaml`
   - Output directories: `output/...` (not `outputs/...`)

## Note on Directory Structure

This version uses a single `output` directory (not `outputs`) for consistency with version2.
The helper functions in version2 will automatically convert any `outputs/` paths to `output/`.

See version2 for the enhanced implementation with stain normalization, LoRA support, and other advanced features. 