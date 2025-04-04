# Guided Medical Image Synthesis Using ControlNet for Data Augmentation

## Overview
This repository implements a pipeline for synthesizing 2D medical images using ControlNet, a diffusion-based generative model. The primary goal is to explore the potential of using synthetic images, guided by structural information (like segmentation maps or edge maps), to augment medical datasets. This implementation focuses on showing the end-to-end workflow and evaluating the feasibility of this approach for downstream tasks like classification or segmentation, particularly addressing data scarcity challenges in medical imaging.

## Project Organization

The project is organized into two versions:

- **Version 1**: Initial implementation with basic ControlNet image generation
- **Version 2**: Enhanced implementation with stain normalization, LoRA fine-tuning support, and advanced techniques

## Version 1 Features

The first implementation (`version1/`) focuses on basic image generation using ControlNet:

- Direct usage of pre-trained ControlNet models
- Basic edge detection for conditioning
- Standard medical prompts
- Simple parameter control

## Version 2 Features

The enhanced implementation (`version2/`) includes:

- Advanced histopathology stain normalization (Macenko and Reinhard)
- LoRA fine-tuning capability for domain adaptation
- Multi-technique edge extraction combining Canny, Sobel, and adaptive thresholding
- Medical-specific prompt engineering for different modalities
- Support for multiple schedulers and customizable parameters

## Project Structure
```
├── version1/              # Initial implementation
│   ├── simple_controlnet.py  # Main generation script for v1
│   ├── generate.py        # Basic text-to-image generation
│   ├── main.py            # CLI for the full pipeline
│   ├── test_pipeline.py   # Tests core functionality
│   └── README.md          # Version 1 documentation
│
├── version2/              # Enhanced implementation
│   ├── scripts/           # Enhanced scripts
│   │   ├── enhanced_controlnet_v2.py   # Advanced generation
│   │   └── prepare_lora_training.py    # LoRA dataset preparation
│   ├── utils/             # Utility modules
│   │   └── stain_normalization.py      # Stain normalization methods
│   ├── models/            # Directory for LoRA models
│   ├── data/              # Directory for processed datasets
│   ├── configs/           # Configuration files
│   └── README.md          # Version 2 documentation
│
├── data/                  # Shared data directory
│   └── pathmnist_samples  # Sample images for conditioning
│
├── output/                # Generated outputs
├── requirements.txt       # Project dependencies
└── README.md              # This file
```

## Quick Start

### Version 1 (Basic Generation)

```bash
# Install dependencies
pip install -r requirements.txt

# Basic Stable Diffusion generation
python version1/generate.py

# ControlNet generation
python version1/simple_controlnet.py --condition_image ./data/pathmnist_samples/sample_0000.png
```

### Version 2 (Enhanced Generation)

```bash
# Enhanced generation with stain normalization
python version2/scripts/enhanced_controlnet_v2.py --condition_image ./data/pathmnist_samples/sample_0000.png --stain_norm macenko

# Prepare dataset for LoRA training
python version2/scripts/prepare_lora_training.py --dataset local --local_dataset_path ./my_dataset --stain_norm macenko
```

Please refer to the README files in each version directory for detailed usage instructions.

## References
1. MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification
2. ControlNet for Conditional Image Generation
3. Hugging Face Diffusers Library Documentation

## License
This project is for educational and research purposes.