# Guided Medical Image Synthesis Using Diffusion Models

## Project Overview
This repository implements a progressive approach to medical image synthesis using diffusion models with increasingly sophisticated conditioning and control mechanisms. The project evolved through three versions, each addressing specific limitations and exploring different aspects of medical image generation:

1. **Version 1**: Basic implementation with pre-trained ControlNet models and simple edge detection for conditioning
2. **Version 2**: Enhanced framework with stain normalization, LoRA fine-tuning, and improved edge detection for histopathology images
3. **Version 3**: Comprehensive framework for anatomically accurate brain MRI synthesis with precise tumor placement control using segmentation masks

The work demonstrates the feasibility of generating anatomically accurate medical images with controllable pathological features, focusing particularly on brain MRI synthesis with precise tumor localization in the final version.

## Key Features Across Versions

### Version 1 (Initial Implementation)
- Basic ControlNet integration for medical image synthesis
- Canny edge detection for structure conditioning
- Simple parameter control for generation
- Tested on PathMNIST histopathology data

### Version 2 (Enhanced Histopathology Implementation)
- Advanced histopathology stain normalization (Macenko and Reinhard methods)
- LoRA fine-tuning for domain adaptation
- Multi-technique edge extraction (Canny, Sobel, adaptive thresholding)
- Medical-specific prompt engineering
- Support for multiple diffusion schedulers

### Version 3 (Anatomically Controlled Brain MRI Synthesis)
- Anatomically accurate generation of brain MRIs with tumors in specific locations
- SAM2 integration for creating and refining segmentation masks
- BraTS dataset support for training and evaluation
- Prompt-based control of tumor location and characteristics
- Modular architecture separating domain adaptation, structural guidance, and segmentation
- Visualization tools to compare synthetic images with real medical data

## Project Structure
```
.
├── README.md                # This file
├── data/                    # Shared data directory
├── output/                  # Generated outputs directory
├── requirements.txt         # Project dependencies
├── version1/                # Initial implementation
│   ├── scripts/             # Basic scripts for ControlNet generation
│   └── README.md            # Version 1 documentation
├── version2/                # Enhanced histopathology implementation
│   ├── scripts/             # Enhanced scripts with stain normalization
│   ├── models/              # LoRA models for histopathology
│   └── README.md            # Version 2 documentation
└── version3/                # Brain MRI synthesis implementation
    ├── main.py              # Main CLI entry point
    ├── scripts/             # Scripts for MRI generation and training
    ├── models/              # LoRA and ControlNet models for MRI
    └── README.md            # Version 3 documentation
```

## Version Evolution and Approach
This project represents a significant evolution in approach to medical image synthesis:

1. The initial version focused on establishing a basic pipeline using ControlNet with Canny edge detection for conditioning, applied to histopathology images.

2. The second version enhanced the histopathology implementation with specialized techniques including stain normalization and improved edge detection to better capture tissue structures.

3. The third version represented a strategic pivot, motivated by fundamental limitations in achieving precise anatomical control with edge-based conditioning. This version focused on brain MRI synthesis with tumor placement control using segmentation-based conditioning instead of edge detection, integrating SAM2 to generate and refine tumor masks.

This progression demonstrates the challenges and solutions in developing effective medical image synthesis systems with increasing levels of anatomical control.

## Quick Start
Each version has its own installation and usage instructions - please refer to the respective README files for detailed guidance.

### General Setup
```bash
# Clone the repository
git clone https://github.com/Zachary-Woo/Guided-Medical-Image-Synthesis.git
cd Guided-Medical-Image-Synthesis

# Install base requirements
pip install -r requirements.txt
```

### Version 1 (Basic Generation)
```bash
# Basic ControlNet generation
python version1/simple_controlnet.py --condition_image ./data/pathmnist_samples/sample_0000.png
```

### Version 2 (Enhanced Histopathology Generation)
```bash
# Enhanced generation with stain normalization
python version2/scripts/enhanced_controlnet_v2.py --condition_image ./data/pathmnist_samples/sample_0000.png --stain_norm macenko
```

### Version 3 (Brain MRI with Tumor Generation)
```bash
# Generate brain MRI with tumor in specific location
python version3/main.py generate --prompt "T1 weighted axial brain MRI with tumor in left temporal lobe" --create_mask

# Generate and visualize compared with real data
python version3/main.py generate --prompt "T1 weighted axial brain MRI with tumor in left temporal lobe" \
    --output_dir output/version3/generated --visualize \
    --brats_dir data/BraTS2021_Training_Data/BraTS2021_00000 --compare_modality t1
```

## Hardware Requirements
- Python 3.8+
- CUDA-compatible GPU (for Version 3, at least 24GB VRAM is recommended)

## References
1. BraTS Dataset: [Brain Tumor Segmentation Challenge](https://www.med.upenn.edu/cbica/brats2023/)
2. PathMNIST: [MedMNIST v2](https://medmnist.com/)
3. SAM2: [Segment Anything 2 by Meta](https://github.com/facebookresearch/segment-anything-2)
4. Stable Diffusion: [Stability AI](https://stability.ai/stable-diffusion)
5. ControlNet: [Zhang et al.](https://github.com/lllyasviel/ControlNet)
6. Diffusers Library: [HuggingFace Diffusers](https://huggingface.co/docs/diffusers/)