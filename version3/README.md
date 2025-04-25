# Medical Image Synthesis Version 3 - Anatomically Accurate Brain MRI Generation

This directory contains the implementation of a specialized medical image synthesis system focused on generating anatomically accurate brain MRI images with tumors in specific locations.

## Key Features

1. **Anatomically Accurate Generation**: Generate brain MRIs with tumors in specific anatomical locations (e.g., "left temporal lobe", "right frontal region") using a combination of LoRA and ControlNet.

2. **SAM2 Integration**: Leverage Meta's Segment Anything 2 model for creating and refining segmentation masks that guide the generation process. Now includes fine-tuning capabilities to improve segmentation accuracy on medical images.

3. **BraTS Dataset Support**: Process and prepare data from the BraTS (Brain Tumor Segmentation) dataset for training custom models.

4. **Modular Architecture**: Separate the concerns of domain adaptation (LoRA), structural guidance (ControlNet), and segmentation accuracy (SAM2) for maximum flexibility.

5. **Prompt-Based Control**: Simple text prompts like "T1 weighted axial brain MRI with tumor in left temporal lobe" drive the generation process.

## System Architecture

The system follows a multi-stage approach:

1. **Data Preparation**: Process BraTS dataset to extract paired MRI slices and tumor masks, generating appropriate descriptive prompts for each slice.

2. **Model Training**: 
   - Train a LoRA adapter that teaches the diffusion model the style and content of brain MRI images
   - Train a ControlNet model that learns to generate MRIs conditioned on tumor segmentation masks
   - Fine-tune SAM2 on BraTS data to improve segmentation mask accuracy for tumor regions

3. **Inference Pipeline**:
   - Parse the user's text prompt to extract tumor location and characteristics
   - Generate or provide a segmentation mask corresponding to the requested tumor
   - Apply LoRA for medical domain adaptation and ControlNet for structural accuracy
   - Generate the final MRI image that maintains both domain fidelity and anatomical accuracy

## Directory Structure

```
version3/
├── main.py                   # Main entry point with CLI commands
├── README.md                 # This file
├── scripts/
│   ├── generate_mri.py       # MRI generation script (inference)
│   ├── prepare_brats.py      # BraTS dataset preparation script
│   ├── segment_mri.py        # SAM2 segmentation script
│   ├── train_lora.py         # LoRA training script
│   ├── train_controlnet.py   # ControlNet training script
│   └── train_sam2.py         # SAM2 fine-tuning script
├── utils/                    # Utility functions and helpers
├── models/                   # Directory for trained model weights
│   ├── mri_lora/             # LoRA weights for brain MRI adaptation
│   ├── segmentation_controlnet/ # ControlNet weights for mask conditioning
│   └── sam2_finetuned/       # Fine-tuned SAM2 model for tumor segmentation
└── configs/                  # Configuration files for training and generation
```

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU with at least 24GB VRAM (e.g., RTX 4090, RTX 3090, A5000)
- BraTS dataset (for training custom models) or sample MRI images

### Installation

```bash
# Create conda environment (recommended)
conda create -n medical-gen python=3.9
conda activate medical-gen

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install required packages
pip install -r requirements.txt

# Test CUDA availability
python version3/main.py test-cuda
```

## Usage Guide

### 1. Generating Brain MRIs with Tumors

To generate an MRI with a tumor in a specific location:

```bash
python version3/main.py generate --prompt "T1 weighted axial brain MRI with tumor in left temporal lobe" --create_mask
```

You can specify the axial slice level to precisely control what part of the brain is shown:

```bash
# Generate a slice showing the ventricles
python version3/main.py generate --prompt "T1 weighted brain MRI with tumor" --create_mask --slice_level ventricles

# Generate a slice through the cerebellum
python version3/main.py generate --prompt "T1 weighted brain MRI with tumor" --create_mask --slice_level cerebellum

# Available slice levels: superior, mid-axial, inferior, ventricles, basal-ganglia, cerebellum
```

### Generate and Visualize in a Single Command

You can automatically compare your generated MRIs with real BraTS data as part of the generation process:

```bash
# Generate and immediately visualize compared to a real BraTS scan
python version3/main.py generate --prompt "T1 weighted axial brain MRI with tumor in left temporal lobe" \
    --output_dir output/version3/generated --visualize \
    --brats_dir data/BraTS2021_Training_Data/BraTS2021_00000 --compare_modality t1

# Generate with auto-created mask and specify slice level
python version3/main.py generate --prompt "T2 weighted brain MRI with tumor" --create_mask \
    --slice_level ventricles --visualize \
    --brats_dir data/BraTS2021_Training_Data/BraTS2021_00000 --compare_modality t2

# Display visualization after generation
python version3/main.py generate --prompt "MRI with tumor in right frontal lobe" --create_mask \
    --visualize --brats_dir data/BraTS2021_Training_Data/BraTS2021_00000 --show_visualization
```

This will:
1. Parse the prompt to identify the tumor location
2. Create a segmentation mask appropriate for that location (when using --create_mask)
3. Use the mask with ControlNet to guide the image generation
4. Apply domain adaptation via LoRA (if available)
5. Save the result to the specified output directory
6. Automatically generate a comparison with real BraTS data (with --visualize flag)

To use a custom mask:

```bash
python version3/main.py generate --prompt "T1 weighted axial brain MRI with large tumor" --mask path/to/mask.png
```

### 2. Segmenting Existing MRIs with SAM2

To segment a tumor in an existing MRI using SAM2:

```bash
# Using text guidance (SAM2 feature)
python version3/main.py segment --input_image path/to/mri.png --prompt_type text --prompt "tumor in brain"

# Using point guidance (click in the center of the tumor)
python version3/main.py segment --input_image path/to/mri.png --prompt_type point --point "256,256"

# Using box guidance (bounding box around tumor)
python version3/main.py segment --input_image path/to/mri.png --prompt_type box --box "200,200,300,300"
```

### 3. Fine-tuning SAM2 for Improved Tumor Segmentation

To fine-tune SAM2 on BraTS dataset to improve tumor segmentation accuracy:

```bash
python version3/main.py train-sam2 --brats_path data/BraTS2021_Training_Data --output_dir version3/models/sam2_finetuned --augmentation
```

This will:
1. Load the SAM2 model (from local files or HuggingFace hub)
2. Extract tumor regions from BraTS MRI volumes
3. Fine-tune the model using point prompts and tumor masks
4. Save the fine-tuned model for use in the segmentation and generation pipelines

For fine-tuning only the mask decoder and prompt encoder (memory efficient):

```bash
python version3/main.py train-sam2 --brats_path data/BraTS2021_Training_Data --max_train_steps 3000
```

For fine-tuning the entire model including the image encoder (requires more GPU memory):

```bash
python version3/main.py train-sam2 --brats_path data/BraTS2021_Training_Data --train_image_encoder
```

### 4. Preparing BraTS Dataset for Training

To prepare the BraTS dataset for training:

```bash
python version3/main.py prepare --brats_path data/BraTS2021_Training_Data --modality t1 --sample_count 1000 --include_healthy
```

This will:
1. Extract 1000 axial slices from the BraTS dataset
2. Create binary tumor masks
3. Generate text prompts describing each slice
4. Prepare metadata files for LoRA and ControlNet training

### 5. Training a LoRA Adapter
To train a LoRA adapter for brain MRI domain adaptation:

```bash
python version3/main.py train-lora --data_path data/processed_brats --output_dir version3/models/mri_lora
```

### 6. Training a ControlNet Model
To train a ControlNet model for mask-conditioned generation:

```bash
python version3/main.py train-controlnet --data_path data/processed_brats --output_dir version3/models/segmentation_controlnet
```

## Examples

### Example 1: Complete Pipeline with Fine-tuned SAM2
```bash
# 1. First fine-tune SAM2 on BraTS data
python version3/main.py train-sam2 --brats_path data/BraTS2021_Training_Data --output_dir version3/models/sam2_finetuned

# 2. Use it to generate a mask for an existing MRI image
python version3/main.py segment --input_image sample_mri.png --prompt_type text --prompt "tumor" --sam_model version3/models/sam2_finetuned

# 3. Generate a new MRI with the same tumor structure
python version3/main.py generate --prompt "T1 weighted axial brain MRI with tumor" --mask output/version3/segments/segment_*/mask.png
```

### Example 2: MRI with Left Temporal Tumor
```bash
python version3/main.py generate --prompt "T1 weighted axial brain MRI with tumor in left temporal lobe" --create_mask --seed 42
```
### Example 3: MRI with Right Frontal Tumor
```bash
python version3/main.py generate --prompt "T1 weighted axial brain MRI with large tumor in right frontal lobe" --create_mask --seed 123
```

### Example 4: Visualizing and Comparing with Real MRI Data
```bash
# First, generate an MRI
python version3/main.py generate --prompt "T1 weighted axial brain MRI with tumor in left temporal lobe" --create_mask --slice_level ventricles

# Then visualize and compare with real BraTS data
python version3/main.py visualize --generated_dir output/version3/generated/mri_[timestamp] --brats_dir data/BraTS2021_Training_Data/BraTS2021_00000 --modality t1

# Compare to different modalities (T1, T2, FLAIR, T1ce)
python version3/main.py visualize --generated_dir output/version3/generated/mri_[timestamp] --brats_dir data/BraTS2021_Training_Data/BraTS2021_00000 --modality t2 --show
```

The visualization creates a side-by-side comparison of:
1. The conditioning mask used to guide generation
2. The generated MRI image
3. A real MRI slice from the BraTS dataset that matches the same slice level

This allows for easy visual comparison between generated and real medical images.

## Evaluation

To evaluate the anatomical accuracy of the generated images:

1. **Visual Assessment**: Compare the tumor location in the generated image with the requested location.

2. **Mask Comparison**: Run SAM2 on the generated image to extract the tumor mask, then compute Dice coefficient with the original conditioning mask.

3. **Expert Validation**: Have a radiologist or medical expert evaluate the realism and anatomical accuracy of the generated images.

## Common Issues and Solutions

- **Out of Memory Errors**: Reduce batch sizes or use a lower resolution like 384x384 instead of 512x512. For SAM2 fine-tuning, avoid training the image encoder if memory limited.
- **SAM2 Installation Issues**: The system supports both local SAM2 repository and HuggingFace transformers implementation. If problems arise, try the transformers approach which has fewer dependencies.
- **Tumor Localization**: If the tumor doesn't appear in the correct location, try increasing the ControlNet conditioning scale (e.g., `--controlnet_conditioning_scale 1.5`).
- **Unrealistic Appearance**: Fine-tune the LoRA weights on a domain-specific dataset for better results.
- **Poor Segmentation Quality**: If SAM2 struggles with tumor boundaries, fine-tune it on BraTS data using the train-sam2 command.

## References

1. BraTS Dataset: [Brain Tumor Segmentation Challenge](https://www.med.upenn.edu/cbica/brats2023/)
2. SAM2: [Segment Anything 2 by Meta](https://github.com/facebookresearch/segment-anything-2)
3. Stable Diffusion: [Stability AI](https://stability.ai/stable-diffusion)
4. Diffusers Library: [HuggingFace Diffusers](https://huggingface.co/docs/diffusers/) 
