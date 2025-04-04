# Medical Image Synthesis Version 2

This directory contains enhanced components for more advanced medical image synthesis, building on the foundation of the original implementation.

## Key Enhancements

1. **Stain Normalization**: Advanced histopathology stain normalization with Macenko and Reinhard methods.
2. **LoRA Fine-tuning Support**: Tools for preparing datasets and fine-tuning domain-specific LoRA adapters.
3. **Improved Edge Detection**: Multi-technique edge extraction combining Canny, Sobel, and adaptive thresholding.
4. **Medical-specific Prompt Engineering**: Specialized prompting for different medical imaging modalities.
5. **Advanced Pipeline Architecture**: Support for multiple schedulers and customizable generation parameters.

## Directory Structure

```
version2/
├── scripts/
│   ├── enhanced_controlnet_v2.py      # Main image generation script with all enhancements
│   └── prepare_lora_training.py       # Script to prepare datasets for LoRA fine-tuning
├── utils/
│   └── stain_normalization.py         # Specialized stain normalization utilities
├── models/                            # Directory for storing LoRA models (created during training)
├── data/                              # Directory for processed datasets (created during dataset preparation)
├── configs/                           # Configuration files for LoRA training
└── README.md                          # This file
```

## Output Organization

All scripts place outputs in the `output/` directory at the project root with the following pattern:

```
output/
├── enhanced_controlnet_v2_1/          # First run of the generation script
│   ├── generated.png                  # Generated image
│   ├── normalized_input.png           # Normalized input image
│   ├── original_input.png             # Original input image 
│   ├── canny_edge.png                 # Edge map used for ControlNet
│   ├── metadata.json                  # Generation parameters
│   └── generate.log                   # Detailed log file
├── enhanced_controlnet_v2_2/          # Second run (auto-incremented)
├── stain_normalization_test_1/        # First run of stain normalization test
│   ├── normalized_macenko.png         # Normalized image
│   ├── visualization_macenko.png      # Side-by-side visualization
│   ├── normalization_comparison.png   # Comparison plot
│   └── test.log                       # Test log file
├── evaluation_v2_1/                   # First run of evaluation script
│   ├── evaluate.log                   # Evaluation log file
│   ├── comparison_training_loss.png   # Training metrics
│   ├── downstream_model_real_only.pth # Trained model file
│   └── metrics.json                   # Performance metrics
└── processed_data/                    # Processed datasets
```

Each script automatically creates a sequentially numbered output directory to prevent overwriting previous results.

## Installation

Ensure all requirements are installed:

```bash
pip install -r ../requirements.txt
# Optional dependencies for LoRA training
pip install diffusers[training] transformers accelerate datasets
```

## Usage Guide

### 1. Enhanced Image Generation

Generate histopathology images with advanced edge detection and stain normalization:

```bash
python version2/scripts/enhanced_controlnet_v2.py --condition_image ./data/pathmnist_samples/sample_0000.png --stain_norm macenko --controlnet_conditioning_scale 1.0
```

#### Key Parameters

- `--stain_norm`: Select normalization method (`macenko`, `reinhard`, or `none`)
- `--scheduler`: Diffusion scheduler (`unipc`, `dpm`, `ddim`, `euler`) 
- `--reference_image`: Optional reference image for stain normalization
- `--save_intermediates`: Save intermediate processing steps for visualization
- `--lora_model`: Path to a fine-tuned LoRA adapter (if available)

### 2. Preparing Data for LoRA Fine-tuning

Process a histopathology dataset for domain-specific LoRA fine-tuning:

```bash
python version2/scripts/prepare_lora_training.py --dataset kather_texture --stain_norm macenko
```

For local datasets:

```bash
python version2/scripts/prepare_lora_training.py --dataset local --local_dataset_path ./my_histopathology_dataset --stain_norm macenko
```

### 3. Using a Reference Image for Stain Normalization

For best results, provide a well-stained histopathology image as reference:

```bash
python version2/scripts/enhanced_controlnet_v2.py --condition_image ./data/pathmnist_samples/sample_0000.png --reference_image ./path/to/reference_he_image.png --stain_norm macenko
```

### 4. Generating Multiple Images with Different Seeds

```bash
python version2/scripts/enhanced_controlnet_v2.py --condition_image ./data/pathmnist_samples/sample_0000.png --num_images 4 --seed 42
```

### 5. Testing Stain Normalization

Test different stain normalization methods on an input image:

```bash
python version2/scripts/test_stain_normalization.py --input_image ./data/pathmnist_samples/sample_0000.png --method macenko --save_visualization
```

### 6. Evaluating with Generated Images

Evaluate downstream task performance using generated images:

```bash
python version2/scripts/evaluate_v2.py --synthetic_data_dir ./output/enhanced_controlnet_v2_1 --task classification
```

### 7. Setting Up the Output Directory

Ensure the output directory exists with any needed subdirectories:

```bash
python version2/main.py setup-output --subdirs "processed_data" "evaluation_results"
```

## Advanced Features

### Fine-tuning with LoRA

After preparing your dataset with `prepare_lora_training.py`, you can fine-tune a LoRA adapter using the generated configuration:

```bash
# This requires additional setup with Hugging Face Accelerate
accelerate launch --config_file=version2/configs/accelerate_config.yaml \
  path/to/diffusers/examples/text_to_image/train_text_to_image_lora.py \
  --config_file version2/configs/lora_config.yaml
```

The resulting LoRA weights can then be used for generation:

```bash
python version2/scripts/enhanced_controlnet_v2.py --condition_image ./data/pathmnist_samples/sample_0000.png --lora_model version2/models/lora_histopathology/pytorch_lora_weights.safetensors --lora_scale 0.8
```

### Expert Validation

For expert validation of your generated images:
1. Generate a batch of images using different parameters
2. Create a simple side-by-side comparison with original images
3. Have an expert rate the realism of the generated images
4. Use this feedback to refine your generation parameters

## Troubleshooting

- **Memory Issues**: If you encounter CUDA out of memory errors, try reducing `--output_size` to 384 or enabling model offloading.
- **Stain Normalization Failures**: If stain normalization fails, ensure your reference image is a well-stained H&E image and try using the `reinhard` method which is more robust for unusual staining patterns.
- **LoRA Training Issues**: Ensure you have the latest version of diffusers and that your dataset has properly formatted metadata.jsonl files. 