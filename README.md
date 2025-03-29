# Guided Medical Image Synthesis Using ControlNet for Data Augmentation

## Overview
This repository contains the implementation of a diffusion-based generative model for creating synthetic 2D medical images using ControlNet. The primary goal is to address data scarcity in medical imaging by generating high-quality synthetic images that can augment existing datasets. Specifically, the project aims to produce synthetic data for rare or underrepresented classes to improve performance in tasks such as tumor segmentation and classification.

## Project Objectives
1. **Generate Realistic Medical Images**  
   Use ControlNet to synthesize images conditioned on textual prompts and segmentation masks.  
2. **Evaluate Downstream Impact**  
   Assess how including synthetic data influences medical image analysis tasks (e.g., tumor segmentation).  
3. **Ensure Anatomical Plausibility**  
   Validate generated images using metrics like Dice or IoU to ensure they are clinically meaningful.

## Problem Definition
Medical datasets often have imbalanced classes, making it difficult to train robust deep learning models. Annotating medical images is expensive and time-consuming, especially for rare conditions. This project seeks to use synthetic image generation to address these challenges, providing an additional stream of data for training and evaluation.

## Approach
1. **Model Selection**  
   - Adapting ControlNet, a diffusion-based architecture, originally designed for natural images.  
   - Fine-tuning for medical imaging tasks by conditioning on text prompts and segmentation masks.  
   - Employing a pretrained Stable Diffusion backbone (v1.5 or v2.x) to reduce training time and resource consumption.
   - Optionally using lightweight fine-tuning methods like LoRA (Low-Rank Adaptation) if domain mismatch is severe.

2. **Datasets**  
   - **Breast Cancer Immunohistochemical (BCI)**: Histopathology images for tumor detection.  
   - **MedMNIST v2**: Collection of multiple 2D medical imaging datasets (e.g., PathMNIST).  
   - **BraTS** (optional 3D extension): Brain MRI scans, used here in 2D slice format if included.  
   - **PanNuke**: Histopathology dataset with nuclei instance segmentation across multiple tissue types.  
   - **CAMELYON16/17**: Lymph node metastasis detection with pixel-level tumor annotations.  
   - **ACDC**: Cardiac MRI dataset with left/right ventricle and myocardium masks.

3. **Training Strategy**  
   - Preprocess images and segmentation masks into a format suitable for ControlNet conditioning (e.g., binary or edge-like mask inputs).  
   - Freeze the main Stable Diffusion UNet weights and train the ControlNet layers, minimizing compute cost and risk of overfitting.  
   - If needed, apply partial or LoRA-based fine-tuning of the backbone to capture domain-specific features (e.g., histology textures).  
   - Use mixed precision (FP16/BF16), gradient checkpointing, and memory-efficient attention (xFormers) to fit within 24 GB VRAM.

4. **Evaluation**  
   - **Image Quality**: Use FID and SSIM to measure how closely synthetic images match real images.  
   - **Anatomical Plausibility**: Compare generated masks against ground-truth annotations using Dice or IoU.  
   - **Downstream Utility**: Train a segmentation or classification model (e.g., U-Net, ResNet) on real vs. synthetic-plus-real data to measure performance gains.

## Experiments
1. **Baseline Generation**  
   - Generate images from text prompts only.  
   - Measure output quality (FID, SSIM).

2. **Mask-Conditioned Generation**  
   - Provide segmentation masks as control inputs.  
   - Evaluate how well structural information is preserved (Dice, IoU).

3. **Data Augmentation Impact**  
   - Retrain segmentation/classification models using synthetic images in the training set.  
   - Compare performance metrics (accuracy, F1, Dice) to baseline results.

4. **Ablation Studies**  
   - Vary ControlNet fine-tuning depth (shallow vs. deep).  
   - Compare text-only conditioning, mask-only conditioning, and combined approaches.  
   - Investigate the effect of LoRA fine-tuning on base diffusion weights.

## Potential Challenges
- **Maintaining Clinical Relevance**: Ensuring generated images remain anatomically realistic.  
- **Data Scarcity**: Limited labeled data for fine-tuning could lead to overfitting, even with partial training techniques.  
- **Evaluation Metrics**: Standardized measures for synthetic medical image realism are still evolving, making it hard to compare to existing methods.

## Recommended Workflow and Tools
1. **Hugging Face Diffusers**  
   - Offers reference scripts for ControlNet training.  
   - Provides built-in support for mixed precision, xFormers, and pretrained Stable Diffusion models.

2. **MONAI**  
   - For downstream segmentation or classification experiments.  
   - Medical imaging–focused transforms and metrics simplify evaluation.

3. **Experiment Tracking**  
   - Tools like Weights & Biases or TensorBoard to log training losses, sample outputs, and model checkpoints for reproducibility.

## Project Structure (Tentative)
├── data  
│   ├── bci_dataset  
│   ├── medmnist  
│   ├── pannuke (optional)
│   ├── camelyon (optional)
│   ├── acdc (optional)
│   └── brats (optional)  
├── src  
│   ├── preprocessing  
│   ├── controlnet_training  
│   ├── evaluation  
│   └── utils  
├── experiments  
│   ├── baseline_generation  
│   ├── mask_conditioned_generation  
│   └── data_augmentation  
└── README.md  


## References
1. Breast Cancer Immunohistochemical Image Generation: A Benchmark Dataset and Challenge Review  
2. Diffusion-Based Data Augmentation for Nuclei Image Segmentation  
3. Generative Models for Medical Imaging (MONAI)  
4. ControlNet for Conditional Image Generation  
5. BCI Dataset  
6. CAMELYON16/17, PanNuke, and ACDC datasets (publicly available)  
7. Stable Diffusion and ControlNet official repositories and checkpoints

## License
This project is for educational and research purposes. Please consult the repository’s LICENSE file for more details.