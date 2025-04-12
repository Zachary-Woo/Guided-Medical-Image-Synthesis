# Corrected Commands for Version 2

# --- Direct Script Calls (enhanced_controlnet_v2.py) ---

# Basic usage with default stain normalization (Macenko)
python version2/scripts/enhanced_controlnet_v2.py --condition_image data/pathmnist_samples/sample_0000.png --output_dir output/test_generation_direct

# Using Reinhard stain normalization with a reference image (ensure reference_images dir exists)
python version2/scripts/enhanced_controlnet_v2.py --condition_image data/pathmnist_samples/sample_0000.png --stain_norm reinhard --reference_image data/reference_images/he_reference.png --save_intermediates

# Generate multiple images with a specific seed
python version2/scripts/enhanced_controlnet_v2.py --condition_image data/pathmnist_samples/sample_0000.png --num_images 4 --seed 42

# Use LoRA (Base)
python version2/scripts/enhanced_controlnet_v2.py --condition_image data/pathmnist_samples/sample_0000.png --lora_model version2/models/lora/medical_lora.safetensors --lora_scale 0.8

# Use LoRA (Custom Trained)
python version2/scripts/enhanced_controlnet_v2.py --condition_image data/pathmnist_samples/sample_0000.png --lora_model version2/models/lora_histopathology/adapter_model.safetensors --lora_scale 0.8

# --- Using the Main Script (main.py) --- # Custom Trained LoRA Model: version2/models/lora_histopathology/adapter_model.safetensors

# Run enhanced generation through main.py (Corrected Command Name)
python version2/main.py generate --condition_image data/pathmnist_samples/sample_0000.png --stain_norm macenko

# Generate with LoRA via main.py (Added Example)
python version2/main.py generate --condition_image data/pathmnist_samples/sample_0000.png --lora_model version2/models/lora/medical_lora.safetensors --lora_scale 0.8

# Test stain normalization (Corrected Command Name, ensure reference_images dir exists)
python version2/main.py test --input_image data/pathmnist_samples/sample_0000.png --reference_image data/reference_images/he_reference.png

# Set up output directory (Correct Command Name)
python version2/main.py setup-output

# Prepare LoRA training for a local dataset (Corrected Command Name & Args)
# Assumes your images are in 'data/training_images'
python version2/main.py prepare --dataset local --local_dataset_path data/pathmnist_128.npz --output_dir output/processed_lora_data --stain_norm macenko


# --- Evaluation Commands ---

# Evaluate synthetic images (Corrected Command Name - NOTE: Script currently non-functional)
# python version2/main.py evaluate --synthetic_data_dir output/generated_images --task classification


# --- Additional Comparison Commands --- 

# 1. Compare Stain Normalization Methods:
# Goal: See the visual difference between Macenko, Reinhard (with a reference), and no normalization.
# Default (Macenko, no specific reference)
python version2/main.py generate --condition_image data/pathmnist_samples/sample_0000.png --output_dir output/compare_norm_macenko
# Reinhard (with specific reference - ensure ref image exists)
python version2/main.py generate --condition_image data/pathmnist_samples/sample_0000.png --stain_norm reinhard --reference_image data/reference_images/he_reference.png --output_dir output/compare_norm_reinhard_ref
# No Normalization
python version2/main.py generate --condition_image data/pathmnist_samples/sample_0000.png --stain_norm none --output_dir output/compare_norm_none

# 2. Compare LoRA Impact:
# Goal: Evaluate the effect of the LoRA adapter and its scaling factor.
# No LoRA (Baseline)
python version2/main.py generate --condition_image data/pathmnist_samples/sample_0000.png --output_dir output/compare_lora_off
# LoRA with default scale (e.g., 0.8)
python version2/main.py generate --condition_image data/pathmnist_samples/sample_0000.png --lora_model version2/models/lora/medical_lora.safetensors --lora_scale 0.8 --output_dir output/compare_lora_on_0_8
# LoRA with a different scale (e.g., lower)
python version2/main.py generate --condition_image data/pathmnist_samples/sample_0000.png --lora_model version2/models/lora/medical_lora.safetensors --lora_scale 0.5 --output_dir output/compare_lora_on_0_5

# 3. Compare Diffusion Parameters (Requires direct script call):
# Goal: See how step count and guidance scale affect quality and generation time.
# Default Steps/Guidance (check script defaults, e.g., 50 steps, 9.0 guidance)
python version2/scripts/enhanced_controlnet_v2.py --condition_image data/pathmnist_samples/sample_0000.png --output_dir output/compare_params_default
# Fewer Steps
python version2/scripts/enhanced_controlnet_v2.py --condition_image data/pathmnist_samples/sample_0000.png --steps 25 --output_dir output/compare_params_steps_25
# Higher Guidance
python version2/scripts/enhanced_controlnet_v2.py --condition_image data/pathmnist_samples/sample_0000.png --guidance_scale 12.0 --output_dir output/compare_params_guidance_12
# Lower Guidance
python version2/scripts/enhanced_controlnet_v2.py --condition_image data/pathmnist_samples/sample_0000.png --guidance_scale 5.0 --output_dir output/compare_params_guidance_5

# 4. Test Robustness with Different Inputs:
# Goal: Ensure the process works reasonably well on different conditioning images.
# Using sample 1 with your preferred settings (e.g., Macenko + LoRA)
python version2/main.py generate --condition_image data/pathmnist_samples/sample_0001.png --lora_model version2/models/lora/medical_lora.safetensors --lora_scale 0.8 --output_dir output/test_input_sample_1


# --- Tips for Analysis ---
# - Use distinct `--output_dir` names for each run.
# - Visual Comparison: Open generated images side-by-side.
# - Metadata: Check the `metadata.json` file in each output directory for parameters and generation time.
# - Keep Notes: Briefly jot down observations for each comparison.


