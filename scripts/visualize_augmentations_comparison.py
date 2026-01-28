#!/usr/bin/env python3
"""
Proper comparison visualization: SAME images across different augmentations.
Each column = same base image
Each row = different augmentation type
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import transforms
from datasets import load_from_disk
import os

from src.utils.augmentations import RandomPixelRemoval, GaussianNoiseAugmentation
from src.utils.imagenet_data import IMAGENET_MEAN, IMAGENET_STD, GaussianBlur

def denormalize(tensor):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor.cpu() * std + mean).clamp(0, 1)

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

print("Loading ImageNet dataset...")
data_dir = '/leonardo_scratch/fast/CNHPC_1905882/clxai/data/imagenet-1k'
hf_dataset = load_from_disk(os.path.join(data_dir, 'train'))

# Select 6 diverse images (different indices for variety)
sample_indices = [100, 5000, 20000, 50000, 100000, 200000]

# Base transform (just resize and crop, no augmentation)
base_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# Different augmentation pipelines
def get_none_transform():
    """Standard contrastive augmentation (no custom aug)"""
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.RandomApply([GaussianBlur(sigma=(0.1, 2.0))], p=0.5),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def get_pixel50_transform():
    """Contrastive + pixel removal (50% probability)"""
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.RandomApply([GaussianBlur(sigma=(0.1, 2.0))], p=0.5),
        RandomPixelRemoval(removal_fraction=0.1, probability=0.5),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def get_noise_transform():
    """Contrastive + Gaussian noise (50% probability)"""
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.RandomApply([GaussianBlur(sigma=(0.1, 2.0))], p=0.5),
        GaussianNoiseAugmentation(probability=0.5, std=0.05),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

# ============================================================
# FIGURE 1: Same image, different augmentations (proper comparison)
# ============================================================
print("\nCreating Figure 1: Same images across augmentation types...")

fig, axes = plt.subplots(4, 6, figsize=(18, 12))
row_labels = ['Original\n(no aug)', 'Standard\nContrastive', 'pixel50\n(+10% removal)', 'noise\n(+Gaussian)']

for col, idx in enumerate(sample_indices):
    img_pil = hf_dataset[idx]['image'].convert('RGB')
    
    # Set same seed for each column to get similar random crops
    seed = 42 + col
    
    # Row 0: Original (just resize/crop)
    img_orig = base_transform(img_pil)
    axes[0, col].imshow(img_orig.permute(1, 2, 0).numpy())
    axes[0, col].axis('off')
    if col == 0:
        axes[0, col].set_ylabel(row_labels[0], fontsize=12, rotation=0, ha='right', va='center')
    
    # Row 1: Standard contrastive (no custom aug)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    img_none = get_none_transform()(img_pil)
    axes[1, col].imshow(denormalize(img_none).permute(1, 2, 0).numpy())
    axes[1, col].axis('off')
    if col == 0:
        axes[1, col].set_ylabel(row_labels[1], fontsize=12, rotation=0, ha='right', va='center')
    
    # Row 2: pixel50
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    img_pixel = get_pixel50_transform()(img_pil)
    axes[2, col].imshow(denormalize(img_pixel).permute(1, 2, 0).numpy())
    axes[2, col].axis('off')
    if col == 0:
        axes[2, col].set_ylabel(row_labels[2], fontsize=12, rotation=0, ha='right', va='center')
    
    # Row 3: noise
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    img_noise = get_noise_transform()(img_pil)
    axes[3, col].imshow(denormalize(img_noise).permute(1, 2, 0).numpy())
    axes[3, col].axis('off')
    if col == 0:
        axes[3, col].set_ylabel(row_labels[3], fontsize=12, rotation=0, ha='right', va='center')

plt.suptitle('Augmentation Comparison: Same Image Across Different Augmentation Types\n(Column = same image, Row = different augmentation)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('results/augmentation_comparison_same_image.png', dpi=150, bbox_inches='tight')
print("Saved: results/augmentation_comparison_same_image.png")

# ============================================================
# FIGURE 2: Pixel removal - with and without (always applied)
# ============================================================
print("\nCreating Figure 2: Pixel removal effect (always applied vs never)...")

# Transforms with pixel removal ALWAYS vs NEVER applied
def pixel_always():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        RandomPixelRemoval(removal_fraction=0.1, probability=1.0),  # ALWAYS
    ])

def pixel_never():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

fig2, axes2 = plt.subplots(2, 6, figsize=(18, 6))

for col, idx in enumerate(sample_indices):
    img_pil = hf_dataset[idx]['image'].convert('RGB')
    
    # Row 0: No pixel removal
    img_clean = pixel_never()(img_pil)
    axes2[0, col].imshow(img_clean.permute(1, 2, 0).numpy())
    axes2[0, col].axis('off')
    if col == 0:
        axes2[0, col].set_ylabel('Original', fontsize=12)
    
    # Row 1: With pixel removal (10%, always applied)
    torch.manual_seed(42 + col)
    np.random.seed(42 + col)
    img_removed = pixel_always()(img_pil)
    axes2[1, col].imshow(img_removed.permute(1, 2, 0).numpy())
    axes2[1, col].axis('off')
    if col == 0:
        axes2[1, col].set_ylabel('pixel50\n(10% black)', fontsize=12)

plt.suptitle('Pixel Removal Effect: 10% Scattered Pixels Set to Black\n(Same images, top=original, bottom=with pixel removal)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('results/pixel_removal_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: results/pixel_removal_comparison.png")

# ============================================================
# FIGURE 3: Noise effect - with and without (always applied)
# ============================================================
print("\nCreating Figure 3: Noise effect (always applied vs never)...")

def noise_always():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        GaussianNoiseAugmentation(probability=1.0, std=0.05),  # ALWAYS
    ])

fig3, axes3 = plt.subplots(2, 6, figsize=(18, 6))

for col, idx in enumerate(sample_indices):
    img_pil = hf_dataset[idx]['image'].convert('RGB')
    
    # Row 0: No noise
    img_clean = pixel_never()(img_pil)
    axes3[0, col].imshow(img_clean.permute(1, 2, 0).numpy())
    axes3[0, col].axis('off')
    if col == 0:
        axes3[0, col].set_ylabel('Original', fontsize=12)
    
    # Row 1: With noise (always applied)
    torch.manual_seed(42 + col)
    img_noisy = noise_always()(img_pil)
    axes3[1, col].imshow(img_noisy.permute(1, 2, 0).numpy())
    axes3[1, col].axis('off')
    if col == 0:
        axes3[1, col].set_ylabel('noise\n(std=0.05)', fontsize=12)

plt.suptitle('Gaussian Noise Effect: std=0.05 Added to Image\n(Same images, top=original, bottom=with noise)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('results/noise_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: results/noise_comparison.png")

# ============================================================
# FIGURE 4: Show probability effect (multiple samples, same image)
# ============================================================
print("\nCreating Figure 4: 50% probability effect (10 samples of same image)...")

fig4, axes4 = plt.subplots(3, 10, figsize=(20, 6))
row_labels = ['Original', 'pixel50 (50% prob)', 'noise (50% prob)']

# Use just one image, show 10 random augmentations
img_pil = hf_dataset[100]['image'].convert('RGB')

for col in range(10):
    # Row 0: Original (same for all)
    if col == 0:
        img_orig = base_transform(img_pil)
        axes4[0, col].imshow(img_orig.permute(1, 2, 0).numpy())
    else:
        axes4[0, col].imshow(img_orig.permute(1, 2, 0).numpy())
    axes4[0, col].axis('off')
    if col == 0:
        axes4[0, col].set_ylabel(row_labels[0], fontsize=10)
    
    # Row 1: pixel50 with random seed
    torch.manual_seed(col * 100)
    np.random.seed(col * 100)
    random.seed(col * 100)
    img_pixel = get_pixel50_transform()(img_pil)
    axes4[1, col].imshow(denormalize(img_pixel).permute(1, 2, 0).numpy())
    axes4[1, col].axis('off')
    if col == 0:
        axes4[1, col].set_ylabel(row_labels[1], fontsize=10)
    
    # Row 2: noise with random seed
    torch.manual_seed(col * 100)
    np.random.seed(col * 100)
    random.seed(col * 100)
    img_noise = get_noise_transform()(img_pil)
    axes4[2, col].imshow(denormalize(img_noise).permute(1, 2, 0).numpy())
    axes4[2, col].axis('off')
    if col == 0:
        axes4[2, col].set_ylabel(row_labels[2], fontsize=10)

plt.suptitle('50% Probability Effect: Same Image Augmented 10 Times\n(~5 should have custom aug applied, ~5 should not)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('results/probability_effect.png', dpi=150, bbox_inches='tight')
print("Saved: results/probability_effect.png")

plt.close('all')
print("\n=== All visualizations created! ===")
print("Check results/ folder for:")
print("  - augmentation_comparison_same_image.png (main comparison)")
print("  - pixel_removal_comparison.png (pixel effect only)")
print("  - noise_comparison.png (noise effect only)")
print("  - probability_effect.png (50% probability demonstration)")
