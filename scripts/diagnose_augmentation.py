#!/usr/bin/env python3
"""Diagnose the pixel removal augmentation behavior."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.utils.augmentations import RandomPixelRemoval
from src.utils.imagenet_data import get_imagenet_loaders, IMAGENET_MEAN, IMAGENET_STD

def denormalize(tensor):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor.cpu() * std + mean).clamp(0, 1)

print("=== Diagnosing Pixel Removal Augmentation ===\n")

# Test 1: Apply pixel removal to a simple tensor
print("TEST 1: Pixel removal on synthetic image")
img = torch.ones(3, 224, 224) * 0.5  # Gray image
pixel_removal = RandomPixelRemoval(removal_fraction=0.1, probability=1.0)
img_removed = pixel_removal(img.clone())

# Check what values the removed pixels have
removed_mask = (img_removed == 0).all(dim=0)  # True where all channels are 0
num_removed = removed_mask.sum().item()
print(f"  Original: all pixels = 0.5")
print(f"  After removal: {num_removed} pixels set to 0 ({100*num_removed/(224*224):.1f}%)")
print(f"  Removed pixel values: {img_removed[:, removed_mask][:, :5]}")  # First 5 removed pixels

# Test 2: Check unique values in removed image
unique_vals = torch.unique(img_removed)
print(f"  Unique values after removal: {unique_vals}")

# Test 3: Apply to real image and trace through pipeline
print("\nTEST 2: Real ImageNet image through full pipeline")
data_dir = '/leonardo_scratch/fast/CNHPC_1905882/clxai/data/imagenet-1k'

# Load WITHOUT pixel removal
loader_none, _ = get_imagenet_loaders(
    data_dir=data_dir, batch_size=1, num_workers=0,
    contrastive=False, augment=False, augmentation_type='none'
)

# Load WITH pixel removal
loader_pixel, _ = get_imagenet_loaders(
    data_dir=data_dir, batch_size=1, num_workers=0,
    contrastive=False, augment=True, augmentation_type='pixel50'
)

img_none, _ = next(iter(loader_none))
img_pixel, _ = next(iter(loader_pixel))

print(f"  Without pixel removal - min: {img_none.min():.3f}, max: {img_none.max():.3f}")
print(f"  With pixel removal - min: {img_pixel.min():.3f}, max: {img_pixel.max():.3f}")

# Denormalize and check
img_none_denorm = denormalize(img_none[0])
img_pixel_denorm = denormalize(img_pixel[0])

print(f"  Denorm without pixel - min: {img_none_denorm.min():.3f}, max: {img_none_denorm.max():.3f}")
print(f"  Denorm with pixel - min: {img_pixel_denorm.min():.3f}, max: {img_pixel_denorm.max():.3f}")

# Test 4: Visualize pixel removal ONLY (no other augmentations)
print("\nTEST 3: Visualizing pixel removal ONLY")
from torchvision import transforms
from datasets import load_from_disk
import os

hf_dataset = load_from_disk(os.path.join(data_dir, 'train'))

# Simple transform: resize, crop, tensor only
simple_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# Get same image
img_pil = hf_dataset[0]['image'].convert('RGB')
img_tensor = simple_transform(img_pil)

# Apply pixel removal manually
pixel_removal = RandomPixelRemoval(removal_fraction=0.1, probability=1.0)
img_removed = pixel_removal(img_tensor.clone())

# Find removed pixels
diff = (img_tensor - img_removed).abs()
removed_mask = diff.sum(dim=0) > 0.01

print(f"  Pixels changed: {removed_mask.sum().item()} ({100*removed_mask.sum().item()/(224*224):.1f}%)")

# Visualize
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# Original
axes[0].imshow(img_tensor.permute(1, 2, 0).numpy())
axes[0].set_title('Original')
axes[0].axis('off')

# After pixel removal
axes[1].imshow(img_removed.permute(1, 2, 0).numpy())
axes[1].set_title('After Pixel Removal\n(10% → black)')
axes[1].axis('off')

# Difference (amplified)
axes[2].imshow((diff.sum(dim=0) * 5).clamp(0, 1).numpy(), cmap='hot')
axes[2].set_title('Removed Pixels\n(highlighted)')
axes[2].axis('off')

# Mask
axes[3].imshow(removed_mask.numpy(), cmap='gray')
axes[3].set_title(f'Removal Mask\n({removed_mask.sum().item()} pixels)')
axes[3].axis('off')

plt.tight_layout()
plt.savefig('results/pixel_removal_diagnosis.png', dpi=150, bbox_inches='tight')
print("  Saved: results/pixel_removal_diagnosis.png")

# Test 5: Check what happens with contrastive augmentation
print("\nTEST 4: Contrastive pipeline with pixel50")
loader_cont, _ = get_imagenet_loaders(
    data_dir=data_dir, batch_size=1, num_workers=0,
    contrastive=True, augment=True, augmentation_type='pixel50'
)

# Fixed seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

images, _ = next(iter(loader_cont))
view0, view1 = images[0][0], images[1][0]  # First sample from batch

# Check for anomalous values
v0_denorm = denormalize(view0)
v1_denorm = denormalize(view1)

print(f"  View 0 - min: {v0_denorm.min():.3f}, max: {v0_denorm.max():.3f}")
print(f"  View 1 - min: {v1_denorm.min():.3f}, max: {v1_denorm.max():.3f}")

# Check for non-black removed pixels
# After denorm, black should be ~0. Check for pixels that should be black but aren't
v0_normalized = view0  # Still in normalized space
# In normalized space, black = (0 - mean)/std ≈ -2
BLACK_THRESHOLD = -1.5  # Should be around -2 for black pixels
potentially_removed = (v0_normalized < BLACK_THRESHOLD).all(dim=0)
print(f"  Potentially removed pixels (normalized < -1.5): {potentially_removed.sum().item()}")

print("\n=== Diagnosis Complete ===")
