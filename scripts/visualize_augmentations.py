#!/usr/bin/env python3
"""Visualize data augmentations for ImageNet SCL training."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.utils.imagenet_data import get_imagenet_loaders, IMAGENET_MEAN, IMAGENET_STD

def denormalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return (tensor.cpu() * std + mean).clamp(0, 1)

print("Loading data and creating visualizations...")
data_dir = '/leonardo_scratch/fast/CNHPC_1905882/clxai/data/imagenet-1k'

# Figure 1: Different augmentation types
fig, axes = plt.subplots(3, 6, figsize=(18, 9))
aug_types = ['none', 'pixel50', 'noise']

for row, aug_type in enumerate(aug_types):
    print(f"  Loading {aug_type}...")
    loader, _ = get_imagenet_loaders(
        data_dir=data_dir, batch_size=6, num_workers=0,
        contrastive=True, augment=True, augmentation_type=aug_type
    )
    images, labels = next(iter(loader))
    view0 = images[0]
    
    for col in range(6):
        img = denormalize(view0[col]).permute(1, 2, 0).numpy()
        axes[row, col].imshow(img)
        axes[row, col].axis('off')
        if col == 0:
            axes[row, col].set_ylabel(aug_type, fontsize=14)

plt.suptitle('Data Augmentation Examples for SCL Training', fontsize=16, y=1.02)
plt.tight_layout()
output1 = Path("results/augmentation_examples.png")
output1.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output1, dpi=150, bbox_inches='tight')
print(f"Saved: {output1}")

# Figure 2: Two views comparison (contrastive learning)
fig2, axes2 = plt.subplots(2, 6, figsize=(18, 6))

loader, _ = get_imagenet_loaders(
    data_dir=data_dir, batch_size=6, num_workers=0,
    contrastive=True, augment=True, augmentation_type='none'
)
images, labels = next(iter(loader))
view0, view1 = images[0], images[1]

for col in range(6):
    img0 = denormalize(view0[col]).permute(1, 2, 0).numpy()
    axes2[0, col].imshow(img0)
    axes2[0, col].axis('off')
    if col == 0:
        axes2[0, col].set_ylabel('View 0', fontsize=14)
    
    img1 = denormalize(view1[col]).permute(1, 2, 0).numpy()
    axes2[1, col].imshow(img1)
    axes2[1, col].axis('off')
    if col == 0:
        axes2[1, col].set_ylabel('View 1', fontsize=14)

plt.suptitle('Contrastive Learning: Two Augmented Views of Same Images', fontsize=16, y=1.02)
plt.tight_layout()
output2 = Path("results/contrastive_views.png")
plt.savefig(output2, dpi=150, bbox_inches='tight')
print(f"Saved: {output2}")

# Figure 3: pixel50 augmentation detail
fig3, axes3 = plt.subplots(2, 4, figsize=(16, 8))

# Show same images with/without pixel50
loader_none, _ = get_imagenet_loaders(
    data_dir=data_dir, batch_size=4, num_workers=0,
    contrastive=False, augment=False, augmentation_type='none'
)
loader_pixel50, _ = get_imagenet_loaders(
    data_dir=data_dir, batch_size=4, num_workers=0,
    contrastive=False, augment=True, augmentation_type='pixel50'
)

images_none, _ = next(iter(loader_none))
images_pixel, _ = next(iter(loader_pixel50))

for col in range(4):
    img_none = denormalize(images_none[col]).permute(1, 2, 0).numpy()
    axes3[0, col].imshow(img_none)
    axes3[0, col].axis('off')
    axes3[0, col].set_title(f'Original {col+1}')
    
    img_pixel = denormalize(images_pixel[col]).permute(1, 2, 0).numpy()
    axes3[1, col].imshow(img_pixel)
    axes3[1, col].axis('off')
    axes3[1, col].set_title(f'pixel50 (10% removal)')

plt.suptitle('F-Fidelity Pixel Removal Augmentation (50% probability)', fontsize=16, y=1.02)
plt.tight_layout()
output3 = Path("results/pixel50_augmentation.png")
plt.savefig(output3, dpi=150, bbox_inches='tight')
print(f"Saved: {output3}")

plt.close('all')
print("\nAll visualizations saved to results/ directory!")
