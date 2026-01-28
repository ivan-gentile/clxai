#!/usr/bin/env python3
"""
Quick test for ImageNet SCL pipeline:
1. Test write permissions
2. Test data loading with augmentations
3. Run a few training iterations
4. Visualize augmented samples
"""

import os
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Test 1: Write permissions
print("=" * 60)
print("TEST 1: Write Permissions")
print("=" * 60)

test_dir = Path("/leonardo_scratch/fast/CNHPC_1905882/clxai/results/test_permissions")
test_dir.mkdir(parents=True, exist_ok=True)
test_file = test_dir / "test_write.txt"

try:
    with open(test_file, 'w') as f:
        f.write(f"Test write at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"User: {os.environ.get('USER', 'unknown')}\n")
        f.write(f"SLURM_JOB_ACCOUNT: {os.environ.get('SLURM_JOB_ACCOUNT', 'N/A')}\n")
    print(f"OK: Successfully wrote to {test_file}")
    
    # Read back
    with open(test_file, 'r') as f:
        content = f.read()
    print(f"Content:\n{content}")
    
    # Clean up
    test_file.unlink()
    print("OK: Write permissions verified!")
except Exception as e:
    print(f"ERROR: Write permission failed: {e}")
    sys.exit(1)

# Test 2: Data loading with augmentations
print("\n" + "=" * 60)
print("TEST 2: Data Loading with Augmentations")
print("=" * 60)

from src.utils.imagenet_data import get_imagenet_loaders, IMAGENET_MEAN, IMAGENET_STD

data_dir = '/leonardo_scratch/fast/CNHPC_1905882/clxai/data/imagenet-1k'

# Test different augmentation types
aug_types = ['none', 'pixel50', 'noise']

for aug_type in aug_types:
    print(f"\nTesting augmentation: {aug_type}")
    try:
        loader, _ = get_imagenet_loaders(
            data_dir=data_dir,
            batch_size=8,
            num_workers=4,
            contrastive=True,
            augment=True,
            augmentation_type=aug_type
        )
        images, labels = next(iter(loader))
        if isinstance(images, list):
            print(f"  View 0: {images[0].shape}, View 1: {images[1].shape}")
        else:
            print(f"  Shape: {images.shape}")
        print(f"  OK: {aug_type} augmentation works")
    except Exception as e:
        print(f"  ERROR: {e}")

# Test 3: Model and training iterations
print("\n" + "=" * 60)
print("TEST 3: Quick Training Test (5 iterations)")
print("=" * 60)

from src.models.imagenet_resnet import get_imagenet_resnet
from src.training.losses import SupConLossV2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Create model
model = get_imagenet_resnet('resnet152_imagenet', encoder_only=True, embedding_dim=128).to(device)
print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# Create loss and optimizer
criterion = SupConLossV2(temperature=0.1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Get a small loader
train_loader, _ = get_imagenet_loaders(
    data_dir=data_dir,
    batch_size=16,
    num_workers=4,
    contrastive=True,
    augment=True,
    augmentation_type='none'
)

# Run 5 iterations
model.train()
for i, (images, labels) in enumerate(train_loader):
    if i >= 5:
        break
    
    t0 = time.time()
    
    # Concatenate two views
    images = torch.cat([images[0], images[1]], dim=0).to(device)
    labels = labels.to(device)
    
    optimizer.zero_grad()
    embeddings = model(images)
    loss = criterion(embeddings, labels)
    loss.backward()
    optimizer.step()
    
    t1 = time.time()
    print(f"  Iter {i+1}: loss={loss.item():.4f}, time={t1-t0:.2f}s")

print("OK: Training iterations completed!")

# Test 4: Save augmentation visualization
print("\n" + "=" * 60)
print("TEST 4: Visualizing Augmentations")
print("=" * 60)

def denormalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Denormalize tensor back to [0, 1] range."""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return (tensor.cpu() * std + mean).clamp(0, 1)

# Create figure
fig, axes = plt.subplots(3, 6, figsize=(18, 9))
aug_types_viz = ['none', 'pixel50', 'noise']

for row, aug_type in enumerate(aug_types_viz):
    loader, _ = get_imagenet_loaders(
        data_dir=data_dir,
        batch_size=6,
        num_workers=0,
        contrastive=True,
        augment=True,
        augmentation_type=aug_type
    )
    
    images, labels = next(iter(loader))
    view0 = images[0]  # First view
    
    for col in range(6):
        img = denormalize(view0[col])
        img = img.permute(1, 2, 0).numpy()
        
        axes[row, col].imshow(img)
        axes[row, col].axis('off')
        if col == 0:
            axes[row, col].set_ylabel(f'{aug_type}', fontsize=14)
        if row == 0:
            axes[row, col].set_title(f'Sample {col+1}', fontsize=12)

plt.suptitle('Data Augmentation Examples (Contrastive View 0)', fontsize=16)
plt.tight_layout()

# Save figure
output_path = Path("/leonardo_scratch/fast/CNHPC_1905882/clxai/results/augmentation_examples.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"OK: Saved augmentation visualization to {output_path}")

# Also save second views comparison
fig2, axes2 = plt.subplots(2, 6, figsize=(18, 6))

loader, _ = get_imagenet_loaders(
    data_dir=data_dir,
    batch_size=6,
    num_workers=0,
    contrastive=True,
    augment=True,
    augmentation_type='none'
)

images, labels = next(iter(loader))
view0, view1 = images[0], images[1]

for col in range(6):
    # View 0
    img0 = denormalize(view0[col]).permute(1, 2, 0).numpy()
    axes2[0, col].imshow(img0)
    axes2[0, col].axis('off')
    if col == 0:
        axes2[0, col].set_ylabel('View 0', fontsize=14)
    
    # View 1
    img1 = denormalize(view1[col]).permute(1, 2, 0).numpy()
    axes2[1, col].imshow(img1)
    axes2[1, col].axis('off')
    if col == 0:
        axes2[1, col].set_ylabel('View 1', fontsize=14)

plt.suptitle('Contrastive Learning: Two Augmented Views of Same Images', fontsize=16)
plt.tight_layout()

output_path2 = Path("/leonardo_scratch/fast/CNHPC_1905882/clxai/results/contrastive_views.png")
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"OK: Saved contrastive views to {output_path2}")

plt.close('all')

print("\n" + "=" * 60)
print("ALL TESTS COMPLETED SUCCESSFULLY!")
print("=" * 60)
print(f"\nVisualization files saved to:")
print(f"  - {output_path}")
print(f"  - {output_path2}")
