#!/usr/bin/env python3
"""
Download CIFAR-10 dataset to the data directory.
Run this on login node (has internet access) before submitting training jobs.
"""

import os
import sys

# Set data directory
DATA_DIR = "/leonardo_scratch/fast/CNHPC_1905882/clxai/data"

print(f"Downloading CIFAR-10 to: {DATA_DIR}")
os.makedirs(DATA_DIR, exist_ok=True)

# Import torchvision and download
import torchvision

print("Downloading training set...")
train_dataset = torchvision.datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
    download=True
)
print(f"Training samples: {len(train_dataset)}")

print("Downloading test set...")
test_dataset = torchvision.datasets.CIFAR10(
    root=DATA_DIR,
    train=False,
    download=True
)
print(f"Test samples: {len(test_dataset)}")

print("\nCIFAR-10 download complete!")
print(f"Data stored in: {DATA_DIR}")

# List contents
print("\nDirectory contents:")
for item in os.listdir(DATA_DIR):
    item_path = os.path.join(DATA_DIR, item)
    if os.path.isdir(item_path):
        print(f"  [DIR] {item}")
    else:
        size = os.path.getsize(item_path) / (1024*1024)
        print(f"  {item} ({size:.1f} MB)")

