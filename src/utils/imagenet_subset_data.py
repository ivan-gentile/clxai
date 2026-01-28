"""
ImageNet 50-class subset data loading utilities for CLXAI.

Based on the Explainable-KD-CNN reference code (Jasper Wi).
Filters ImageNet to 50 specific classes and remaps labels to 0-49.

Reference:
    https://github.com/JasperWi/Explainable-KD-CNN/blob/main/knowledge_distillation_training.py
"""

import os
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as transforms
from typing import Tuple, Optional, Callable, List
from PIL import Image

from datasets import load_from_disk

from src.utils.augmentations import RandomPatchRemoval, RandomPixelRemoval, GaussianNoiseAugmentation

# ImageNet statistics
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Default ImageNet data directory
DEFAULT_IMAGENET_DIR = "/leonardo_scratch/fast/CNHPC_1905882/clxai/data/imagenet-1k"

# 50 selected classes from ImageNet (from Explainable-KD-CNN reference)
# These are the original ImageNet class indices
SELECTED_CLASSES = [
    1, 3, 11, 31, 222, 277, 284, 295, 301, 325,
    330, 333, 342, 368, 386, 388, 404, 412, 418, 436,
    449, 466, 487, 492, 502, 510, 531, 532, 574, 579,
    606, 617, 659, 670, 695, 703, 748, 829, 846, 851,
    861, 879, 883, 898, 900, 914, 919, 951, 959, 992
]

# Create mapping from original class to new class (0-49)
CLASS_MAPPING = {orig_class: new_class for new_class, orig_class in enumerate(SELECTED_CLASSES)}


class ImageNet50HFDataset(Dataset):
    """
    PyTorch Dataset wrapper for HuggingFace ImageNet dataset with 50-class filtering.
    
    Args:
        hf_dataset: HuggingFace dataset loaded from disk
        transform: Transform to apply to images
        selected_classes: List of original ImageNet class indices to include
        class_mapping: Dict mapping original class to new class (0-49)
    """
    
    def __init__(
        self,
        hf_dataset,
        transform: Optional[Callable] = None,
        selected_classes: List[int] = SELECTED_CLASSES,
        class_mapping: dict = CLASS_MAPPING
    ):
        self.dataset = hf_dataset
        self.transform = transform
        self.selected_classes = set(selected_classes)
        self.class_mapping = class_mapping
        
        # Filter indices to only include selected classes
        print(f"Filtering dataset to {len(selected_classes)} classes...")
        self.valid_indices = []
        for idx in range(len(hf_dataset)):
            label = hf_dataset[idx]['label']
            if label in self.selected_classes:
                self.valid_indices.append(idx)
        
        print(f"  Found {len(self.valid_indices)} samples from {len(hf_dataset)} total")
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Get actual index in the underlying dataset
        real_idx = self.valid_indices[idx]
        item = self.dataset[real_idx]
        image = item['image']
        original_label = item['label']
        
        # Remap label to 0-49
        new_label = self.class_mapping[original_label]
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, new_label


class TwoCropTransform:
    """Create two augmented views of the same image for contrastive learning."""
    
    def __init__(self, base_transform):
        self.base_transform = base_transform
    
    def __call__(self, x):
        return [self.base_transform(x), self.base_transform(x)]


class GaussianBlur:
    """
    Gaussian blur augmentation (as used in SimCLR/SupCon papers).
    """
    
    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma
    
    def __call__(self, x):
        import random
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        from torchvision.transforms import functional as F
        return F.gaussian_blur(x, kernel_size=23, sigma=sigma)


def get_imagenet50_train_transforms(
    augment: bool = True,
    augmentation_type: str = 'none',
    image_size: int = 224
) -> transforms.Compose:
    """
    Get training transforms for ImageNet-50 classification.
    Based on the reference code but WITHOUT RandomErasing (replaced by F-Fidelity augmentations).
    
    Args:
        augment: Whether to apply data augmentation
        augmentation_type: Type of augmentation - 'none', 'pixel50', 'noise', 'patch', 'pixel'
        image_size: Target image size (default: 224)
    
    Returns:
        Composed transforms
    """
    transform_list = []
    
    if augment:
        # From reference code (without RandomErasing)
        transform_list.extend([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
        ])
    else:
        transform_list.extend([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
        ])
    
    transform_list.append(transforms.ToTensor())
    
    # Add F-Fidelity augmentation (replacing RandomErasing from reference)
    if augmentation_type == 'pixel50':
        transform_list.append(RandomPixelRemoval(
            removal_fraction=0.1,
            probability=0.5
        ))
    elif augmentation_type == 'noise':
        transform_list.append(GaussianNoiseAugmentation(
            probability=0.5,
            std=0.05
        ))
    elif augmentation_type == 'patch':
        transform_list.append(RandomPatchRemoval(
            probability=0.5,
            patch_sizes=(1/4, 1/8, 1/16),
            replacement='mean',
            dataset='imagenet'
        ))
    elif augmentation_type == 'pixel':
        transform_list.append(RandomPixelRemoval(
            removal_fraction=0.1,
            probability=1.0
        ))
    
    transform_list.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    
    return transforms.Compose(transform_list)


def get_imagenet50_test_transforms(image_size: int = 224) -> transforms.Compose:
    """Get test/evaluation transforms for ImageNet-50."""
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_imagenet50_contrastive_transforms(
    augmentation_type: str = 'none',
    image_size: int = 224
) -> transforms.Compose:
    """
    Get transforms for contrastive learning on ImageNet-50.
    
    Based on SupCon paper (Khosla et al., NeurIPS 2020):
    - RandomResizedCrop with scale (0.08, 1.0)
    - RandomHorizontalFlip
    - ColorJitter (0.8, 0.8, 0.8, 0.2) with p=0.8
    - RandomGrayscale with p=0.2
    - GaussianBlur with p=0.5
    
    Args:
        augmentation_type: Type of custom augmentation - 'none', 'pixel50', 'noise', 'patch', 'pixel'
        image_size: Target image size (default: 224)
    
    Returns:
        Composed transforms
    """
    transform_list = [
        transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
    ]
    
    # GaussianBlur after ToTensor
    transform_list.append(transforms.RandomApply([GaussianBlur(sigma=(0.1, 2.0))], p=0.5))
    
    # Add F-Fidelity augmentation (replacing RandomErasing)
    if augmentation_type == 'pixel50':
        transform_list.append(RandomPixelRemoval(
            removal_fraction=0.1,
            probability=0.5
        ))
    elif augmentation_type == 'noise':
        transform_list.append(GaussianNoiseAugmentation(
            probability=0.5,
            std=0.05
        ))
    elif augmentation_type == 'patch':
        transform_list.append(RandomPatchRemoval(
            probability=0.5,
            patch_sizes=(1/4, 1/8, 1/16),
            replacement='mean',
            dataset='imagenet'
        ))
    elif augmentation_type == 'pixel':
        transform_list.append(RandomPixelRemoval(
            removal_fraction=0.1,
            probability=1.0
        ))
    
    transform_list.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    
    return transforms.Compose(transform_list)


def get_imagenet50_datasets(
    data_dir: str = DEFAULT_IMAGENET_DIR,
    contrastive: bool = False,
    augment: bool = True,
    augmentation_type: str = 'none',
    image_size: int = 224,
) -> Tuple[Dataset, Dataset]:
    """
    Get ImageNet-50 train and validation datasets.
    
    Args:
        data_dir: Directory containing ImageNet Arrow files
        contrastive: Whether to use contrastive transforms (for SCL training)
        augment: Whether to apply data augmentation
        augmentation_type: Type of augmentation - 'none', 'pixel50', 'noise', 'patch', 'pixel'
        image_size: Target image size (default: 224)
    
    Returns:
        train_dataset, val_dataset
    """
    # Load HuggingFace datasets from disk
    train_path = os.path.join(data_dir, 'train')
    val_path = os.path.join(data_dir, 'validation')
    
    print(f"Loading ImageNet train from: {train_path}")
    train_hf = load_from_disk(train_path)
    print(f"  Full train samples: {len(train_hf)}")
    
    print(f"Loading ImageNet validation from: {val_path}")
    val_hf = load_from_disk(val_path)
    print(f"  Full validation samples: {len(val_hf)}")
    
    # Create transforms
    if contrastive:
        train_transform = TwoCropTransform(
            get_imagenet50_contrastive_transforms(augmentation_type, image_size)
        )
    else:
        train_transform = get_imagenet50_train_transforms(
            augment=augment,
            augmentation_type=augmentation_type,
            image_size=image_size
        )
    
    val_transform = get_imagenet50_test_transforms(image_size)
    
    # Create filtered PyTorch datasets
    train_dataset = ImageNet50HFDataset(train_hf, transform=train_transform)
    val_dataset = ImageNet50HFDataset(val_hf, transform=val_transform)
    
    print(f"ImageNet-50 train samples: {len(train_dataset)}")
    print(f"ImageNet-50 val samples: {len(val_dataset)}")
    
    return train_dataset, val_dataset


def get_imagenet50_loaders(
    data_dir: str = DEFAULT_IMAGENET_DIR,
    batch_size: int = 128,
    num_workers: int = 16,
    augment: bool = True,
    contrastive: bool = False,
    pin_memory: bool = True,
    augmentation_type: str = 'none',
    image_size: int = 224,
) -> Tuple[DataLoader, DataLoader]:
    """
    Get ImageNet-50 train and validation data loaders.
    
    Args:
        data_dir: Directory containing ImageNet Arrow files
        batch_size: Batch size
        num_workers: Number of data loading workers
        augment: Whether to apply data augmentation
        contrastive: Whether to use contrastive transforms (for SCL training)
        pin_memory: Whether to pin memory for faster GPU transfer
        augmentation_type: Type of augmentation - 'none', 'pixel50', 'noise', 'patch', 'pixel'
        image_size: Target image size (default: 224)
    
    Returns:
        train_loader, val_loader
    """
    train_dataset, val_dataset = get_imagenet50_datasets(
        data_dir=data_dir,
        contrastive=contrastive,
        augment=augment,
        augmentation_type=augmentation_type,
        image_size=image_size
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    return train_loader, val_loader


def get_class_names() -> List[str]:
    """Return list of 50 class names (if available)."""
    # Could add class name lookup here if needed
    return [f"class_{i}" for i in range(50)]


if __name__ == "__main__":
    print("=" * 60)
    print("ImageNet-50 Data Loading Test")
    print("=" * 60)
    
    print(f"\nSelected classes ({len(SELECTED_CLASSES)}): {SELECTED_CLASSES[:10]}...")
    print(f"Class mapping example: {SELECTED_CLASSES[0]} -> 0, {SELECTED_CLASSES[49]} -> 49")
    
    # Test basic loading
    print("\nTesting ImageNet-50 loaders...")
    try:
        train_loader, val_loader = get_imagenet50_loaders(
            batch_size=4,
            num_workers=0,  # Use 0 for testing
            contrastive=False,
            augmentation_type='pixel50'
        )
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        
        # Get a sample batch
        images, labels = next(iter(val_loader))
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Label range: {labels.min()} - {labels.max()} (should be 0-49)")
        
    except Exception as e:
        print(f"Error loading ImageNet-50: {e}")
        import traceback
        traceback.print_exc()
    
    # Test contrastive loader
    print("\n" + "=" * 60)
    print("Testing Contrastive Transforms")
    print("=" * 60)
    
    try:
        train_loader_cl, _ = get_imagenet50_loaders(
            batch_size=4,
            num_workers=0,
            contrastive=True,
            augmentation_type='pixel50'
        )
        images, labels = next(iter(train_loader_cl))
        print(f"Contrastive batch: {len(images)} views")
        print(f"View 0 shape: {images[0].shape}")
        print(f"View 1 shape: {images[1].shape}")
        print(f"Labels: {labels.tolist()} (should be 0-49)")
        
    except Exception as e:
        print(f"Error with contrastive loader: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nImageNet-50 data loading test completed!")
