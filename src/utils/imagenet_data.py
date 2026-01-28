"""
ImageNet-1K data loading utilities for CLXAI.

Supports multiple augmentation types for contrastive learning:
- 'none': Standard augmentation (RandomResizedCrop, RandomHorizontalFlip, ColorJitter, Grayscale)
- 'pixel50': F-Fidelity style pixel removal with 50% probability (priority)
- 'noise': Gaussian noise augmentation (priority)
- 'patch': Patch removal augmentation
- 'pixel': F-Fidelity style pixel removal with 100% probability

Uses HuggingFace datasets Arrow format for efficient loading.
"""

import os
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from typing import Tuple, Optional, Callable
from PIL import Image

from datasets import load_from_disk

from src.utils.augmentations import RandomPatchRemoval, RandomPixelRemoval, GaussianNoiseAugmentation

# ImageNet statistics
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Default ImageNet data directory
DEFAULT_IMAGENET_DIR = "/leonardo_scratch/fast/CNHPC_1905882/clxai/data/imagenet-1k"


def get_imagenet_stats() -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Return ImageNet mean and std for normalization."""
    return IMAGENET_MEAN, IMAGENET_STD


class ImageNetHFDataset(Dataset):
    """
    PyTorch Dataset wrapper for HuggingFace ImageNet dataset.
    
    Args:
        hf_dataset: HuggingFace dataset loaded from disk
        transform: Transform to apply to images
    """
    
    def __init__(self, hf_dataset, transform: Optional[Callable] = None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        # Convert to RGB if needed (some images might be grayscale)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class TwoCropTransform:
    """Create two augmented views of the same image for contrastive learning."""
    
    def __init__(self, base_transform):
        self.base_transform = base_transform
    
    def __call__(self, x):
        return [self.base_transform(x), self.base_transform(x)]


class GaussianBlur:
    """
    Gaussian blur augmentation (as used in SimCLR/SupCon papers).
    
    Args:
        sigma: Range for sigma selection (default: [0.1, 2.0])
    """
    
    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma
    
    def __call__(self, x):
        import random
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        from torchvision.transforms import functional as F
        return F.gaussian_blur(x, kernel_size=23, sigma=sigma)


def get_imagenet_train_transforms(
    augment: bool = True,
    augmentation_type: str = 'none',
    image_size: int = 224
) -> transforms.Compose:
    """
    Get training transforms for ImageNet classification.
    
    Args:
        augment: Whether to apply data augmentation
        augmentation_type: Type of augmentation - 'none', 'pixel50', 'noise', 'patch', 'pixel'
        image_size: Target image size (default: 224)
    
    Returns:
        Composed transforms
    """
    transform_list = []
    
    if augment:
        transform_list.extend([
            transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
        ])
    else:
        transform_list.extend([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
        ])
    
    transform_list.append(transforms.ToTensor())
    
    # Add custom augmentation (applied on tensor in [0,1] range, before normalize)
    if augmentation_type == 'pixel50':
        # Priority: F-Fidelity style pixel removal with 50% probability
        transform_list.append(RandomPixelRemoval(
            removal_fraction=0.1,  # β = 10% as in F-Fidelity paper
            probability=0.5
        ))
    elif augmentation_type == 'noise':
        # Priority: Gaussian noise augmentation
        transform_list.append(GaussianNoiseAugmentation(
            probability=0.5,
            std=0.05
        ))
    elif augmentation_type == 'patch':
        # Patch removal scaled for ImageNet (224x224)
        # Patch sizes: 56, 28, 14 pixels (1/4, 1/8, 1/16 of 224)
        transform_list.append(RandomPatchRemoval(
            probability=0.5,
            patch_sizes=(1/4, 1/8, 1/16),
            replacement='mean',
            dataset='imagenet'
        ))
    elif augmentation_type == 'pixel':
        # F-Fidelity style: 10% scattered pixel removal, always applied
        transform_list.append(RandomPixelRemoval(
            removal_fraction=0.1,
            probability=1.0
        ))
    
    transform_list.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    
    return transforms.Compose(transform_list)


def get_imagenet_test_transforms(image_size: int = 224) -> transforms.Compose:
    """Get test/evaluation transforms for ImageNet."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_imagenet_contrastive_transforms(
    augmentation_type: str = 'none',
    image_size: int = 224
) -> transforms.Compose:
    """
    Get transforms for contrastive learning on ImageNet.
    
    Based on SupCon paper (Khosla et al., NeurIPS 2020):
    - RandomResizedCrop with scale (0.08, 1.0)
    - RandomHorizontalFlip
    - ColorJitter (0.8, 0.8, 0.8, 0.2) with p=0.8 (stronger than CIFAR)
    - RandomGrayscale with p=0.2
    - GaussianBlur with p=0.5 (ImageNet-specific)
    
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
            transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)  # Stronger than CIFAR (0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
    ]
    
    # GaussianBlur after ToTensor (SupCon paper uses this for ImageNet)
    transform_list.append(transforms.RandomApply([GaussianBlur(sigma=(0.1, 2.0))], p=0.5))
    
    # Add custom augmentation (applied on tensor in [0,1] range, before normalize)
    if augmentation_type == 'pixel50':
        # Priority: F-Fidelity style pixel removal with 50% probability
        transform_list.append(RandomPixelRemoval(
            removal_fraction=0.1,
            probability=0.5
        ))
    elif augmentation_type == 'noise':
        # Priority: Gaussian noise augmentation
        transform_list.append(GaussianNoiseAugmentation(
            probability=0.5,
            std=0.05
        ))
    elif augmentation_type == 'patch':
        # Patch removal scaled for ImageNet
        transform_list.append(RandomPatchRemoval(
            probability=0.5,
            patch_sizes=(1/4, 1/8, 1/16),
            replacement='mean',
            dataset='imagenet'
        ))
    elif augmentation_type == 'pixel':
        # F-Fidelity style: 10% scattered pixel removal, always applied
        transform_list.append(RandomPixelRemoval(
            removal_fraction=0.1,
            probability=1.0
        ))
    
    transform_list.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    
    return transforms.Compose(transform_list)


def get_imagenet_datasets(
    data_dir: str = DEFAULT_IMAGENET_DIR,
    contrastive: bool = False,
    augment: bool = True,
    augmentation_type: str = 'none',
    image_size: int = 224,
) -> Tuple[Dataset, Dataset]:
    """
    Get ImageNet train and validation datasets (for DDP with custom samplers).
    
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
    
    train_hf = load_from_disk(train_path)
    val_hf = load_from_disk(val_path)
    
    # Create transforms
    if contrastive:
        train_transform = TwoCropTransform(
            get_imagenet_contrastive_transforms(augmentation_type, image_size)
        )
    else:
        train_transform = get_imagenet_train_transforms(
            augment=augment,
            augmentation_type=augmentation_type,
            image_size=image_size
        )
    
    val_transform = get_imagenet_test_transforms(image_size)
    
    # Create PyTorch datasets
    train_dataset = ImageNetHFDataset(train_hf, transform=train_transform)
    val_dataset = ImageNetHFDataset(val_hf, transform=val_transform)
    
    return train_dataset, val_dataset


def get_imagenet_loaders(
    data_dir: str = DEFAULT_IMAGENET_DIR,
    batch_size: int = 256,
    num_workers: int = 16,
    augment: bool = True,
    contrastive: bool = False,
    pin_memory: bool = True,
    augmentation_type: str = 'none',
    image_size: int = 224,
    distributed: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Get ImageNet train and validation data loaders.
    
    Args:
        data_dir: Directory containing ImageNet Arrow files
        batch_size: Batch size
        num_workers: Number of data loading workers
        augment: Whether to apply data augmentation (for classification training)
        contrastive: Whether to use contrastive transforms (for SCL training)
        pin_memory: Whether to pin memory for faster GPU transfer
        augmentation_type: Type of augmentation - 'none', 'pixel50', 'noise', 'patch', 'pixel'
        image_size: Target image size (default: 224)
        distributed: Whether using distributed training (affects sampler)
    
    Returns:
        train_loader, val_loader
    """
    # Load HuggingFace datasets from disk
    train_path = os.path.join(data_dir, 'train')
    val_path = os.path.join(data_dir, 'validation')
    
    print(f"Loading ImageNet train from: {train_path}")
    train_hf = load_from_disk(train_path)
    print(f"  Train samples: {len(train_hf)}")
    
    print(f"Loading ImageNet validation from: {val_path}")
    val_hf = load_from_disk(val_path)
    print(f"  Validation samples: {len(val_hf)}")
    
    # Create transforms
    if contrastive:
        train_transform = TwoCropTransform(
            get_imagenet_contrastive_transforms(augmentation_type, image_size)
        )
    else:
        train_transform = get_imagenet_train_transforms(
            augment=augment,
            augmentation_type=augmentation_type,
            image_size=image_size
        )
    
    val_transform = get_imagenet_test_transforms(image_size)
    
    # Create PyTorch datasets
    train_dataset = ImageNetHFDataset(train_hf, transform=train_transform)
    val_dataset = ImageNetHFDataset(val_hf, transform=val_transform)
    
    # Create samplers for distributed training
    train_sampler = None
    val_sampler = None
    
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    return train_loader, val_loader


def get_raw_imagenet_loader(
    data_dir: str = DEFAULT_IMAGENET_DIR,
    batch_size: int = 256,
    num_workers: int = 16,
    train: bool = False,
    image_size: int = 224,
) -> DataLoader:
    """
    Get ImageNet loader WITHOUT normalization (for visualization).
    
    Args:
        data_dir: Directory containing ImageNet Arrow files
        batch_size: Batch size
        num_workers: Number of workers
        train: Whether to load training set
        image_size: Target image size
    
    Returns:
        DataLoader with images in [0, 1] range
    """
    split = 'train' if train else 'validation'
    hf_dataset = load_from_disk(os.path.join(data_dir, split))
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])
    
    dataset = ImageNetHFDataset(hf_dataset, transform=transform)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )


def denormalize_imagenet(tensor: torch.Tensor) -> torch.Tensor:
    """
    Denormalize an ImageNet tensor back to [0, 1] range.
    
    Args:
        tensor: Normalized tensor (C, H, W) or (B, C, H, W)
    
    Returns:
        Denormalized tensor in [0, 1] range
    """
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    
    if tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    
    mean = mean.to(tensor.device)
    std = std.to(tensor.device)
    
    return tensor * std + mean


if __name__ == "__main__":
    print("=" * 60)
    print("ImageNet Data Loading Test")
    print("=" * 60)
    
    # Test basic loading
    print("\nTesting ImageNet loaders...")
    try:
        train_loader, val_loader = get_imagenet_loaders(
            batch_size=4,
            num_workers=0,  # Use 0 for testing
            contrastive=False
        )
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        
        # Get a sample batch
        images, labels = next(iter(val_loader))
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Label range: {labels.min()} - {labels.max()}")
        
    except Exception as e:
        print(f"Error loading ImageNet: {e}")
        print("Make sure ImageNet is downloaded to the expected location.")
    
    # Test contrastive loader
    print("\n" + "=" * 60)
    print("Testing Contrastive Transforms")
    print("=" * 60)
    
    try:
        train_loader_cl, _ = get_imagenet_loaders(
            batch_size=4,
            num_workers=0,
            contrastive=True
        )
        images, labels = next(iter(train_loader_cl))
        print(f"Contrastive batch: {len(images)} views")
        print(f"View 0 shape: {images[0].shape}")
        print(f"View 1 shape: {images[1].shape}")
        
    except Exception as e:
        print(f"Error with contrastive loader: {e}")
    
    # Test augmentation types
    print("\n" + "=" * 60)
    print("Testing Augmentation Types")
    print("=" * 60)
    
    for aug_type in ['none', 'pixel50', 'noise', 'patch', 'pixel']:
        try:
            train_loader, _ = get_imagenet_loaders(
                batch_size=2,
                num_workers=0,
                contrastive=True,
                augmentation_type=aug_type
            )
            images, _ = next(iter(train_loader))
            print(f"  {aug_type}: OK - shape {images[0].shape}")
        except Exception as e:
            print(f"  {aug_type}: ERROR - {e}")
    
    print("\nImageNet data loading test completed!")
