"""
CIFAR-10/100 data loading utilities for CLXAI.

Supports multiple augmentation types:
- 'none': Standard augmentation (RandomCrop, RandomHorizontalFlip)
- 'patch': F-Fidelity style patch removal for occlusion robustness
- 'noise': Gaussian noise for continuous perturbation robustness
"""

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, Optional

from src.utils.augmentations import RandomPatchRemoval, GaussianNoiseAugmentation


# CIFAR-10 statistics
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# CIFAR-100 statistics
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def get_cifar10_stats() -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Return CIFAR-10 mean and std for normalization."""
    return CIFAR10_MEAN, CIFAR10_STD


def get_cifar100_stats() -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Return CIFAR-100 mean and std for normalization."""
    return CIFAR100_MEAN, CIFAR100_STD


def get_train_transforms(
    augment: bool = True,
    augmentation_type: str = 'none'
) -> transforms.Compose:
    """
    Get training transforms for CIFAR-10.
    
    Args:
        augment: Whether to apply data augmentation
        augmentation_type: Type of augmentation - 'none', 'patch', or 'noise'
    
    Returns:
        Composed transforms
    """
    transform_list = []
    
    if augment:
        transform_list.extend([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    
    transform_list.append(transforms.ToTensor())
    
    # Add custom augmentation (applied on tensor in [0,1] range, before normalize)
    if augmentation_type == 'patch':
        transform_list.append(RandomPatchRemoval(
            probability=0.5,
            patch_sizes=(1/4, 1/8, 1/16),  # 8, 4, 2 pixels for CIFAR-10
            replacement='mean',
            dataset='cifar10'
        ))
    elif augmentation_type == 'noise':
        transform_list.append(GaussianNoiseAugmentation(
            probability=0.5,
            std=0.05
        ))
    
    transform_list.append(transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD))
    
    return transforms.Compose(transform_list)


def get_test_transforms() -> transforms.Compose:
    """Get test/evaluation transforms for CIFAR-10."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def get_contrastive_transforms(augmentation_type: str = 'none') -> transforms.Compose:
    """
    Get transforms for contrastive learning.
    Returns two augmented views of each image.
    
    Args:
        augmentation_type: Type of augmentation - 'none', 'patch', or 'noise'
    
    Returns:
        Composed transforms
    """
    transform_list = [
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
    ]
    
    # Add custom augmentation (applied on tensor in [0,1] range, before normalize)
    if augmentation_type == 'patch':
        transform_list.append(RandomPatchRemoval(
            probability=0.5,
            patch_sizes=(1/4, 1/8, 1/16),
            replacement='mean',
            dataset='cifar10'
        ))
    elif augmentation_type == 'noise':
        transform_list.append(GaussianNoiseAugmentation(
            probability=0.5,
            std=0.05
        ))
    
    transform_list.append(transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD))
    
    return transforms.Compose(transform_list)


class TwoCropTransform:
    """Create two augmented views of the same image for contrastive learning."""
    
    def __init__(self, base_transform):
        self.base_transform = base_transform
    
    def __call__(self, x):
        return [self.base_transform(x), self.base_transform(x)]


def get_cifar10_loaders(
    data_dir: str = './data',
    batch_size: int = 128,
    num_workers: int = 4,
    augment: bool = True,
    contrastive: bool = False,
    pin_memory: bool = True,
    augmentation_type: str = 'none',
) -> Tuple[DataLoader, DataLoader]:
    """
    Get CIFAR-10 train and test data loaders.
    
    Args:
        data_dir: Directory to store/load data
        batch_size: Batch size
        num_workers: Number of data loading workers
        augment: Whether to apply data augmentation (for CE training)
        contrastive: Whether to use contrastive transforms (for SCL training)
        pin_memory: Whether to pin memory for faster GPU transfer
        augmentation_type: Type of augmentation - 'none', 'patch', or 'noise'
    
    Returns:
        train_loader, test_loader
    """
    if contrastive:
        train_transform = TwoCropTransform(get_contrastive_transforms(augmentation_type))
    else:
        train_transform = get_train_transforms(augment=augment, augmentation_type=augmentation_type)
    
    test_transform = get_test_transforms()
    
    # download=False since compute nodes have no internet
    # Data must be pre-downloaded on login node
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,
        transform=train_transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=False,
        transform=test_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, test_loader


def get_raw_cifar10_loader(
    data_dir: str = './data',
    batch_size: int = 128,
    num_workers: int = 4,
    train: bool = False,
) -> DataLoader:
    """
    Get CIFAR-10 loader WITHOUT normalization (for visualization).
    
    Args:
        data_dir: Directory to store/load data
        batch_size: Batch size
        num_workers: Number of workers
        train: Whether to load training set
    
    Returns:
        DataLoader with images in [0, 1] range
    """
    transform = transforms.ToTensor()
    
    dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=train,
        download=False,  # Compute nodes have no internet
        transform=transform
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )


def denormalize(tensor: torch.Tensor, dataset: str = 'cifar10') -> torch.Tensor:
    """
    Denormalize a CIFAR tensor back to [0, 1] range.
    
    Args:
        tensor: Normalized tensor (C, H, W) or (B, C, H, W)
        dataset: 'cifar10' or 'cifar100'
    
    Returns:
        Denormalized tensor in [0, 1] range
    """
    if dataset == 'cifar100':
        mean = torch.tensor(CIFAR100_MEAN).view(3, 1, 1)
        std = torch.tensor(CIFAR100_STD).view(3, 1, 1)
    else:
        mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
        std = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    
    if tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    
    mean = mean.to(tensor.device)
    std = std.to(tensor.device)
    
    return tensor * std + mean


# ============================================================================
# CIFAR-100 Functions
# ============================================================================

def get_cifar100_train_transforms(
    augment: bool = True,
    augmentation_type: str = 'none'
) -> transforms.Compose:
    """
    Get training transforms for CIFAR-100.
    
    Args:
        augment: Whether to apply data augmentation
        augmentation_type: Type of augmentation - 'none', 'patch', or 'noise'
    
    Returns:
        Composed transforms
    """
    transform_list = []
    
    if augment:
        transform_list.extend([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    
    transform_list.append(transforms.ToTensor())
    
    # Add custom augmentation (applied on tensor in [0,1] range, before normalize)
    if augmentation_type == 'patch':
        transform_list.append(RandomPatchRemoval(
            probability=0.5,
            patch_sizes=(1/4, 1/8, 1/16),
            replacement='mean',
            dataset='cifar100'
        ))
    elif augmentation_type == 'noise':
        transform_list.append(GaussianNoiseAugmentation(
            probability=0.5,
            std=0.05
        ))
    
    transform_list.append(transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD))
    
    return transforms.Compose(transform_list)


def get_cifar100_test_transforms() -> transforms.Compose:
    """Get test/evaluation transforms for CIFAR-100."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])


def get_cifar100_contrastive_transforms(augmentation_type: str = 'none') -> transforms.Compose:
    """
    Get transforms for contrastive learning on CIFAR-100.
    Returns two augmented views of each image.
    
    Args:
        augmentation_type: Type of augmentation - 'none', 'patch', or 'noise'
    
    Returns:
        Composed transforms
    """
    transform_list = [
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
    ]
    
    # Add custom augmentation (applied on tensor in [0,1] range, before normalize)
    if augmentation_type == 'patch':
        transform_list.append(RandomPatchRemoval(
            probability=0.5,
            patch_sizes=(1/4, 1/8, 1/16),
            replacement='mean',
            dataset='cifar100'
        ))
    elif augmentation_type == 'noise':
        transform_list.append(GaussianNoiseAugmentation(
            probability=0.5,
            std=0.05
        ))
    
    transform_list.append(transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD))
    
    return transforms.Compose(transform_list)


def get_cifar100_loaders(
    data_dir: str = './data',
    batch_size: int = 128,
    num_workers: int = 4,
    augment: bool = True,
    contrastive: bool = False,
    pin_memory: bool = True,
    augmentation_type: str = 'none',
) -> Tuple[DataLoader, DataLoader]:
    """
    Get CIFAR-100 train and test data loaders.
    
    Args:
        data_dir: Directory to store/load data
        batch_size: Batch size
        num_workers: Number of data loading workers
        augment: Whether to apply data augmentation (for CE training)
        contrastive: Whether to use contrastive transforms (for SCL training)
        pin_memory: Whether to pin memory for faster GPU transfer
        augmentation_type: Type of augmentation - 'none', 'patch', or 'noise'
    
    Returns:
        train_loader, test_loader
    """
    if contrastive:
        train_transform = TwoCropTransform(get_cifar100_contrastive_transforms(augmentation_type))
    else:
        train_transform = get_cifar100_train_transforms(augment=augment, augmentation_type=augmentation_type)
    
    test_transform = get_cifar100_test_transforms()
    
    # download=False since compute nodes have no internet
    # Data must be pre-downloaded on login node
    train_dataset = torchvision.datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=False,
        transform=train_transform
    )
    
    test_dataset = torchvision.datasets.CIFAR100(
        root=data_dir,
        train=False,
        download=False,
        transform=test_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, test_loader


# ============================================================================
# Generic Data Loading Functions
# ============================================================================

def get_data_loaders(
    dataset: str = 'cifar10',
    data_dir: str = './data',
    batch_size: int = 128,
    num_workers: int = 4,
    augment: bool = True,
    contrastive: bool = False,
    pin_memory: bool = True,
    augmentation_type: str = 'none',
) -> Tuple[DataLoader, DataLoader]:
    """
    Generic function to get train and test data loaders.
    
    Args:
        dataset: 'cifar10' or 'cifar100'
        data_dir: Directory to store/load data
        batch_size: Batch size
        num_workers: Number of data loading workers
        augment: Whether to apply data augmentation
        contrastive: Whether to use contrastive transforms
        pin_memory: Whether to pin memory for faster GPU transfer
        augmentation_type: Type of augmentation - 'none', 'patch', or 'noise'
    
    Returns:
        train_loader, test_loader
    """
    if dataset == 'cifar10':
        return get_cifar10_loaders(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            augment=augment,
            contrastive=contrastive,
            pin_memory=pin_memory,
            augmentation_type=augmentation_type
        )
    elif dataset == 'cifar100':
        return get_cifar100_loaders(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            augment=augment,
            contrastive=contrastive,
            pin_memory=pin_memory,
            augmentation_type=augmentation_type
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Use 'cifar10' or 'cifar100'.")


def get_num_classes(dataset: str) -> int:
    """Return the number of classes for a given dataset."""
    if dataset == 'cifar10':
        return 10
    elif dataset == 'cifar100':
        return 100
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


if __name__ == "__main__":
    print("=" * 50)
    print("CIFAR-10 Data Loading")
    print("=" * 50)
    
    train_loader, test_loader = get_cifar10_loaders(batch_size=64)
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Label range: {labels.min()} - {labels.max()}")
    
    # Test contrastive loader
    train_loader_cl, _ = get_cifar10_loaders(batch_size=64, contrastive=True)
    images, labels = next(iter(train_loader_cl))
    print(f"Contrastive batch: {len(images)} views, shape {images[0].shape}")
    
    print("\n" + "=" * 50)
    print("CIFAR-100 Data Loading")
    print("=" * 50)
    
    train_loader_100, test_loader_100 = get_cifar100_loaders(batch_size=64)
    print(f"Train batches: {len(train_loader_100)}")
    print(f"Test batches: {len(test_loader_100)}")
    
    images, labels = next(iter(train_loader_100))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Label range: {labels.min()} - {labels.max()}")
    
    # Test contrastive loader
    train_loader_cl_100, _ = get_cifar100_loaders(batch_size=64, contrastive=True)
    images, labels = next(iter(train_loader_cl_100))
    print(f"Contrastive batch: {len(images)} views, shape {images[0].shape}")
    
    print("\n" + "=" * 50)
    print("Generic get_data_loaders()")
    print("=" * 50)
    train_loader, test_loader = get_data_loaders(dataset='cifar10', batch_size=64)
    print(f"CIFAR-10 via generic: {len(train_loader)} train batches")
    train_loader, test_loader = get_data_loaders(dataset='cifar100', batch_size=64)
    print(f"CIFAR-100 via generic: {len(train_loader)} train batches")
    print(f"CIFAR-10 classes: {get_num_classes('cifar10')}")
    print(f"CIFAR-100 classes: {get_num_classes('cifar100')}")
    
    print("\n" + "=" * 50)
    print("Augmentation Types")
    print("=" * 50)
    
    # Test patch augmentation
    train_loader_patch, _ = get_data_loaders(
        dataset='cifar10', batch_size=64, augmentation_type='patch'
    )
    print(f"Patch augmentation loader: {len(train_loader_patch)} train batches")
    
    # Test noise augmentation
    train_loader_noise, _ = get_data_loaders(
        dataset='cifar10', batch_size=64, augmentation_type='noise'
    )
    print(f"Noise augmentation loader: {len(train_loader_noise)} train batches")
    
    # Test contrastive with augmentation
    train_loader_cl_patch, _ = get_data_loaders(
        dataset='cifar10', batch_size=64, contrastive=True, augmentation_type='patch'
    )
    images, labels = next(iter(train_loader_cl_patch))
    print(f"Contrastive + Patch: {len(images)} views, shape {images[0].shape}")