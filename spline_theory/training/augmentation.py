"""
Additional augmentation strategies for spline theory experiments.

Implements:
- Patch occlusion (random rectangular patches)
- Gaussian noise injection
- Combined augmentation pipelines

These augmentations test hypothesis about whether CL makes augmentation redundant
for achieving robust geometry.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from typing import Tuple, Optional, Literal


class PatchOcclusion:
    """
    Random patch occlusion augmentation.
    
    Randomly masks rectangular patches of the image with a fill value.
    Tests robustness to partial occlusion.
    """
    
    def __init__(
        self,
        patch_size: Tuple[int, int] = (8, 8),
        num_patches: int = 1,
        fill_value: float = 0.0,
        random_location: bool = True
    ):
        """
        Args:
            patch_size: (height, width) of each patch
            num_patches: Number of patches to occlude
            fill_value: Value to fill patches (0=black, 0.5=gray)
            random_location: If True, randomize patch locations
        """
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.fill_value = fill_value
        self.random_location = random_location
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply patch occlusion.
        
        Args:
            img: Input image tensor (C, H, W)
        
        Returns:
            Occluded image tensor
        """
        img = img.clone()
        _, h, w = img.shape
        ph, pw = self.patch_size
        
        for _ in range(self.num_patches):
            if self.random_location:
                # Random top-left corner
                top = np.random.randint(0, max(1, h - ph + 1))
                left = np.random.randint(0, max(1, w - pw + 1))
            else:
                # Center patch
                top = (h - ph) // 2
                left = (w - pw) // 2
            
            img[:, top:top + ph, left:left + pw] = self.fill_value
        
        return img


class GaussianNoise:
    """
    Gaussian noise injection augmentation.
    
    Adds random Gaussian noise to the image.
    """
    
    def __init__(
        self,
        mean: float = 0.0,
        std: float = 0.1,
        clip: bool = True
    ):
        """
        Args:
            mean: Mean of Gaussian noise
            std: Standard deviation of noise
            clip: Whether to clip output to valid range
        """
        self.mean = mean
        self.std = std
        self.clip = clip
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply Gaussian noise.
        
        Args:
            img: Input image tensor (C, H, W)
        
        Returns:
            Noisy image tensor
        """
        noise = torch.randn_like(img) * self.std + self.mean
        noisy_img = img + noise
        
        if self.clip:
            # Clip to reasonable range (assuming normalized images)
            noisy_img = torch.clamp(noisy_img, -3.0, 3.0)
        
        return noisy_img


class RandomPatchSize:
    """
    Patch occlusion with randomly varying patch sizes.
    """
    
    def __init__(
        self,
        min_size: int = 4,
        max_size: int = 12,
        num_patches: int = 1,
        fill_value: float = 0.0
    ):
        self.min_size = min_size
        self.max_size = max_size
        self.num_patches = num_patches
        self.fill_value = fill_value
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        img = img.clone()
        _, h, w = img.shape
        
        for _ in range(self.num_patches):
            ph = np.random.randint(self.min_size, self.max_size + 1)
            pw = np.random.randint(self.min_size, self.max_size + 1)
            
            top = np.random.randint(0, max(1, h - ph + 1))
            left = np.random.randint(0, max(1, w - pw + 1))
            
            img[:, top:top + ph, left:left + pw] = self.fill_value
        
        return img


# CIFAR statistics for normalization
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def get_augmentation_transform(
    augmentation_type: Literal["none", "standard", "patch", "noise", "strong"],
    dataset: str = "cifar10",
    for_contrastive: bool = False
) -> transforms.Compose:
    """
    Get augmentation transform based on type.
    
    Augmentation levels for Phase 4 experiments:
    - none: Only normalization
    - standard: Random crop + horizontal flip
    - patch: Standard + random patch occlusion
    - noise: Standard + Gaussian noise
    - strong: Standard + patch + noise
    
    Args:
        augmentation_type: Type of augmentation
        dataset: 'cifar10' or 'cifar100'
        for_contrastive: If True, use contrastive-specific augmentations
    
    Returns:
        Composed transform
    """
    if dataset == "cifar100":
        mean, std = CIFAR100_MEAN, CIFAR100_STD
    else:
        mean, std = CIFAR10_MEAN, CIFAR10_STD
    
    normalize = transforms.Normalize(mean, std)
    
    if for_contrastive:
        # Contrastive learning augmentations (stronger)
        base_transforms = [
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ]
    else:
        # Standard augmentations
        base_transforms = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    
    if augmentation_type == "none":
        return transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    
    elif augmentation_type == "standard":
        return transforms.Compose(base_transforms)
    
    elif augmentation_type == "patch":
        return transforms.Compose([
            *base_transforms,
            PatchOcclusion(patch_size=(8, 8), num_patches=1, fill_value=0.0),
        ])
    
    elif augmentation_type == "noise":
        return transforms.Compose([
            *base_transforms,
            GaussianNoise(mean=0.0, std=0.1),
        ])
    
    elif augmentation_type == "strong":
        return transforms.Compose([
            *base_transforms,
            PatchOcclusion(patch_size=(6, 6), num_patches=2, fill_value=0.0),
            GaussianNoise(mean=0.0, std=0.05),
        ])
    
    else:
        raise ValueError(f"Unknown augmentation_type: {augmentation_type}")


class TwoCropTransform:
    """Create two augmented views of the same image for contrastive learning."""
    
    def __init__(self, base_transform):
        self.base_transform = base_transform
    
    def __call__(self, x):
        return [self.base_transform(x), self.base_transform(x)]


def get_contrastive_transform(
    augmentation_type: str = "standard",
    dataset: str = "cifar10"
):
    """
    Get contrastive learning transform that produces two views.
    
    Args:
        augmentation_type: Base augmentation type
        dataset: Dataset name
    
    Returns:
        TwoCropTransform that produces two augmented views
    """
    base_transform = get_augmentation_transform(
        augmentation_type=augmentation_type,
        dataset=dataset,
        for_contrastive=True
    )
    return TwoCropTransform(base_transform)


if __name__ == "__main__":
    # Test augmentations
    print("=" * 60)
    print("Testing Augmentation Strategies")
    print("=" * 60)
    
    # Create dummy image
    img = torch.randn(3, 32, 32)
    print(f"Original image shape: {img.shape}")
    print(f"Original range: [{img.min():.3f}, {img.max():.3f}]")
    
    # Test PatchOcclusion
    print("\nTesting PatchOcclusion:")
    patch_aug = PatchOcclusion(patch_size=(8, 8), num_patches=2)
    patched = patch_aug(img)
    print(f"  Patched shape: {patched.shape}")
    print(f"  Number of zero patches: {(patched == 0).sum().item()}")
    
    # Test GaussianNoise
    print("\nTesting GaussianNoise:")
    noise_aug = GaussianNoise(std=0.1)
    noisy = noise_aug(img)
    print(f"  Noisy shape: {noisy.shape}")
    print(f"  Noise magnitude: {(noisy - img).abs().mean():.4f}")
    
    # Test augmentation transforms
    print("\nTesting augmentation transforms:")
    for aug_type in ["none", "standard", "patch", "noise", "strong"]:
        transform = get_augmentation_transform(aug_type, dataset="cifar10")
        print(f"  {aug_type}: {len(transform.transforms)} transforms")
    
    print("\n" + "=" * 60)
    print("All augmentation tests passed!")
    print("=" * 60)
