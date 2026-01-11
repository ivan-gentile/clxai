"""
Custom data augmentations for CLXAI.

Includes:
- RandomPatchRemoval: Rectangular patch removal augmentation (our approach)
- RandomPixelRemoval: F-Fidelity style scattered pixel removal (ICLR 2025)
- GaussianNoiseAugmentation: Additive Gaussian noise for perturbation robustness

References:
- F-Fidelity: Zheng et al., "F-Fidelity: A Robust Framework for Faithfulness 
  Evaluation of Explainable AI", ICLR 2025. https://arxiv.org/abs/2410.02970
"""

import random
import numpy as np
import torch
from typing import Tuple, Optional

# CIFAR-10 statistics for replacement values
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)


class RandomPatchRemoval:
    """
    F-Fidelity style random patch removal augmentation.
    
    Randomly removes rectangular patches from images by replacing them
    with a neutral value (dataset mean or black).
    
    This makes models robust to occlusion-based perturbations used in
    XAI faithfulness evaluation (pixel flipping).
    
    Args:
        probability: Probability of applying augmentation (default: 0.5)
        patch_sizes: List of patch sizes as fractions of image dimension
                    (default: [1/4, 1/8, 1/16] = [8, 4, 2] pixels for CIFAR-10)
        replacement: 'mean' for dataset mean, 'black' for zeros (default: 'mean')
        dataset: 'cifar10' or 'cifar100' for mean values (default: 'cifar10')
    """
    
    def __init__(
        self,
        probability: float = 0.5,
        patch_sizes: Tuple[float, ...] = (1/4, 1/8, 1/16),
        replacement: str = 'mean',
        dataset: str = 'cifar10'
    ):
        self.probability = probability
        self.patch_sizes = patch_sizes
        self.replacement = replacement
        
        if dataset == 'cifar100':
            self.mean = torch.tensor(CIFAR100_MEAN).view(3, 1, 1)
        else:
            self.mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply random patch removal to image.
        
        Args:
            img: Tensor of shape (C, H, W) in range [0, 1]
        
        Returns:
            Augmented tensor
        """
        if random.random() >= self.probability:
            return img
        
        C, H, W = img.shape
        
        # Select random patch size
        patch_fraction = random.choice(self.patch_sizes)
        patch_h = int(H * patch_fraction)
        patch_w = int(W * patch_fraction)
        
        # Ensure at least 1 pixel
        patch_h = max(1, patch_h)
        patch_w = max(1, patch_w)
        
        # Random position
        y = random.randint(0, H - patch_h)
        x = random.randint(0, W - patch_w)
        
        # Clone to avoid modifying original
        img = img.clone()
        
        # Apply replacement
        if self.replacement == 'mean':
            # Ensure mean is on same device as image
            mean = self.mean.to(img.device)
            img[:, y:y+patch_h, x:x+patch_w] = mean
        else:  # black
            img[:, y:y+patch_h, x:x+patch_w] = 0.0
        
        return img
    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'probability={self.probability}, '
                f'patch_sizes={self.patch_sizes}, '
                f'replacement={self.replacement})')


class RandomPixelRemoval:
    """
    F-Fidelity style random pixel removal augmentation (ICLR 2025).
    
    Randomly removes SCATTERED INDIVIDUAL PIXELS from images by setting them to 0.
    This is the exact implementation from the F-Fidelity paper, which differs from
    our RandomPatchRemoval that removes contiguous rectangular patches.
    
    Key differences from RandomPatchRemoval:
    - Removes individual scattered pixels, not contiguous patches
    - Fixed fraction (not probability-based per image)
    - Replaces with 0 (black), not dataset mean
    - Applied to 100% of images during fine-tuning
    
    Reference:
        Zheng et al., "F-Fidelity: A Robust Framework for Faithfulness 
        Evaluation of Explainable AI", ICLR 2025.
        https://github.com/AslanDing/Finetune-Fidelity
    
    Args:
        removal_fraction: Fraction of pixels to remove (default: 0.1 = 10%, β in paper)
        probability: Probability of applying augmentation (default: 1.0 for F-Fidelity)
    """
    
    def __init__(
        self,
        removal_fraction: float = 0.1,
        probability: float = 1.0
    ):
        self.removal_fraction = removal_fraction
        self.probability = probability
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply random pixel removal to image.
        
        Args:
            tensor: Tensor of shape (C, H, W) in range [0, 1]
        
        Returns:
            Augmented tensor with scattered pixels set to 0
        """
        if random.random() >= self.probability:
            return tensor
        
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected input type torch.Tensor, but got {type(tensor)}")
        
        C, H, W = tensor.shape
        num_pixels = H * W
        num_pixels_to_remove = int(self.removal_fraction * num_pixels)
        
        # Create mask and select random pixels to remove
        mask = torch.ones(H, W, dtype=tensor.dtype, device=tensor.device)
        indices = np.random.choice(num_pixels, num_pixels_to_remove, replace=False)
        mask.view(-1)[indices] = 0
        
        # Apply mask (broadcasts across channels)
        tensor = tensor * mask.unsqueeze(0)
        
        return tensor
    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'removal_fraction={self.removal_fraction}, '
                f'probability={self.probability})')


class GaussianNoiseAugmentation:
    """
    Additive Gaussian noise augmentation.
    
    Adds random Gaussian noise to images to make models robust to
    continuous perturbations used in XAI faithfulness evaluation.
    
    Args:
        probability: Probability of applying augmentation (default: 0.5)
        std: Standard deviation of Gaussian noise (default: 0.05)
        mean: Mean of Gaussian noise (default: 0.0)
    """
    
    def __init__(
        self,
        probability: float = 0.5,
        std: float = 0.05,
        mean: float = 0.0
    ):
        self.probability = probability
        self.std = std
        self.mean = mean
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply Gaussian noise to image.
        
        Args:
            img: Tensor of shape (C, H, W) in range [0, 1]
        
        Returns:
            Augmented tensor (clamped to [0, 1])
        """
        if random.random() >= self.probability:
            return img
        
        # Generate noise
        noise = torch.randn_like(img) * self.std + self.mean
        
        # Add noise and clamp to valid range
        img = img + noise
        img = torch.clamp(img, 0.0, 1.0)
        
        return img
    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'probability={self.probability}, '
                f'std={self.std}, '
                f'mean={self.mean})')


class CombinedAugmentation:
    """
    Wrapper to apply a custom augmentation after ToTensor but before Normalize.
    
    This is useful for augmentations that operate on tensor values in [0, 1].
    """
    
    def __init__(self, augmentation):
        self.augmentation = augmentation
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return self.augmentation(img)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.augmentation})'


if __name__ == "__main__":
    # Test augmentations
    print("Testing RandomPatchRemoval (our approach)...")
    patch_aug = RandomPatchRemoval(probability=1.0)  # Always apply for testing
    
    # Create dummy image (3, 32, 32)
    img = torch.rand(3, 32, 32)
    aug_img = patch_aug(img)
    print(f"  Input shape: {img.shape}")
    print(f"  Output shape: {aug_img.shape}")
    print(f"  Values changed: {not torch.allclose(img, aug_img)}")
    
    print("\nTesting RandomPixelRemoval (F-Fidelity style)...")
    pixel_aug = RandomPixelRemoval(removal_fraction=0.1, probability=1.0)
    
    img = torch.rand(3, 32, 32)
    aug_img = pixel_aug(img)
    num_zeros = (aug_img == 0).sum().item()
    expected_zeros = int(0.1 * 32 * 32 * 3)  # 10% of all values (3 channels)
    print(f"  Input shape: {img.shape}")
    print(f"  Output shape: {aug_img.shape}")
    print(f"  Zero values: {num_zeros} (expected ~{expected_zeros} per channel)")
    print(f"  Removal fraction: {num_zeros / (32 * 32 * 3):.2%}")
    print(f"  Values changed: {not torch.allclose(img, aug_img)}")
    
    print("\nTesting GaussianNoiseAugmentation...")
    noise_aug = GaussianNoiseAugmentation(probability=1.0, std=0.05)
    
    img = torch.rand(3, 32, 32)
    aug_img = noise_aug(img)
    print(f"  Input shape: {img.shape}")
    print(f"  Output shape: {aug_img.shape}")
    print(f"  Output range: [{aug_img.min():.4f}, {aug_img.max():.4f}]")
    print(f"  Values changed: {not torch.allclose(img, aug_img)}")
    
    print("\nAugmentations test completed!")
