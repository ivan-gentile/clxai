"""
ImageNet 50-class subset data loading utilities.
Filters ImageNet to 50 specific classes and remaps labels to 0-49.

The 50 classes follow the ImageNet-S50 subset from Gao et al. (TPAMI 2022).
"""

import os
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from typing import Tuple, Optional, Callable, List
from datasets import load_from_disk

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SELECTED_CLASSES = [
    1, 3, 11, 31, 222, 277, 284, 295, 301, 325,
    330, 333, 342, 368, 386, 388, 404, 412, 418, 436,
    449, 466, 487, 492, 502, 510, 531, 532, 574, 579,
    606, 617, 659, 670, 695, 703, 748, 829, 846, 851,
    861, 879, 883, 898, 900, 914, 919, 951, 959, 992
]

CLASS_MAPPING = {orig: new for new, orig in enumerate(SELECTED_CLASSES)}


class TwoCropTransform:
    """Create two augmented views of the same image for contrastive learning."""
    def __init__(self, base_transform):
        self.base_transform = base_transform
    def __call__(self, x):
        return [self.base_transform(x), self.base_transform(x)]


class GaussianBlur:
    """Gaussian blur augmentation (SimCLR/SupCon)."""
    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma
    def __call__(self, x):
        import random
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        from torchvision.transforms import functional as F
        return F.gaussian_blur(x, kernel_size=23, sigma=sigma)


class ImageNet50HFDataset(Dataset):
    """PyTorch Dataset wrapper for HuggingFace ImageNet with 50-class filtering."""

    def __init__(self, hf_dataset, transform=None,
                 selected_classes=SELECTED_CLASSES,
                 class_mapping=CLASS_MAPPING):
        self.dataset = hf_dataset
        self.transform = transform
        self.selected_classes = set(selected_classes)
        self.class_mapping = class_mapping

        self.valid_indices = [
            idx for idx in range(len(hf_dataset))
            if hf_dataset[idx]['label'] in self.selected_classes
        ]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        item = self.dataset[self.valid_indices[idx]]
        image = item['image']
        if image.mode != 'RGB':
            image = image.convert('RGB')
        new_label = self.class_mapping[item['label']]
        if self.transform:
            image = self.transform(image)
        return image, new_label


def get_imagenet50_train_transforms(augment=True, image_size=224):
    transform_list = []
    if augment:
        transform_list.extend([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
        ])
    else:
        transform_list.extend([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
        ])
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return transforms.Compose(transform_list)


def get_imagenet50_test_transforms(image_size=224):
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_imagenet50_contrastive_transforms(image_size=224):
    """Contrastive transforms following SupCon paper (Khosla et al., 2020)."""
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.RandomApply([GaussianBlur(sigma=(0.1, 2.0))], p=0.5),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_imagenet50_loaders(
    data_dir,
    batch_size=128,
    num_workers=8,
    augment=True,
    contrastive=False,
) -> Tuple[DataLoader, DataLoader]:
    train_hf = load_from_disk(os.path.join(data_dir, 'train'))
    val_hf = load_from_disk(os.path.join(data_dir, 'validation'))

    if contrastive:
        train_transform = TwoCropTransform(get_imagenet50_contrastive_transforms())
    else:
        train_transform = get_imagenet50_train_transforms(augment=augment)

    val_transform = get_imagenet50_test_transforms()

    train_dataset = ImageNet50HFDataset(train_hf, transform=train_transform)
    val_dataset = ImageNet50HFDataset(val_hf, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True,
                              persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            persistent_workers=num_workers > 0)
    return train_loader, val_loader
