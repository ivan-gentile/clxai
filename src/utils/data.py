"""
CIFAR10 data loading utilities.
"""

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from typing import Tuple

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class TwoCropTransform:
    """Create two augmented views of the same image for contrastive learning."""
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return [self.base_transform(x), self.base_transform(x)]


def get_train_transforms(augment=True):
    """Standard training transforms for CIFAR10 (CE training)."""
    transform_list = []
    if augment:
        transform_list.extend([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return transforms.Compose(transform_list)


def get_test_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def get_contrastive_transforms():
    """Contrastive learning augmentations following SimCLR/SupCon."""
    return transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def get_cifar10_loaders(
    data_dir='./data',
    batch_size=128,
    num_workers=4,
    augment=True,
    contrastive=False,
) -> Tuple[DataLoader, DataLoader]:
    if contrastive:
        train_transform = TwoCropTransform(get_contrastive_transforms())
    else:
        train_transform = get_train_transforms(augment=augment)

    test_transform = get_test_transforms()

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers,
                             pin_memory=True)
    return train_loader, test_loader


def denormalize(tensor, mean=CIFAR10_MEAN, std=CIFAR10_STD):
    """Denormalize a tensor back to [0, 1] range."""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    if tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    mean, std = mean.to(tensor.device), std.to(tensor.device)
    return tensor * std + mean
