"""Extended training utilities for spline theory experiments."""

from .extended_trainer import (
    ExtendedTrainer,
    CHECKPOINT_EPOCHS,
    train_extended,
)
from .augmentation import (
    get_augmentation_transform,
    PatchOcclusion,
    GaussianNoise,
)

__all__ = [
    "ExtendedTrainer",
    "CHECKPOINT_EPOCHS",
    "train_extended",
    "get_augmentation_transform",
    "PatchOcclusion",
    "GaussianNoise",
]
