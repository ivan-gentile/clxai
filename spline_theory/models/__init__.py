"""Model variants with configurable normalization for spline theory experiments."""

from .resnet_variants import (
    get_norm_layer,
    BasicBlockVariant,
    BottleneckVariant,
    ResNet18Variant,
    ResNet152Variant,
    get_resnet_variant,
)

__all__ = [
    "get_norm_layer",
    "BasicBlockVariant",
    "BottleneckVariant",
    "ResNet18Variant",
    "ResNet152Variant",
    "get_resnet_variant",
]
