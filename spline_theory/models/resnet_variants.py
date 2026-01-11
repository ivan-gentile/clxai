"""
ResNet variants with configurable normalization layers.

Supports:
- BatchNorm (BN) - standard
- GroupNorm (GN) - groups=32
- LayerNorm (LN) - implemented as GroupNorm with 1 group
- Identity (ID) - no normalization

This enables ablation studies testing the hypothesis that BatchNorm removal
helps CE models converge to CL-like geometry through grokking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Literal

NormType = Literal["bn", "gn", "ln", "id"]


def get_norm_layer(norm_type: NormType, num_features: int) -> nn.Module:
    """
    Get normalization layer based on type.
    
    Args:
        norm_type: Type of normalization ('bn', 'gn', 'ln', 'id')
        num_features: Number of features/channels
    
    Returns:
        Normalization layer module
    """
    if norm_type == "bn":
        return nn.BatchNorm2d(num_features)
    elif norm_type == "gn":
        # GroupNorm with 32 groups (standard for ResNet)
        # Handle case where num_features < 32
        num_groups = min(32, num_features)
        return nn.GroupNorm(num_groups, num_features)
    elif norm_type == "ln":
        # LayerNorm equivalent: GroupNorm with 1 group
        return nn.GroupNorm(1, num_features)
    elif norm_type == "id":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}. Use 'bn', 'gn', 'ln', or 'id'.")


class BasicBlockVariant(nn.Module):
    """
    Basic residual block with configurable normalization.
    Used in ResNet-18 and ResNet-34.
    """
    
    expansion = 1
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        norm_type: NormType = "bn"
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=(norm_type == "id")
        )
        self.norm1 = get_norm_layer(norm_type, out_channels)
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=(norm_type == "id")
        )
        self.norm2 = get_norm_layer(norm_type, out_channels)
        
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class BottleneckVariant(nn.Module):
    """
    Bottleneck residual block with configurable normalization.
    Used in ResNet-50, ResNet-101, ResNet-152.
    """
    
    expansion = 4
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        norm_type: NormType = "bn"
    ):
        super().__init__()
        
        # 1x1 conv to reduce channels
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=(norm_type == "id")
        )
        self.norm1 = get_norm_layer(norm_type, out_channels)
        
        # 3x3 conv
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=(norm_type == "id")
        )
        self.norm2 = get_norm_layer(norm_type, out_channels)
        
        # 1x1 conv to expand channels
        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion,
            kernel_size=1, bias=(norm_type == "id")
        )
        self.norm3 = get_norm_layer(norm_type, out_channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.norm3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNet18Variant(nn.Module):
    """
    ResNet-18 with configurable normalization for classification.
    Adapted for CIFAR-10/100 (32x32 input).
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
        norm_type: NormType = "bn",
        embedding_dim: int = 128
    ):
        super().__init__()
        
        self.norm_type = norm_type
        self.in_planes = 64
        self.feature_dim = 512
        
        # Initial conv layer (adapted for CIFAR 32x32 input)
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3,
            stride=1, padding=1, bias=(norm_type == "id")
        )
        self.norm1 = get_norm_layer(norm_type, 64)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        # Projection head for contrastive learning (optional use)
        self.projection = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embedding_dim)
        )
        
        self._initialize_weights()
    
    def _make_layer(
        self,
        out_channels: int,
        num_blocks: int,
        stride: int
    ) -> nn.Sequential:
        downsample = None
        
        if stride != 1 or self.in_planes != out_channels:
            downsample_layers = [
                nn.Conv2d(
                    self.in_planes, out_channels,
                    kernel_size=1, stride=stride, bias=(self.norm_type == "id")
                )
            ]
            if self.norm_type != "id":
                downsample_layers.append(get_norm_layer(self.norm_type, out_channels))
            downsample = nn.Sequential(*downsample_layers)
        
        layers = []
        layers.append(BasicBlockVariant(
            self.in_planes, out_channels, stride, downsample, self.norm_type
        ))
        self.in_planes = out_channels
        
        for _ in range(1, num_blocks):
            layers.append(BasicBlockVariant(
                out_channels, out_channels, norm_type=self.norm_type
            ))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification head."""
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning class logits."""
        features = self.forward_features(x)
        logits = self.fc(features)
        return logits
    
    def get_embedding(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Get embedding (features before classification head)."""
        features = self.forward_features(x)
        if normalize:
            features = F.normalize(features, dim=1)
        return features
    
    def get_projected_embedding(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Get projected embedding for contrastive learning."""
        features = self.forward_features(x)
        embeddings = self.projection(features)
        if normalize:
            embeddings = F.normalize(embeddings, dim=1)
        return embeddings
    
    def forward_with_embedding(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return both logits and embeddings."""
        features = self.forward_features(x)
        logits = self.fc(features)
        embeddings = F.normalize(features, dim=1)
        return logits, embeddings


class ResNet152Variant(nn.Module):
    """
    ResNet-152 with configurable normalization for classification.
    Uses Bottleneck blocks with layer config [3, 8, 36, 3].
    Adapted for CIFAR-10/100 (32x32 input).
    """
    
    def __init__(
        self,
        num_classes: int = 100,
        in_channels: int = 3,
        norm_type: NormType = "bn",
        embedding_dim: int = 128
    ):
        super().__init__()
        
        self.norm_type = norm_type
        self.in_planes = 64
        self.feature_dim = 512 * BottleneckVariant.expansion  # 2048
        
        # Initial conv layer (adapted for CIFAR 32x32 input)
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3,
            stride=1, padding=1, bias=(norm_type == "id")
        )
        self.norm1 = get_norm_layer(norm_type, 64)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual blocks: [3, 8, 36, 3] for ResNet-152
        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 8, stride=2)
        self.layer3 = self._make_layer(256, 36, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.feature_dim, num_classes)
        
        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, embedding_dim)
        )
        
        self._initialize_weights()
    
    def _make_layer(
        self,
        out_channels: int,
        num_blocks: int,
        stride: int
    ) -> nn.Sequential:
        downsample = None
        
        if stride != 1 or self.in_planes != out_channels * BottleneckVariant.expansion:
            downsample_layers = [
                nn.Conv2d(
                    self.in_planes, out_channels * BottleneckVariant.expansion,
                    kernel_size=1, stride=stride, bias=(self.norm_type == "id")
                )
            ]
            if self.norm_type != "id":
                downsample_layers.append(
                    get_norm_layer(self.norm_type, out_channels * BottleneckVariant.expansion)
                )
            downsample = nn.Sequential(*downsample_layers)
        
        layers = []
        layers.append(BottleneckVariant(
            self.in_planes, out_channels, stride, downsample, self.norm_type
        ))
        self.in_planes = out_channels * BottleneckVariant.expansion
        
        for _ in range(1, num_blocks):
            layers.append(BottleneckVariant(
                self.in_planes, out_channels, norm_type=self.norm_type
            ))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification head."""
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning class logits."""
        features = self.forward_features(x)
        logits = self.fc(features)
        return logits
    
    def get_embedding(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Get embedding (features before classification head)."""
        features = self.forward_features(x)
        if normalize:
            features = F.normalize(features, dim=1)
        return features
    
    def get_projected_embedding(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Get projected embedding for contrastive learning."""
        features = self.forward_features(x)
        embeddings = self.projection(features)
        if normalize:
            embeddings = F.normalize(embeddings, dim=1)
        return embeddings
    
    def forward_with_embedding(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return both logits and embeddings."""
        features = self.forward_features(x)
        logits = self.fc(features)
        embeddings = F.normalize(features, dim=1)
        return logits, embeddings


class ResNetEncoderVariant(nn.Module):
    """
    ResNet encoder (without classification head) with configurable normalization.
    For contrastive learning experiments.
    """
    
    def __init__(
        self,
        architecture: str = "resnet18",
        embedding_dim: int = 128,
        in_channels: int = 3,
        norm_type: NormType = "bn"
    ):
        super().__init__()
        
        if architecture == "resnet18":
            self.encoder = ResNet18Variant(
                num_classes=10,  # Placeholder, not used
                in_channels=in_channels,
                norm_type=norm_type,
                embedding_dim=embedding_dim
            )
            self.feature_dim = 512
        elif architecture == "resnet152":
            self.encoder = ResNet152Variant(
                num_classes=100,  # Placeholder
                in_channels=in_channels,
                norm_type=norm_type,
                embedding_dim=embedding_dim
            )
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        # Remove classification head - we only want the encoder
        self.encoder.fc = nn.Identity()
        self.projection = self.encoder.projection
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning projected embeddings."""
        features = self.encoder.forward_features(x)
        embeddings = self.projection(features)
        embeddings = F.normalize(embeddings, dim=1)
        return embeddings
    
    def get_embedding(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Get embedding (features before projection)."""
        features = self.encoder.forward_features(x)
        if normalize:
            features = F.normalize(features, dim=1)
        return features


def get_resnet_variant(
    architecture: str = "resnet18",
    num_classes: int = 10,
    norm_type: NormType = "bn",
    encoder_only: bool = False,
    embedding_dim: int = 128
) -> nn.Module:
    """
    Factory function to create ResNet variant with configurable normalization.
    
    Args:
        architecture: 'resnet18' or 'resnet152'
        num_classes: Number of output classes
        norm_type: Normalization type ('bn', 'gn', 'ln', 'id')
        encoder_only: If True, return encoder for contrastive learning
        embedding_dim: Dimension of contrastive embeddings
    
    Returns:
        ResNet model with specified configuration
    """
    if encoder_only:
        return ResNetEncoderVariant(
            architecture=architecture,
            embedding_dim=embedding_dim,
            norm_type=norm_type
        )
    
    if architecture == "resnet18":
        return ResNet18Variant(
            num_classes=num_classes,
            norm_type=norm_type,
            embedding_dim=embedding_dim
        )
    elif architecture == "resnet152":
        return ResNet152Variant(
            num_classes=num_classes,
            norm_type=norm_type,
            embedding_dim=embedding_dim
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}. Use 'resnet18' or 'resnet152'.")


if __name__ == "__main__":
    # Test all normalization variants
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(4, 3, 32, 32).to(device)
    
    print("=" * 60)
    print("Testing ResNet-18 Variants")
    print("=" * 60)
    
    for norm_type in ["bn", "gn", "ln", "id"]:
        model = get_resnet_variant(
            architecture="resnet18",
            num_classes=10,
            norm_type=norm_type
        ).to(device)
        
        logits = model(x)
        embeddings = model.get_embedding(x)
        
        params = sum(p.numel() for p in model.parameters())
        print(f"Norm={norm_type:2s}: logits={logits.shape}, emb={embeddings.shape}, params={params:,}")
    
    print("\n" + "=" * 60)
    print("Testing ResNet-18 Encoder Variants (for contrastive learning)")
    print("=" * 60)
    
    for norm_type in ["bn", "gn", "ln", "id"]:
        encoder = get_resnet_variant(
            architecture="resnet18",
            norm_type=norm_type,
            encoder_only=True,
            embedding_dim=128
        ).to(device)
        
        embeddings = encoder(x)
        features = encoder.get_embedding(x)
        
        print(f"Norm={norm_type:2s}: projected_emb={embeddings.shape}, features={features.shape}")
    
    print("\n" + "=" * 60)
    print("Testing ResNet-152 Variants")
    print("=" * 60)
    
    for norm_type in ["bn", "id"]:  # Skip gn/ln for speed
        model = get_resnet_variant(
            architecture="resnet152",
            num_classes=100,
            norm_type=norm_type
        ).to(device)
        
        logits = model(x)
        embeddings = model.get_embedding(x)
        
        params = sum(p.numel() for p in model.parameters())
        print(f"Norm={norm_type:2s}: logits={logits.shape}, emb={embeddings.shape}, params={params:,}")
