"""
ResNet implementations for ImageNet (224x224 images) using timm library.
Supports Supervised Contrastive Learning with projection heads.

Uses timm (PyTorch Image Models) for battle-tested ResNet implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not available. Install with: pip install timm")


class ResNet152ImageNetEncoder(nn.Module):
    """
    ResNet-152 encoder for ImageNet 224x224 images using timm backbone.
    
    Designed for contrastive learning with projection head.
    Uses timm's well-tested ResNet-152 implementation.
    
    Args:
        embedding_dim: Dimension of output embeddings (default: 128)
        pretrained: Whether to use ImageNet pretrained weights (default: False)
    """
    
    def __init__(self, embedding_dim: int = 128, pretrained: bool = False):
        super().__init__()
        
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required. Install with: pip install timm")
        
        # Create ResNet-152 backbone from timm (no classifier head)
        self.backbone = timm.create_model(
            'resnet152',
            pretrained=pretrained,
            num_classes=0,  # Remove classifier, returns pooled features
            global_pool='avg'  # Global average pooling
        )
        
        self.feature_dim = 2048  # ResNet-152 feature dimension
        
        # Projection head for contrastive learning: 2048 -> 2048 -> embedding_dim
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, embedding_dim)
        )
        
        # Initialize projection head
        self._init_projection()
    
    def _init_projection(self):
        """Initialize projection head weights."""
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before projection head (2048-dim)."""
        return self.backbone(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning L2-normalized projected embeddings."""
        features = self.forward_features(x)
        embeddings = self.projection(features)
        embeddings = F.normalize(embeddings, dim=1)
        return embeddings
    
    def get_embedding(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Get embedding (features before projection head).
        
        Args:
            x: Input tensor
            normalize: Whether to L2-normalize embeddings
        
        Returns:
            2048-dimensional feature vector
        """
        features = self.forward_features(x)
        if normalize:
            features = F.normalize(features, dim=1)
        return features


class ResNet152ImageNet(nn.Module):
    """
    ResNet-152 for ImageNet classification (Cross-Entropy training) using timm.
    
    Args:
        num_classes: Number of output classes (default: 1000 for ImageNet)
        pretrained: Whether to use ImageNet pretrained weights (default: False)
    """
    
    def __init__(self, num_classes: int = 1000, pretrained: bool = False):
        super().__init__()
        
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required. Install with: pip install timm")
        
        # Create full ResNet-152 with classifier from timm
        self.model = timm.create_model(
            'resnet152',
            pretrained=pretrained,
            num_classes=num_classes
        )
        
        self.feature_dim = 2048
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning class logits."""
        return self.model(x)
    
    def get_embedding(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Get embedding (features before classification head)."""
        features = self.model.forward_features(x)
        features = self.model.global_pool(features)
        if normalize:
            features = F.normalize(features, dim=1)
        return features
    
    def forward_with_embedding(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return both logits and normalized embeddings."""
        features = self.model.forward_features(x)
        features = self.model.global_pool(features)
        logits = self.model.fc(features)
        embeddings = F.normalize(features, dim=1)
        return logits, embeddings


class ResNet50ImageNetEncoder(nn.Module):
    """
    ResNet-50 encoder for ImageNet 224x224 images using timm backbone.
    Smaller version of ResNet-152 for faster experiments.
    
    Args:
        embedding_dim: Dimension of output embeddings (default: 128)
        pretrained: Whether to use ImageNet pretrained weights (default: False)
    """
    
    def __init__(self, embedding_dim: int = 128, pretrained: bool = False):
        super().__init__()
        
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required. Install with: pip install timm")
        
        self.backbone = timm.create_model(
            'resnet50',
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )
        
        self.feature_dim = 2048
        
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, embedding_dim)
        )
        
        self._init_projection()
    
    def _init_projection(self):
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        embeddings = self.projection(features)
        embeddings = F.normalize(embeddings, dim=1)
        return embeddings
    
    def get_embedding(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        features = self.forward_features(x)
        if normalize:
            features = F.normalize(features, dim=1)
        return features


def get_imagenet_resnet(
    architecture: str = 'resnet152',
    num_classes: int = 1000,
    encoder_only: bool = True,
    embedding_dim: int = 128,
    pretrained: bool = False
) -> nn.Module:
    """
    Factory function to create ImageNet ResNet models using timm.
    
    Args:
        architecture: 'resnet50', 'resnet152', 'resnet50_imagenet', or 'resnet152_imagenet'
        num_classes: Number of output classes (for classification)
        encoder_only: If True, return encoder for contrastive learning
        embedding_dim: Dimension of contrastive embeddings
        pretrained: Whether to use ImageNet pretrained weights
    
    Returns:
        ResNet model
    """
    if architecture in ['resnet152', 'resnet152_imagenet']:
        if encoder_only:
            return ResNet152ImageNetEncoder(embedding_dim=embedding_dim, pretrained=pretrained)
        else:
            return ResNet152ImageNet(num_classes=num_classes, pretrained=pretrained)
    elif architecture in ['resnet50', 'resnet50_imagenet']:
        if encoder_only:
            return ResNet50ImageNetEncoder(embedding_dim=embedding_dim, pretrained=pretrained)
        else:
            raise NotImplementedError("ResNet50ImageNet classifier not implemented")
    else:
        raise ValueError(f"Unknown architecture: {architecture}. Use 'resnet50' or 'resnet152'.")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test with ImageNet-sized input (224x224)
    x = torch.randn(4, 3, 224, 224).to(device)
    
    print("=" * 60)
    print("ResNet-152 ImageNet Encoder (timm backbone)")
    print("=" * 60)
    
    encoder_152 = ResNet152ImageNetEncoder(embedding_dim=128, pretrained=False).to(device)
    
    params = sum(p.numel() for p in encoder_152.parameters())
    print(f"Parameters: {params:,} ({params/1e6:.1f}M)")
    print(f"Feature dim: {encoder_152.feature_dim}")
    
    embeddings = encoder_152(x)
    print(f"Input shape: {x.shape}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding norm: {embeddings.norm(dim=1).mean():.4f} (should be ~1.0)")
    
    features = encoder_152.get_embedding(x)
    print(f"Feature shape (before projection): {features.shape}")
    
    print("\n" + "=" * 60)
    print("ResNet-152 ImageNet Classifier (timm)")
    print("=" * 60)
    
    classifier = ResNet152ImageNet(num_classes=1000, pretrained=False).to(device)
    params = sum(p.numel() for p in classifier.parameters())
    print(f"Parameters: {params:,} ({params/1e6:.1f}M)")
    
    logits = classifier(x)
    print(f"Logits shape: {logits.shape}")
    
    print("\n" + "=" * 60)
    print("ResNet-50 ImageNet Encoder (timm backbone)")
    print("=" * 60)
    
    encoder_50 = ResNet50ImageNetEncoder(embedding_dim=128, pretrained=False).to(device)
    params = sum(p.numel() for p in encoder_50.parameters())
    print(f"Parameters: {params:,} ({params/1e6:.1f}M)")
    
    embeddings = encoder_50(x)
    print(f"Embedding shape: {embeddings.shape}")
    
    print("\n" + "=" * 60)
    print("Factory Function Test")
    print("=" * 60)
    
    model = get_imagenet_resnet('resnet152', encoder_only=True)
    print(f"ResNet-152 encoder: {type(model).__name__}")
    
    model = get_imagenet_resnet('resnet152', encoder_only=False)
    print(f"ResNet-152 classifier: {type(model).__name__}")
    
    print("\nAll tests passed!")
