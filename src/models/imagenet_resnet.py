"""
ResNet-50 for ImageNet-S50 (224x224 images) using timm.
Supports both Cross-Entropy and Supervised Contrastive Learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import timm


class ResNet50ImageNet(nn.Module):
    """ResNet-50 for classification (Cross-Entropy training)."""

    def __init__(self, num_classes: int = 50, pretrained: bool = False):
        super().__init__()
        self.model = timm.create_model('resnet50', pretrained=pretrained,
                                       num_classes=num_classes)
        self.feature_dim = 2048

    def forward(self, x):
        return self.model(x)

    def get_embedding(self, x, normalize=True):
        features = self.model.forward_features(x)
        features = self.model.global_pool(features)
        if normalize:
            features = F.normalize(features, dim=1)
        return features

    def forward_with_embedding(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.model.forward_features(x)
        features = self.model.global_pool(features)
        logits = self.model.fc(features)
        embeddings = F.normalize(features, dim=1)
        return logits, embeddings


class ResNet50ImageNetEncoder(nn.Module):
    """ResNet-50 encoder for contrastive learning with projection head."""

    def __init__(self, embedding_dim: int = 128, pretrained: bool = False):
        super().__init__()
        self.backbone = timm.create_model('resnet50', pretrained=pretrained,
                                          num_classes=0, global_pool='avg')
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

    def forward_features(self, x):
        return self.backbone(x)

    def forward(self, x):
        features = self.forward_features(x)
        embeddings = self.projection(features)
        return F.normalize(embeddings, dim=1)

    def get_embedding(self, x, normalize=True):
        features = self.forward_features(x)
        if normalize:
            features = F.normalize(features, dim=1)
        return features


def get_imagenet_resnet(architecture='resnet50', num_classes=50,
                        encoder_only=True, embedding_dim=128,
                        pretrained=False):
    if encoder_only:
        return ResNet50ImageNetEncoder(embedding_dim=embedding_dim,
                                       pretrained=pretrained)
    return ResNet50ImageNet(num_classes=num_classes, pretrained=pretrained)
