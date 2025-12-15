"""
Contrastive Learning loss implementations.

Includes:
- Supervised Contrastive Loss (Khosla et al., NeurIPS 2020)
- Triplet Loss with hard negative mining (Schroff et al., 2015)
- NT-Xent Loss (SimCLR)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss.
    
    For each anchor, positives are samples with the same label,
    negatives are samples with different labels.
    """
    
    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07):
        """
        Args:
            temperature: Temperature for softmax scaling
            base_temperature: Base temperature for normalization
        """
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
    
    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute SupCon loss.
        
        Args:
            features: Normalized embeddings of shape (2*B, D) where
                     first half are view1, second half are view2
            labels: Labels of shape (B,)
            mask: Optional contrastive mask of shape (B, B)
        
        Returns:
            Loss value
        """
        device = features.device
        batch_size = labels.shape[0]
        
        # Duplicate labels for both views
        labels = labels.contiguous().view(-1, 1)
        
        # Create mask: 1 where labels match, 0 otherwise
        if mask is None:
            mask = torch.eq(labels, labels.T).float().to(device)
        
        # Number of views (2 for two augmented views)
        contrast_count = features.shape[0] // batch_size
        
        # Split features into anchor and contrast
        contrast_feature = features
        anchor_feature = features
        
        # Compute similarity matrix
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )
        
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # Tile mask for multiple views
        mask = mask.repeat(contrast_count, contrast_count)
        
        # Mask out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * contrast_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        
        # Compute mean of log-likelihood over positive pairs
        # Avoid division by zero
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(
            mask_pos_pairs < 1e-6,
            torch.ones_like(mask_pos_pairs),
            mask_pos_pairs
        )
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs
        
        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(contrast_count, batch_size).mean()
        
        return loss


class SupConLossV2(nn.Module):
    """
    Simplified Supervised Contrastive Loss implementation.
    More numerically stable version.
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Concatenated features from both views [z1; z2], shape (2B, D)
            labels: Labels for one batch, shape (B,)
        """
        device = features.device
        batch_size = labels.shape[0]
        
        # Features are already L2 normalized
        # Compute all pairwise similarities
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create labels for both views
        labels_full = labels.repeat(2)
        
        # Positive mask: same label (excluding self)
        pos_mask = torch.eq(labels_full.unsqueeze(0), labels_full.unsqueeze(1)).float()
        
        # Remove diagonal (self-similarity)
        self_mask = torch.eye(2 * batch_size, device=device)
        pos_mask = pos_mask - self_mask
        
        # For numerical stability
        logits_max, _ = sim_matrix.max(dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()
        
        # Compute log softmax
        exp_logits = torch.exp(logits) * (1 - self_mask)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
        
        # Mean log prob for positive pairs
        num_positives = pos_mask.sum(dim=1)
        num_positives = torch.clamp(num_positives, min=1)
        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / num_positives
        
        # Loss is negative mean log probability
        loss = -mean_log_prob_pos.mean()
        
        return loss


class NTXentLoss(nn.Module):
    """
    NT-Xent Loss (Normalized Temperature-scaled Cross Entropy).
    Used in SimCLR for self-supervised contrastive learning.
    """
    
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1: Embeddings from view 1, shape (B, D)
            z2: Embeddings from view 2, shape (B, D)
        """
        batch_size = z1.shape[0]
        device = z1.device
        
        # Concatenate embeddings
        z = torch.cat([z1, z2], dim=0)  # (2B, D)
        
        # Compute similarity matrix
        sim = torch.matmul(z, z.T) / self.temperature  # (2B, 2B)
        
        # Create positive pair mask
        # Positive pairs: (i, i+B) and (i+B, i)
        pos_mask = torch.zeros(2 * batch_size, 2 * batch_size, device=device)
        pos_mask[:batch_size, batch_size:] = torch.eye(batch_size, device=device)
        pos_mask[batch_size:, :batch_size] = torch.eye(batch_size, device=device)
        
        # Mask out diagonal (self-similarity)
        mask = torch.eye(2 * batch_size, device=device).bool()
        sim = sim.masked_fill(mask, -float('inf'))
        
        # Compute loss
        # For each sample, positive is the augmented version
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=device),
            torch.arange(batch_size, device=device)
        ])
        
        loss = F.cross_entropy(sim, labels)
        
        return loss


class TripletLoss(nn.Module):
    """
    Triplet Loss with hard negative mining.
    
    Reference: Schroff et al. "FaceNet: A Unified Embedding for Face Recognition
               and Clustering" (CVPR 2015)
    
    For each anchor, selects:
    - Positive: sample with same label
    - Negative: sample with different label
    
    Mining strategies:
    - 'hard': hardest negative (closest to anchor)
    - 'semi-hard': negatives farther than positive but within margin
    - 'all': all valid triplets (batch-all)
    """
    
    def __init__(
        self,
        margin: float = 0.3,
        mining: str = 'hard',
        squared: bool = False
    ):
        """
        Args:
            margin: Distance margin between positive and negative pairs
            mining: Mining strategy - 'hard', 'semi-hard', or 'all'
            squared: Use squared Euclidean distance instead of Euclidean
        """
        super().__init__()
        self.margin = margin
        self.mining = mining
        self.squared = squared
        
        if mining not in ['hard', 'semi-hard', 'all']:
            raise ValueError(f"Unknown mining strategy: {mining}")
    
    def _pairwise_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise Euclidean distances.
        
        Args:
            embeddings: Tensor of shape (B, D)
        
        Returns:
            Distance matrix of shape (B, B)
        """
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
        dot_product = torch.matmul(embeddings, embeddings.T)
        square_norm = torch.diag(dot_product)
        
        # Expand to matrix form
        distances = (
            square_norm.unsqueeze(0) 
            + square_norm.unsqueeze(1) 
            - 2.0 * dot_product
        )
        
        # Ensure non-negative (numerical stability)
        distances = torch.clamp(distances, min=0.0)
        
        if not self.squared:
            # Add small epsilon before sqrt for stability
            mask = (distances == 0.0).float()
            distances = distances + mask * 1e-16
            distances = torch.sqrt(distances)
            distances = distances * (1.0 - mask)
        
        return distances
    
    def _get_anchor_positive_mask(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Get mask for valid anchor-positive pairs.
        Valid if: labels[i] == labels[j] and i != j
        """
        device = labels.device
        batch_size = labels.shape[0]
        
        # Check same labels
        labels_equal = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))
        
        # Exclude diagonal (self-pairs)
        indices_not_equal = ~torch.eye(batch_size, dtype=torch.bool, device=device)
        
        return labels_equal & indices_not_equal
    
    def _get_anchor_negative_mask(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Get mask for valid anchor-negative pairs.
        Valid if: labels[i] != labels[k]
        """
        return ~torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))
    
    def _batch_hard_triplet_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Build triplets using hardest positive and negative for each anchor.
        """
        device = embeddings.device
        batch_size = embeddings.shape[0]
        
        # Compute pairwise distances
        pairwise_dist = self._pairwise_distances(embeddings)
        
        # Get masks
        anchor_positive_mask = self._get_anchor_positive_mask(labels).float()
        anchor_negative_mask = self._get_anchor_negative_mask(labels).float()
        
        # For each anchor, get the hardest positive (farthest)
        # Multiply by mask to zero out invalid pairs
        anchor_positive_dist = anchor_positive_mask * pairwise_dist
        
        # Get max distance for each anchor
        hardest_positive_dist, _ = anchor_positive_dist.max(dim=1)
        
        # For each anchor, get the hardest negative (closest)
        # Add max value to invalid pairs so they're not selected as minimum
        max_dist = pairwise_dist.max()
        anchor_negative_dist = pairwise_dist + max_dist * (1.0 - anchor_negative_mask)
        
        # Get min distance for each anchor
        hardest_negative_dist, _ = anchor_negative_dist.min(dim=1)
        
        # Compute triplet loss
        triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
        
        # Only keep anchors that have at least one positive
        valid_anchors = anchor_positive_mask.sum(dim=1) > 0
        triplet_loss = triplet_loss[valid_anchors]
        
        if triplet_loss.numel() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        return triplet_loss.mean()
    
    def _batch_semi_hard_triplet_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Build triplets using semi-hard negatives.
        Semi-hard: negatives that are farther than the positive but within margin.
        """
        device = embeddings.device
        batch_size = embeddings.shape[0]
        
        # Compute pairwise distances
        pairwise_dist = self._pairwise_distances(embeddings)
        
        # Get masks
        anchor_positive_mask = self._get_anchor_positive_mask(labels)
        anchor_negative_mask = self._get_anchor_negative_mask(labels)
        
        # For each (anchor, positive), find semi-hard negatives
        # d(a, p) < d(a, n) < d(a, p) + margin
        
        # Get all anchor-positive distances
        ap_distances = pairwise_dist.unsqueeze(2)  # (B, B, 1)
        an_distances = pairwise_dist.unsqueeze(1)  # (B, 1, B)
        
        # Loss for all possible triplets
        loss = ap_distances - an_distances + self.margin  # (B, B, B)
        
        # Create triplet mask
        # Valid triplet: anchor_positive_mask[a, p] and anchor_negative_mask[a, n]
        ap_mask = anchor_positive_mask.unsqueeze(2)
        an_mask = anchor_negative_mask.unsqueeze(1)
        triplet_mask = ap_mask & an_mask
        
        # Semi-hard mask: d(a,n) > d(a,p) but d(a,n) < d(a,p) + margin
        semi_hard_mask = (an_distances > ap_distances) & (loss > 0)
        
        # Combined mask
        valid_triplets = triplet_mask & semi_hard_mask
        
        # Apply mask
        loss = loss * valid_triplets.float()
        
        # Count valid triplets
        num_valid = valid_triplets.sum()
        
        if num_valid == 0:
            # Fall back to hard mining if no semi-hard triplets
            return self._batch_hard_triplet_loss(embeddings, labels)
        
        return loss.sum() / num_valid.float()
    
    def _batch_all_triplet_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Build all valid triplets and average the loss.
        """
        device = embeddings.device
        
        # Compute pairwise distances
        pairwise_dist = self._pairwise_distances(embeddings)
        
        # Get masks
        anchor_positive_mask = self._get_anchor_positive_mask(labels)
        anchor_negative_mask = self._get_anchor_negative_mask(labels)
        
        # Compute loss for all triplets (B, B, B)
        ap_distances = pairwise_dist.unsqueeze(2)
        an_distances = pairwise_dist.unsqueeze(1)
        
        triplet_loss = ap_distances - an_distances + self.margin
        
        # Create valid triplet mask
        ap_mask = anchor_positive_mask.unsqueeze(2)
        an_mask = anchor_negative_mask.unsqueeze(1)
        triplet_mask = (ap_mask & an_mask).float()
        
        # Apply mask and ReLU
        triplet_loss = F.relu(triplet_loss) * triplet_mask
        
        # Count valid triplets (those with non-zero loss)
        num_positive_triplets = (triplet_loss > 1e-16).sum()
        
        if num_positive_triplets == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Average over positive triplets
        return triplet_loss.sum() / num_positive_triplets.float()
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet loss with specified mining strategy.
        
        Args:
            embeddings: Embedding vectors of shape (B, D)
            labels: Class labels of shape (B,)
        
        Returns:
            Triplet loss value
        """
        if self.mining == 'hard':
            return self._batch_hard_triplet_loss(embeddings, labels)
        elif self.mining == 'semi-hard':
            return self._batch_semi_hard_triplet_loss(embeddings, labels)
        elif self.mining == 'all':
            return self._batch_all_triplet_loss(embeddings, labels)
        else:
            raise ValueError(f"Unknown mining strategy: {self.mining}")


if __name__ == "__main__":
    # Test losses
    batch_size = 32
    dim = 128
    n_classes = 10
    
    # Random features and labels
    z1 = F.normalize(torch.randn(batch_size, dim), dim=1)
    z2 = F.normalize(torch.randn(batch_size, dim), dim=1)
    labels = torch.randint(0, n_classes, (batch_size,))
    
    # Test SupConLoss
    features = torch.cat([z1, z2], dim=0)
    supcon = SupConLoss(temperature=0.07)
    loss = supcon(features, labels)
    print(f"SupCon Loss: {loss.item():.4f}")
    
    # Test SupConLossV2
    supcon_v2 = SupConLossV2(temperature=0.07)
    loss_v2 = supcon_v2(features, labels)
    print(f"SupCon V2 Loss: {loss_v2.item():.4f}")
    
    # Test NT-Xent
    ntxent = NTXentLoss(temperature=0.5)
    loss_ntxent = ntxent(z1, z2)
    print(f"NT-Xent Loss: {loss_ntxent.item():.4f}")
    
    # Test Triplet Loss with different mining strategies
    print("\nTriplet Loss tests:")
    for mining in ['hard', 'semi-hard', 'all']:
        triplet = TripletLoss(margin=0.3, mining=mining)
        loss_triplet = triplet(z1, labels)
        print(f"  Triplet Loss ({mining}): {loss_triplet.item():.4f}")
