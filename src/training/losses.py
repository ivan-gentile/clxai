"""
Contrastive Learning loss implementations.

- Supervised Contrastive Loss (Khosla et al., NeurIPS 2020)
- Triplet Loss with semi-hard negative mining (Schroff et al., CVPR 2015)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss.
    Pulls together samples with the same label and pushes apart samples
    with different labels in the embedding space.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features: Concatenated features from both views [z1; z2], shape (2B, D)
            labels: Labels for one batch, shape (B,)
        """
        device = features.device
        batch_size = labels.shape[0]

        sim_matrix = torch.matmul(features, features.T) / self.temperature
        labels_full = labels.repeat(2)

        pos_mask = torch.eq(labels_full.unsqueeze(0), labels_full.unsqueeze(1)).float()
        self_mask = torch.eye(2 * batch_size, device=device)
        pos_mask = pos_mask - self_mask

        logits_max, _ = sim_matrix.max(dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()

        exp_logits = torch.exp(logits) * (1 - self_mask)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        num_positives = torch.clamp(pos_mask.sum(dim=1), min=1)
        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / num_positives

        return -mean_log_prob_pos.mean()


class TripletLoss(nn.Module):
    """
    Triplet Loss with configurable mining strategy.

    Mining strategies:
    - 'semi-hard': negatives farther than positive but within margin (preferred)
    - 'hard': hardest negative (closest to anchor)
    - 'all': all valid triplets
    """

    def __init__(self, margin: float = 0.3, mining: str = 'semi-hard',
                 squared: bool = False):
        super().__init__()
        self.margin = margin
        self.mining = mining
        self.squared = squared

    def _pairwise_distances(self, embeddings):
        dot_product = torch.matmul(embeddings, embeddings.T)
        square_norm = torch.diag(dot_product)
        distances = square_norm.unsqueeze(0) + square_norm.unsqueeze(1) - 2.0 * dot_product
        distances = torch.clamp(distances, min=0.0)

        if not self.squared:
            mask = (distances == 0.0).float()
            distances = distances + mask * 1e-16
            distances = torch.sqrt(distances)
            distances = distances * (1.0 - mask)
        return distances

    def _get_anchor_positive_mask(self, labels):
        device = labels.device
        batch_size = labels.shape[0]
        labels_equal = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))
        indices_not_equal = ~torch.eye(batch_size, dtype=torch.bool, device=device)
        return labels_equal & indices_not_equal

    def _get_anchor_negative_mask(self, labels):
        return ~torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))

    def _batch_hard_triplet_loss(self, embeddings, labels):
        pairwise_dist = self._pairwise_distances(embeddings)
        anchor_positive_mask = self._get_anchor_positive_mask(labels).float()
        anchor_negative_mask = self._get_anchor_negative_mask(labels).float()

        anchor_positive_dist = anchor_positive_mask * pairwise_dist
        hardest_positive_dist, _ = anchor_positive_dist.max(dim=1)

        max_dist = pairwise_dist.max()
        anchor_negative_dist = pairwise_dist + max_dist * (1.0 - anchor_negative_mask)
        hardest_negative_dist, _ = anchor_negative_dist.min(dim=1)

        triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
        valid_anchors = anchor_positive_mask.sum(dim=1) > 0
        triplet_loss = triplet_loss[valid_anchors]

        if triplet_loss.numel() == 0:
            return (embeddings * 0).sum()
        return triplet_loss.mean()

    def _batch_semi_hard_triplet_loss(self, embeddings, labels):
        pairwise_dist = self._pairwise_distances(embeddings)
        anchor_positive_mask = self._get_anchor_positive_mask(labels)
        anchor_negative_mask = self._get_anchor_negative_mask(labels)

        ap_distances = pairwise_dist.unsqueeze(2)
        an_distances = pairwise_dist.unsqueeze(1)
        loss = ap_distances - an_distances + self.margin

        ap_mask = anchor_positive_mask.unsqueeze(2)
        an_mask = anchor_negative_mask.unsqueeze(1)
        triplet_mask = ap_mask & an_mask

        semi_hard_mask = (an_distances > ap_distances) & (loss > 0)
        valid_triplets = triplet_mask & semi_hard_mask
        loss = loss * valid_triplets.float()
        num_valid = valid_triplets.sum()

        if num_valid == 0:
            return self._batch_hard_triplet_loss(embeddings, labels)
        return loss.sum() / num_valid.float()

    def forward(self, embeddings, labels):
        if self.mining == 'hard':
            return self._batch_hard_triplet_loss(embeddings, labels)
        elif self.mining == 'semi-hard':
            return self._batch_semi_hard_triplet_loss(embeddings, labels)
        else:
            raise ValueError(f"Unknown mining strategy: {self.mining}")
