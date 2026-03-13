"""
Linear probe classifier for Supervised Contrastive Learning.
After contrastive pretraining, a linear classifier is trained on frozen embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class LinearClassifier(nn.Module):
    """Linear probe classifier for SCL embeddings."""

    def __init__(self, input_dim: int = 512, num_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

    def predict(self, embeddings):
        with torch.no_grad():
            return self.forward(embeddings).argmax(dim=1)

    def predict_proba(self, embeddings):
        with torch.no_grad():
            return F.softmax(self.forward(embeddings), dim=1)


def train_linear_classifier(
    classifier: LinearClassifier,
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    val_embeddings: Optional[torch.Tensor] = None,
    val_labels: Optional[torch.Tensor] = None,
    epochs: int = 100,
    lr: float = 0.1,
    weight_decay: float = 1e-4,
    device: str = 'cuda'
) -> Dict[str, list]:
    """Train linear classifier on frozen embeddings using SGD."""
    classifier = classifier.to(device)
    train_embeddings = train_embeddings.to(device)
    train_labels = train_labels.to(device)

    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr,
                                momentum=0.9, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        classifier.train()
        logits = classifier(train_embeddings)
        loss = criterion(logits, train_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc = (logits.argmax(dim=1) == train_labels).float().mean().item()
        history['train_loss'].append(loss.item())
        history['train_acc'].append(train_acc)

        if val_embeddings is not None:
            classifier.eval()
            with torch.no_grad():
                val_logits = classifier(val_embeddings.to(device))
                val_acc = (val_logits.argmax(dim=1) == val_labels.to(device)).float().mean().item()
                history['val_acc'].append(val_acc)

    return history
