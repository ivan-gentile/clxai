"""
Supervised Contrastive Learning / Triplet Loss training for ResNet-18 on CIFAR10.

Two-stage training:
  1. Contrastive/metric pretraining of the encoder
  2. Linear probe on frozen embeddings (100 epochs, SGD, lr=0.1)

Usage:
    python -m src.training.train_scl --config configs/cifar10/scl.yaml --seed 0
    python -m src.training.train_scl --config configs/cifar10/triplet.yaml --seed 0
"""

import argparse
import time
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import yaml

from src.models.resnet import get_resnet18
from src.models.classifiers import LinearClassifier, train_linear_classifier
from src.training.losses import SupConLoss, TripletLoss
from src.utils.data import get_cifar10_loaders


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, loader, criterion, optimizer, device, loss_type='supcon'):
    model.train()
    total_loss = 0.0
    for images, labels in tqdm(loader, desc="Training"):
        if isinstance(images, list):
            images = torch.cat([images[0], images[1]], dim=0).to(device)
        else:
            images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        if loss_type == 'triplet':
            embeddings = model.get_embedding(images, normalize=False)
        else:
            embeddings = model(images)

        loss = criterion(embeddings, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    embeddings, labels = [], []
    for images, targets in loader:
        if isinstance(images, list):
            images = images[0]
        features = model.get_embedding(images.to(device), normalize=True)
        embeddings.append(features.cpu())
        labels.append(targets)
    return torch.cat(embeddings).numpy(), torch.cat(labels).numpy()


def main():
    parser = argparse.ArgumentParser(description='Train SCL/Triplet ResNet-18 on CIFAR10')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--loss', type=str, default='supcon', choices=['supcon', 'triplet'])
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--margin', type=float, default=0.3)
    parser.add_argument('--mining', type=str, default='semi-hard')
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='results/cifar10/scl')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--eval_freq', type=int, default=50)
    args = parser.parse_args()

    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            config = yaml.safe_load(f)
        for section in ['model', 'training', 'data', 'output']:
            if section in config:
                for k, v in config[section].items():
                    if not hasattr(args, k) or getattr(args, k) == parser.get_default(k):
                        setattr(args, k, v)
        if 'triplet' in config:
            for k, v in config['triplet'].items():
                setattr(args, k, v)

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}, Seed: {args.seed}, Loss: {args.loss}")

    output_dir = Path(args.output_dir) / f'seed{args.seed}'
    output_dir.mkdir(parents=True, exist_ok=True)

    use_contrastive = (args.loss == 'supcon')
    train_loader, test_loader = get_cifar10_loaders(
        data_dir=args.data_dir, batch_size=args.batch_size,
        num_workers=args.num_workers, contrastive=use_contrastive, augment=True)

    train_loader_eval, _ = get_cifar10_loaders(
        data_dir=args.data_dir, batch_size=args.batch_size,
        num_workers=args.num_workers, contrastive=False, augment=False)

    model = get_resnet18(encoder_only=True, embedding_dim=128).to(device)

    if args.loss == 'supcon':
        criterion = SupConLoss(temperature=args.temperature)
    else:
        criterion = TripletLoss(margin=args.margin, mining=args.mining)

    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)

    warmup = args.warmup_epochs
    epochs = args.epochs

    def lr_lambda(epoch):
        if epoch < warmup:
            return 0.1 + 0.9 * (epoch / warmup)
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup) / (epochs - warmup)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_loss = float('inf')
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, criterion, optimizer, device, args.loss)
        scheduler.step()

        if loss < best_loss:
            best_loss = loss
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'loss': best_loss}, output_dir / 'best_model.pt')

        if epoch % args.eval_freq == 0 or epoch == epochs:
            print(f"Epoch {epoch}: loss={loss:.4f}")

    torch.save({'epoch': epochs, 'model_state_dict': model.state_dict()},
               output_dir / 'final_model.pt')

    # Stage 2: Linear Probe
    print("\nTraining linear probe on frozen embeddings...")
    train_emb, train_labels = extract_embeddings(model, train_loader_eval, device)
    test_emb, test_labels = extract_embeddings(model, test_loader, device)

    linear_clf = LinearClassifier(input_dim=512, num_classes=10)
    history = train_linear_classifier(
        linear_clf,
        torch.tensor(train_emb, dtype=torch.float32),
        torch.tensor(train_labels, dtype=torch.long),
        torch.tensor(test_emb, dtype=torch.float32),
        torch.tensor(test_labels, dtype=torch.long),
        epochs=100, lr=0.1, device=str(device))

    linear_acc = history['val_acc'][-1] * 100
    print(f"Linear probe accuracy: {linear_acc:.2f}%")
    torch.save(linear_clf.state_dict(), output_dir / 'linear_classifier.pt')

    print(f"\nCompleted in {(time.time()-start_time)/60:.1f} min.")


if __name__ == "__main__":
    main()
