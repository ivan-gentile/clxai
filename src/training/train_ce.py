"""
Cross-Entropy training for ResNet-18 on CIFAR10.

Usage:
    python -m src.training.train_ce --epochs 200 --lr 0.1 --batch_size 128 --seed 0
"""

import argparse
import time
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import yaml

from src.models.resnet import get_resnet18
from src.utils.data import get_cifar10_loaders


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        pbar.set_postfix(loss=f'{loss.item():.4f}', acc=f'{100.*correct/total:.2f}%')

    return total_loss / len(loader), 100. * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return total_loss / len(loader), 100. * correct / total


def main():
    parser = argparse.ArgumentParser(description='Train CE ResNet-18 on CIFAR10')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='results/cifar10/ce')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_freq', type=int, default=50)
    args = parser.parse_args()

    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            config = yaml.safe_load(f)
        for section in ['model', 'training', 'data', 'output']:
            if section in config:
                for k, v in config[section].items():
                    if not hasattr(args, k) or getattr(args, k) == parser.get_default(k):
                        setattr(args, k, v)

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}, Seed: {args.seed}")

    output_dir = Path(args.output_dir) / f'seed{args.seed}'
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = get_cifar10_loaders(
        data_dir=args.data_dir, batch_size=args.batch_size,
        num_workers=args.num_workers, augment=True)

    model = get_resnet18(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion,
                                             optimizer, device, epoch)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch}: train_acc={train_acc:.2f}%, test_acc={test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'test_acc': best_acc}, output_dir / 'best_model.pt')

        if epoch % args.save_freq == 0:
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'test_acc': test_acc}, output_dir / f'checkpoint_epoch_{epoch}.pt')

    torch.save({'epoch': args.epochs, 'model_state_dict': model.state_dict(),
                'test_acc': test_acc}, output_dir / 'final_model.pt')

    print(f"\nCompleted in {(time.time()-start_time)/60:.1f} min. Best acc: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
