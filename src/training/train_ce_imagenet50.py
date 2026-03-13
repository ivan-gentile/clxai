"""
Cross-Entropy training for ResNet-50 on ImageNet-S50 (50-class subset).

Usage:
    python -m src.training.train_ce_imagenet50 --config configs/imagenet50/ce.yaml --seed 0
"""

import argparse
import time
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DataParallel
from tqdm import tqdm
import numpy as np
import yaml

from src.models.imagenet_resnet import get_imagenet_resnet
from src.utils.imagenet_data import get_imagenet50_loaders


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
    return {'train_loss': total_loss / len(loader), 'train_acc': 100. * correct / total}


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="Evaluating"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return {'val_loss': total_loss / len(loader), 'val_acc': 100. * correct / total}


def main():
    parser = argparse.ArgumentParser(description='ImageNet-S50 CE Training')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=160)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--optimizer', default='adamw', choices=['sgd', 'adamw', 'radam'])
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results/imagenet50/ce')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--eval_freq', type=int, default=10)
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
    n_gpus = torch.cuda.device_count()
    print(f"Device: {device}, GPUs: {n_gpus}, Seed: {args.seed}")

    output_dir = Path(args.output_dir) / f'seed{args.seed}'
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = get_imagenet50_loaders(
        data_dir=args.data_dir, batch_size=args.batch_size,
        num_workers=args.num_workers, augment=True, contrastive=False)
    _, val_loader_eval = get_imagenet50_loaders(
        data_dir=args.data_dir, batch_size=args.batch_size,
        num_workers=args.num_workers, augment=False, contrastive=False)

    model = get_imagenet_resnet(num_classes=50, encoder_only=False).to(device)
    if n_gpus > 1:
        model = DataParallel(model)

    criterion = nn.CrossEntropyLoss()

    opt_name = getattr(args, 'optimizer', 'adamw')
    if opt_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=args.weight_decay)
    elif opt_name == 'radam':
        optimizer = optim.RAdam(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)

    warmup = args.warmup_epochs
    epochs = args.epochs
    def lr_lambda(epoch):
        if epoch < warmup:
            return 0.1 + 0.9 * (epoch / warmup)
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup) / (epochs - warmup)))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_acc = 0.0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        scheduler.step()

        if epoch % args.eval_freq == 0 or epoch == epochs:
            val_metrics = evaluate(model, val_loader_eval, criterion, device)
            print(f"Epoch {epoch}: val_acc={val_metrics['val_acc']:.2f}%")
            if val_metrics['val_acc'] > best_acc:
                best_acc = val_metrics['val_acc']
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save({'epoch': epoch, 'model_state_dict': model_to_save.state_dict(),
                            'val_acc': best_acc}, output_dir / 'best_model.pt')

    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save({'epoch': epochs, 'model_state_dict': model_to_save.state_dict()},
               output_dir / 'final_model.pt')

    print(f"\nCompleted in {(time.time()-start_time)/3600:.2f}h. Best acc: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
