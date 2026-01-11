"""
Cross-Entropy training for ResNet on CIFAR-10/100.
Supports ResNet-18 and ResNet-152 architectures.
"""

import os
import sys
import argparse
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.resnet import get_model
from src.utils.data import get_data_loaders, get_num_classes

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(
    model: nn.Module,
    train_loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
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
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })
    
    return {
        'train_loss': total_loss / len(train_loader),
        'train_acc': 100. * correct / total
    }


def evaluate(
    model: nn.Module,
    test_loader,
    criterion: nn.Module,
    device: torch.device
) -> dict:
    """Evaluate model on test set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return {
        'test_loss': total_loss / len(test_loader),
        'test_acc': 100. * correct / total
    }


def train_ce_model(config: dict):
    """
    Main training function for CE model.
    
    Args:
        config: Training configuration dictionary
    """
    # Set seed for reproducibility
    seed = config.get('seed', 42)
    set_seed(seed)
    print(f"Using seed: {seed}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get dataset and architecture from config
    dataset = config.get('dataset', 'cifar10')
    architecture = config.get('architecture', 'resnet18')
    num_classes = get_num_classes(dataset)
    
    print(f"Dataset: {dataset} ({num_classes} classes)")
    print(f"Architecture: {architecture}")
    
    # Create output directory
    output_dir = Path(config.get('output_dir', 'results/models/ce_baseline'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb if available
    if WANDB_AVAILABLE and config.get('use_wandb', True):
        wandb.init(
            project=config.get('wandb_project', 'clxai'),
            name=config.get('run_name', 'ce_baseline'),
            config=config
        )
    
    # Get augmentation type
    augmentation_type = config.get('augmentation_type', 'none')
    print(f"Augmentation type: {augmentation_type}")
    
    # Data loaders
    train_loader, test_loader = get_data_loaders(
        dataset=dataset,
        data_dir=config.get('data_dir', './data'),
        batch_size=config.get('batch_size', 128),
        num_workers=config.get('num_workers', 4),
        augment=True,
        augmentation_type=augmentation_type
    )
    
    # Model
    model = get_model(
        architecture=architecture,
        num_classes=num_classes,
        encoder_only=False
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.get('lr', 0.1),
        momentum=config.get('momentum', 0.9),
        weight_decay=config.get('weight_decay', 5e-4)
    )
    
    # Scheduler
    epochs = config.get('epochs', 200)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    best_acc = 0.0
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Evaluate
        test_metrics = evaluate(model, test_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Log metrics
        metrics = {**train_metrics, **test_metrics, 'lr': scheduler.get_last_lr()[0]}
        print(f"Epoch {epoch}: train_acc={train_metrics['train_acc']:.2f}%, "
              f"test_acc={test_metrics['test_acc']:.2f}%")
        
        if WANDB_AVAILABLE and config.get('use_wandb', True):
            wandb.log(metrics, step=epoch)
        
        # Save best model
        if test_metrics['test_acc'] > best_acc:
            best_acc = test_metrics['test_acc']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': best_acc,
                'config': config
            }, output_dir / 'best_model.pt')
        
        # Save checkpoint periodically
        if epoch % config.get('save_freq', 50) == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_metrics['test_acc'],
                'config': config
            }, output_dir / f'checkpoint_epoch_{epoch}.pt')
    
    # Final save
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_acc': test_metrics['test_acc'],
        'config': config
    }, output_dir / 'final_model.pt')
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes")
    print(f"Best test accuracy: {best_acc:.2f}%")
    
    if WANDB_AVAILABLE and config.get('use_wandb', True):
        wandb.finish()
    
    return model, best_acc


def main():
    parser = argparse.ArgumentParser(description='Train CE ResNet on CIFAR-10/100')
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'],
                        help='Dataset to use')
    parser.add_argument('--architecture', type=str, default='resnet18', choices=['resnet18', 'resnet152'],
                        help='Model architecture')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='results/models/ce_baseline')
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--run_name', type=str, default='ce_baseline')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--augmentation_type', type=str, default='none',
                        choices=['none', 'patch', 'noise', 'pixel', 'pixel50'],
                        help='Augmentation type: none, patch, noise, pixel (F-Fidelity 100%%), or pixel50 (F-Fidelity 50%%)')
    args = parser.parse_args()
    
    # Load config from file or use defaults
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        # Flatten nested config sections
        if 'model' in config:
            config['architecture'] = config['model'].get('architecture', 'resnet18')
            config['num_classes'] = config['model'].get('num_classes', 10)
        if 'training' in config:
            for k, v in config['training'].items():
                config[k] = v
        if 'data' in config:
            for k, v in config['data'].items():
                config[k] = v
        if 'logging' in config:
            for k, v in config['logging'].items():
                config[k] = v
        if 'output' in config:
            for k, v in config['output'].items():
                config[k] = v
        # Handle augmentation config section
        if 'augmentation' in config:
            config['augmentation_type'] = config['augmentation'].get('type', 'none')
        # Override with command line args if provided
        if args.seed != 42:
            config['seed'] = args.seed
        if args.output_dir != 'results/models/ce_baseline':
            config['output_dir'] = args.output_dir
        if args.run_name != 'ce_baseline':
            config['run_name'] = args.run_name
        if args.dataset != 'cifar10':
            config['dataset'] = args.dataset
        if args.architecture != 'resnet18':
            config['architecture'] = args.architecture
        if args.augmentation_type != 'none':
            config['augmentation_type'] = args.augmentation_type
    else:
        config = {
            'dataset': args.dataset,
            'architecture': args.architecture,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'momentum': 0.9,
            'weight_decay': 5e-4,
            'data_dir': args.data_dir,
            'output_dir': args.output_dir,
            'use_wandb': not args.no_wandb,
            'run_name': args.run_name,
            'wandb_project': 'clxai',
            'num_workers': 4,
            'save_freq': 50,
            'seed': args.seed,
            'augmentation_type': args.augmentation_type
        }
    
    train_ce_model(config)


if __name__ == "__main__":
    main()
