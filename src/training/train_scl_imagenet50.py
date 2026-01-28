"""
Supervised Contrastive Learning for ResNet-50 on ImageNet-50 (50-class subset).

Based on:
- SupCon paper (Khosla et al., NeurIPS 2020)
- Explainable-KD-CNN reference (Jasper Wi)

Key features:
- 50-class ImageNet subset for faster training
- RAdam/AdamW optimizer support
- F-Fidelity pixel50 augmentation
- Two-stage training: SupCon pretraining + Linear probe

Usage:
    python src/training/train_scl_imagenet50.py --config configs/supcon_imagenet50_r50.yaml
"""

import os
import sys
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

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.imagenet_resnet import get_imagenet_resnet
from src.models.classifiers import KNNClassifier, LinearClassifier, train_linear_classifier
from src.training.losses import SupConLossV2, TripletLoss
from src.utils.imagenet_subset_data import get_imagenet50_loaders, get_imagenet50_datasets, SELECTED_CLASSES

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_optimizer(model, config: dict):
    """
    Create optimizer based on config.
    
    Supports: SGD, Adam, AdamW, RAdam
    """
    optimizer_name = config.get('optimizer', 'sgd').lower()
    lr = config.get('lr', 0.001)
    weight_decay = config.get('weight_decay', 1e-4)
    momentum = config.get('momentum', 0.9)
    
    if optimizer_name == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'adam':
        return optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'adamw':
        return optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'radam':
        return optim.RAdam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}. Use sgd, adam, adamw, or radam")


def train_epoch(model, loader, criterion, optimizer, device, epoch, loss_type='supcon', grad_clip=None):
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    
    # Handle DataParallel wrapper
    model_ref = model.module if hasattr(model, 'module') else model
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    
    for images, labels in pbar:
        # For contrastive learning, images is a list of two views
        if isinstance(images, list):
            images = torch.cat([images[0], images[1]], dim=0).to(device)
        else:
            images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        if loss_type == 'triplet':
            embeddings = model_ref.get_embedding(images, normalize=False)
        else:
            embeddings = model(images)
        
        loss = criterion(embeddings, labels)
        loss.backward()
        
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        total_loss += loss.item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return {'train_loss': total_loss / len(loader)}


@torch.no_grad()
def extract_embeddings(model, loader, device, max_samples=None):
    """Extract embeddings from model."""
    model.eval()
    embeddings, labels, total = [], [], 0
    model_ref = model.module if hasattr(model, 'module') else model
    
    for images, targets in tqdm(loader, desc="Extracting embeddings"):
        if isinstance(images, list):
            images = images[0]
        features = model_ref.get_embedding(images.to(device), normalize=True)
        embeddings.append(features.cpu())
        labels.append(targets)
        total += images.size(0)
        if max_samples and total >= max_samples:
            break
    
    return torch.cat(embeddings, dim=0).numpy(), torch.cat(labels, dim=0).numpy()


def evaluate_knn(model, train_loader, test_loader, device, k=200, max_train=50000):
    """Evaluate model using k-NN classifier."""
    train_emb, train_labels = extract_embeddings(model, train_loader, device, max_train)
    test_emb, test_labels = extract_embeddings(model, test_loader, device)
    
    knn = KNNClassifier(k=k, metric='cosine')
    knn.fit(train_emb, train_labels)
    
    return knn.score(test_emb, test_labels) * 100


def train_scl_imagenet50(config: dict):
    """Main training function for ImageNet-50 SupCon."""
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gpus = torch.cuda.device_count()
    print(f"\n{'='*60}")
    print(f"ImageNet-50 SupCon Training")
    print(f"{'='*60}")
    print(f"Device: {device}, GPUs: {n_gpus}")
    
    set_seed(config.get('seed', 42))
    
    # Output directory
    output_dir = Path(config.get('output_dir', 'results/models/imagenet50_r50/supcon'))
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")
    
    # Initialize wandb
    if WANDB_AVAILABLE and config.get('use_wandb', True):
        wandb.init(
            project=config.get('wandb_project', 'clxai'),
            name=config.get('run_name', 'supcon_imagenet50_r50'),
            config=config
        )
    
    # Data loaders
    loss_type = config.get('loss', 'supcon')
    augmentation_type = config.get('augmentation_type', 'pixel50')
    
    print(f"\nLoading ImageNet-50 data...")
    print(f"  Loss type: {loss_type}")
    print(f"  Augmentation: {augmentation_type}")
    print(f"  Contrastive: {loss_type == 'supcon'}")
    
    train_loader, val_loader = get_imagenet50_loaders(
        data_dir=config.get('data_dir', '/leonardo_scratch/fast/CNHPC_1905882/clxai/data/imagenet-1k'),
        batch_size=config.get('batch_size', 128),
        num_workers=config.get('num_workers', 8),
        contrastive=(loss_type == 'supcon'),
        augmentation_type=augmentation_type
    )
    
    # Also create non-contrastive loaders for evaluation
    train_loader_eval, val_loader_eval = get_imagenet50_loaders(
        data_dir=config.get('data_dir', '/leonardo_scratch/fast/CNHPC_1905882/clxai/data/imagenet-1k'),
        batch_size=config.get('batch_size', 128),
        num_workers=config.get('num_workers', 8),
        contrastive=False,
        augment=False
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Model
    architecture = config.get('architecture', 'resnet50_imagenet')
    embedding_dim = config.get('embedding_dim', 128)
    pretrained = config.get('pretrained', False)
    
    model = get_imagenet_resnet(
        architecture=architecture,
        num_classes=50,  # Will be ignored for encoder
        encoder_only=True,
        embedding_dim=embedding_dim,
        pretrained=pretrained
    ).to(device)
    
    feature_dim = model.feature_dim if hasattr(model, 'feature_dim') else 2048
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\nModel: {architecture}")
    print(f"  Parameters: {total_params/1e6:.1f}M")
    print(f"  Feature dim: {feature_dim}")
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  Pretrained: {pretrained}")
    
    # Multi-GPU with DataParallel
    if n_gpus > 1:
        model = DataParallel(model)
        print(f"  Using DataParallel on {n_gpus} GPUs")
    
    # Loss function
    temperature = config.get('temperature', 0.07)
    if loss_type == 'supcon':
        criterion = SupConLossV2(temperature=temperature)
        print(f"\nLoss: SupCon (temperature={temperature})")
    else:
        criterion = TripletLoss(margin=0.3, mining='semi-hard')
        print(f"\nLoss: Triplet (margin=0.3)")
    
    # Optimizer
    optimizer = get_optimizer(model, config)
    optimizer_name = config.get('optimizer', 'radam')
    lr = config.get('lr', 0.001)
    print(f"\nOptimizer: {optimizer_name.upper()}")
    print(f"  LR: {lr}")
    print(f"  Weight decay: {config.get('weight_decay', 1e-4)}")
    
    # Scheduler
    epochs = config.get('epochs', 160)
    warmup = config.get('warmup_epochs', 5)
    
    def lr_lambda(epoch):
        if epoch < warmup:
            return 0.1 + 0.9 * (epoch / warmup)
        else:
            return 0.5 * (1 + np.cos(np.pi * (epoch - warmup) / (epochs - warmup)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    print(f"\nScheduler: Warmup ({warmup} epochs) + Cosine Annealing")
    print(f"Total epochs: {epochs}")
    
    # Resume from checkpoint
    start_epoch = 1
    best_acc = 0.0
    resume_path = config.get('resume')
    
    if resume_path and Path(resume_path).exists():
        print(f"\nResuming from: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model_ref = model.module if hasattr(model, 'module') else model
        model_ref.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_acc = checkpoint.get('knn_acc', 0.0)
        print(f"  Resumed from epoch {start_epoch-1}, best_acc={best_acc:.2f}%")
    
    # Training loop
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    eval_freq = config.get('eval_freq', 10)
    save_freq = config.get('save_freq', 10)
    
    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        
        metrics = train_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, loss_type, config.get('grad_clip')
        )
        
        scheduler.step()
        
        # Evaluation
        if epoch % eval_freq == 0 or epoch == epochs:
            knn_acc = evaluate_knn(
                model, train_loader_eval, val_loader_eval, device,
                k=config.get('knn_k', 200)
            )
            metrics['knn_acc'] = knn_acc
            
            print(f"Epoch {epoch}: loss={metrics['train_loss']:.4f}, kNN={knn_acc:.2f}%, time={time.time()-t0:.0f}s")
            
            if knn_acc > best_acc:
                best_acc = knn_acc
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_to_save.state_dict(),
                    'knn_acc': best_acc
                }, output_dir / 'best_model.pt')
                print(f"  New best! Saved best_model.pt")
        else:
            print(f"Epoch {epoch}: loss={metrics['train_loss']:.4f}, time={time.time()-t0:.0f}s")
        
        # Logging
        if WANDB_AVAILABLE and config.get('use_wandb', True):
            metrics['lr'] = scheduler.get_last_lr()[0]
            wandb.log(metrics, step=epoch)
        
        # Save checkpoint
        if epoch % save_freq == 0:
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'knn_acc': best_acc
            }, output_dir / f'checkpoint_epoch_{epoch}.pt')
            print(f"  Saved checkpoint_epoch_{epoch}.pt")
            
            # Keep only last 3 checkpoints
            old_ckpts = sorted(output_dir.glob('checkpoint_epoch_*.pt'),
                             key=lambda p: int(p.stem.split('_')[-1]))[:-3]
            for old_ckpt in old_ckpts:
                old_ckpt.unlink()
    
    # Final model
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save({
        'epoch': epochs,
        'model_state_dict': model_to_save.state_dict()
    }, output_dir / 'final_model.pt')
    
    # Stage 2: Linear Probe
    print(f"\n{'='*60}")
    print("Stage 2: Training Linear Probe")
    print(f"{'='*60}\n")
    
    linear_epochs = config.get('linear_probe', {}).get('epochs', 100)
    linear_lr = config.get('linear_probe', {}).get('lr', 0.1)
    num_classes = config.get('num_classes', 50)
    
    print(f"Extracting embeddings...")
    train_emb, train_labels = extract_embeddings(model, train_loader_eval, device)
    test_emb, test_labels = extract_embeddings(model, val_loader_eval, device)
    
    print(f"  Train embeddings: {train_emb.shape}")
    print(f"  Test embeddings: {test_emb.shape}")
    
    linear_clf = LinearClassifier(input_dim=feature_dim, num_classes=num_classes)
    
    print(f"\nTraining linear classifier ({linear_epochs} epochs, lr={linear_lr})...")
    history = train_linear_classifier(
        linear_clf,
        torch.tensor(train_emb),
        torch.tensor(train_labels),
        torch.tensor(test_emb),
        torch.tensor(test_labels),
        epochs=linear_epochs,
        lr=linear_lr,
        device=str(device)
    )
    
    linear_acc = history['val_acc'][-1] * 100
    print(f"\nLinear probe accuracy: {linear_acc:.2f}%")
    
    torch.save(linear_clf.state_dict(), output_dir / 'linear_classifier.pt')
    
    # Summary
    total_time = (time.time() - start_time) / 3600
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"  Total time: {total_time:.2f} hours")
    print(f"  Best kNN accuracy: {best_acc:.2f}%")
    print(f"  Linear probe accuracy: {linear_acc:.2f}%")
    print(f"  Target: >80% (goal: 84%)")
    print(f"  Output: {output_dir}")
    
    if WANDB_AVAILABLE and config.get('use_wandb', True):
        wandb.log({
            'final_knn_acc': best_acc,
            'linear_probe_acc': linear_acc
        })
        wandb.finish()
    
    return best_acc, linear_acc


def main():
    parser = argparse.ArgumentParser(description='ImageNet-50 SupCon Training')
    
    parser.add_argument('--config', type=str, help='Path to config YAML file')
    parser.add_argument('--architecture', default='resnet50_imagenet')
    parser.add_argument('--epochs', type=int, default=160)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--optimizer', default='radam', choices=['sgd', 'adam', 'adamw', 'radam'])
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--loss', default='supcon', choices=['supcon', 'triplet'])
    parser.add_argument('--data_dir', default='/leonardo_scratch/fast/CNHPC_1905882/clxai/data/imagenet-1k')
    parser.add_argument('--output_dir', default='results/models/imagenet50_r50/supcon')
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--run_name', default='supcon_imagenet50_r50')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--augmentation_type', default='pixel50',
                       choices=['none', 'pixel50', 'pixel', 'noise', 'patch'])
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--eval_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--knn_k', type=int, default=200)
    parser.add_argument('--grad_clip', type=float, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--pretrained', action='store_true')
    
    args = parser.parse_args()
    
    # Load config from YAML or use command line args
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            config = yaml.safe_load(f)
        
        # Flatten nested config sections
        for section in ['model', 'training', 'data', 'evaluation', 'logging', 'output']:
            if section in config:
                for k, v in config[section].items():
                    if k not in config:
                        config[k] = v
        
        # Handle nested supcon config
        if 'supcon' in config:
            for k, v in config['supcon'].items():
                if k not in config:
                    config[k] = v
        
        # Handle nested linear_probe config
        if 'linear_probe' in config.get('evaluation', {}):
            config['linear_probe'] = config['evaluation']['linear_probe']
    else:
        config = vars(args).copy()
        config['use_wandb'] = not args.no_wandb
        config['wandb_project'] = 'clxai'
        config['warmup_epochs'] = 5
        config['embedding_dim'] = 128
        config['num_classes'] = 50
        config['linear_probe'] = {'epochs': 100, 'lr': 0.1, 'num_classes': 50}
    
    # Override with resume if provided
    if args.resume:
        config['resume'] = args.resume
    
    train_scl_imagenet50(config)


if __name__ == "__main__":
    main()
