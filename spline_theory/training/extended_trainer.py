"""
Extended training framework for spline theory experiments.

Supports:
- Very long training (up to 10,000 epochs)
- Dense checkpointing at specified milestone epochs
- Gradient norm tracking for grokking detection
- Resume from checkpoint
- Comprehensive logging
"""

import os
import sys
import time
import random
from pathlib import Path
from typing import Optional, Dict, List, Callable, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# Default checkpoint epochs for dense monitoring of training dynamics
CHECKPOINT_EPOCHS = [100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000, 10000]


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_gradient_norm(model: nn.Module) -> float:
    """Compute total gradient norm across all parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


class ExtendedTrainer:
    """
    Extended training loop with comprehensive monitoring and checkpointing.
    
    Designed for:
    - Testing grokking hypothesis with very long training
    - Dense checkpointing at milestone epochs
    - Tracking gradient norms for phase transition detection
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: str = "cuda",
        checkpoint_dir: str = "checkpoints",
        checkpoint_epochs: Optional[List[int]] = None,
        max_epochs: int = 10000,
        eval_frequency: int = 10,
        use_wandb: bool = True,
        project_name: str = "spline_theory",
        run_name: str = "extended_training"
    ):
        """
        Args:
            model: Model to train
            train_loader: Training data loader
            test_loader: Test data loader
            optimizer: Optimizer
            criterion: Loss function
            device: Device to use
            checkpoint_dir: Directory to save checkpoints
            checkpoint_epochs: List of epochs to save checkpoints (default: CHECKPOINT_EPOCHS)
            max_epochs: Maximum training epochs
            eval_frequency: Evaluate every N epochs
            use_wandb: Whether to log to wandb
            project_name: Wandb project name
            run_name: Wandb run name
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_epochs = checkpoint_epochs or CHECKPOINT_EPOCHS
        self.max_epochs = max_epochs
        self.eval_frequency = eval_frequency
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.project_name = project_name
        self.run_name = run_name
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.best_acc = 0.0
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": [],
            "lr": [],
            "gradient_norm": [],
            "epoch": []
        }
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        gradient_norms = []
        
        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            # Track gradient norm before optimizer step
            grad_norm = compute_gradient_norm(self.model)
            gradient_norms.append(grad_norm)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        return {
            "train_loss": total_loss / len(self.train_loader),
            "train_acc": 100.0 * correct / total,
            "gradient_norm": np.mean(gradient_norms)
        }
    
    @torch.no_grad()
    def _evaluate(self) -> Dict[str, float]:
        """Evaluate on test set."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in self.test_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        return {
            "test_loss": total_loss / len(self.test_loader),
            "test_acc": 100.0 * correct / total
        }
    
    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
        is_milestone: bool = False
    ):
        """Save training checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "history": self.history,
            "best_acc": self.best_acc
        }
        
        # Always save latest
        torch.save(checkpoint, self.checkpoint_dir / "latest.pt")
        
        # Save best model
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best_model.pt")
        
        # Save milestone checkpoints
        if is_milestone:
            torch.save(checkpoint, self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load checkpoint and return starting epoch.
        
        Args:
            checkpoint_path: Path to checkpoint file
        
        Returns:
            Epoch to resume from
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint.get("history", self.history)
        self.best_acc = checkpoint.get("best_acc", 0.0)
        
        return checkpoint["epoch"]
    
    def train(
        self,
        scheduler: Optional[object] = None,
        resume_from: Optional[str] = None,
        evaluation_callback: Optional[Callable] = None
    ) -> Dict[str, List]:
        """
        Run extended training loop.
        
        Args:
            scheduler: Learning rate scheduler
            resume_from: Path to checkpoint to resume from
            evaluation_callback: Optional callback for additional evaluation
                                 (e.g., adversarial robustness)
        
        Returns:
            Training history
        """
        start_epoch = 1
        
        # Resume from checkpoint if specified
        if resume_from and Path(resume_from).exists():
            start_epoch = self.load_checkpoint(resume_from) + 1
            print(f"Resumed from epoch {start_epoch - 1}")
        
        # Initialize wandb
        if self.use_wandb:
            wandb.init(
                project=self.project_name,
                name=self.run_name,
                resume="allow" if resume_from else None
            )
        
        start_time = time.time()
        
        for epoch in range(start_epoch, self.max_epochs + 1):
            self.current_epoch = epoch
            
            # Train for one epoch
            train_metrics = self._train_epoch()
            
            # Update scheduler
            if scheduler is not None:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
            else:
                current_lr = self.optimizer.param_groups[0]["lr"]
            
            train_metrics["lr"] = current_lr
            
            # Evaluate periodically
            is_eval_epoch = (epoch % self.eval_frequency == 0) or (epoch == self.max_epochs)
            is_milestone = epoch in self.checkpoint_epochs
            
            if is_eval_epoch or is_milestone:
                test_metrics = self._evaluate()
                
                # Run additional evaluation callback
                if evaluation_callback is not None:
                    extra_metrics = evaluation_callback(self.model, epoch)
                    test_metrics.update(extra_metrics)
                
                # Check for best model
                is_best = test_metrics["test_acc"] > self.best_acc
                if is_best:
                    self.best_acc = test_metrics["test_acc"]
                
                # Save checkpoint
                all_metrics = {**train_metrics, **test_metrics}
                self.save_checkpoint(
                    epoch, all_metrics,
                    is_best=is_best,
                    is_milestone=is_milestone
                )
                
                # Update history
                self.history["epoch"].append(epoch)
                self.history["train_loss"].append(train_metrics["train_loss"])
                self.history["train_acc"].append(train_metrics["train_acc"])
                self.history["test_loss"].append(test_metrics["test_loss"])
                self.history["test_acc"].append(test_metrics["test_acc"])
                self.history["lr"].append(current_lr)
                self.history["gradient_norm"].append(train_metrics["gradient_norm"])
                
                # Log to console
                elapsed = time.time() - start_time
                print(f"Epoch {epoch}/{self.max_epochs} "
                      f"[{elapsed/3600:.1f}h] "
                      f"train_acc={train_metrics['train_acc']:.2f}% "
                      f"test_acc={test_metrics['test_acc']:.2f}% "
                      f"grad_norm={train_metrics['gradient_norm']:.4f}")
                
                # Log to wandb
                if self.use_wandb:
                    wandb.log(all_metrics, step=epoch)
            else:
                # Just log training metrics
                if self.use_wandb:
                    wandb.log(train_metrics, step=epoch)
        
        # Final cleanup
        if self.use_wandb:
            wandb.finish()
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/3600:.2f} hours")
        print(f"Best test accuracy: {self.best_acc:.2f}%")
        
        return self.history


def train_extended(
    config: Dict,
    model: Optional[nn.Module] = None,
    resume_checkpoint: Optional[str] = None
) -> Dict:
    """
    Convenience function for extended training from config.
    
    Args:
        config: Training configuration dictionary
        model: Optional pre-initialized model
        resume_checkpoint: Path to checkpoint to resume from
    
    Returns:
        Training history
    """
    from spline_theory.models.resnet_variants import get_resnet_variant
    from src.utils.data import get_data_loaders, get_num_classes
    
    # Set seed
    seed = config.get("seed", 42)
    set_seed(seed)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get dataset info
    dataset = config.get("dataset", "cifar10")
    num_classes = get_num_classes(dataset)
    
    # Create model if not provided
    if model is None:
        model = get_resnet_variant(
            architecture=config.get("architecture", "resnet18"),
            num_classes=num_classes,
            norm_type=config.get("norm_type", "bn"),
            encoder_only=False
        )
    
    # Data loaders
    train_loader, test_loader = get_data_loaders(
        dataset=dataset,
        data_dir=config.get("data_dir", "./data"),
        batch_size=config.get("batch_size", 128),
        num_workers=config.get("num_workers", 4),
        augment=True,
        contrastive=False
    )
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.get("lr", 0.1),
        momentum=config.get("momentum", 0.9),
        weight_decay=config.get("weight_decay", 5e-4)
    )
    
    # Scheduler
    max_epochs = config.get("max_epochs", 10000)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)
    
    # Create trainer
    trainer = ExtendedTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=str(device),
        checkpoint_dir=config.get("checkpoint_dir", "checkpoints"),
        checkpoint_epochs=config.get("checkpoint_epochs", CHECKPOINT_EPOCHS),
        max_epochs=max_epochs,
        eval_frequency=config.get("eval_frequency", 10),
        use_wandb=config.get("use_wandb", True),
        project_name=config.get("project_name", "spline_theory"),
        run_name=config.get("run_name", "extended_training")
    )
    
    # Train
    history = trainer.train(
        scheduler=scheduler,
        resume_from=resume_checkpoint
    )
    
    return history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extended training for spline theory")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--architecture", type=str, default="resnet18")
    parser.add_argument("--norm_type", type=str, default="bn",
                        choices=["bn", "gn", "ln", "id"])
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_wandb", action="store_true")
    
    args = parser.parse_args()
    
    # Build config
    if args.config and Path(args.config).exists():
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {
            "dataset": args.dataset,
            "architecture": args.architecture,
            "norm_type": args.norm_type,
            "max_epochs": args.max_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "checkpoint_dir": args.checkpoint_dir,
            "seed": args.seed,
            "use_wandb": not args.no_wandb,
            "run_name": f"{args.architecture}_{args.norm_type}_extended"
        }
    
    train_extended(config, resume_checkpoint=args.resume)
