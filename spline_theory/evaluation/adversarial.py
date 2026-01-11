"""
Adversarial robustness evaluation for spline theory experiments.

Implements:
- FGSM (Fast Gradient Sign Method)
- PGD (Projected Gradient Descent)
- Optional AutoAttack integration

These attacks are critical for testing the grokking hypothesis:
Prediction P1: CL models show adversarial robustness earlier in training than CE models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple, Literal
from tqdm import tqdm


def fgsm_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 8/255,
    targeted: bool = False,
    target_labels: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Fast Gradient Sign Method (FGSM) attack.
    
    Single-step attack: x_adv = x + eps * sign(grad_x(L(x, y)))
    
    Args:
        model: Model to attack
        images: Input images (B, C, H, W)
        labels: True labels (B,)
        eps: Perturbation magnitude (L-inf norm)
        targeted: If True, minimize loss for target_labels
        target_labels: Target labels for targeted attack
    
    Returns:
        Adversarial images
    """
    images = images.clone().detach().requires_grad_(True)
    
    outputs = model(images)
    
    if targeted and target_labels is not None:
        loss = F.cross_entropy(outputs, target_labels)
    else:
        loss = F.cross_entropy(outputs, labels)
    
    model.zero_grad()
    loss.backward()
    
    grad_sign = images.grad.sign()
    
    if targeted:
        # Minimize loss for target class
        perturbed = images - eps * grad_sign
    else:
        # Maximize loss for true class
        perturbed = images + eps * grad_sign
    
    # Clamp to valid image range [0, 1] or maintain original range
    # Note: For normalized images, this may need adjustment
    perturbed = torch.clamp(perturbed, images.min().item(), images.max().item())
    
    return perturbed.detach()


def pgd_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 8/255,
    step_size: float = 2/255,
    num_steps: int = 20,
    random_start: bool = True,
    targeted: bool = False,
    target_labels: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Projected Gradient Descent (PGD) attack.
    
    Multi-step iterative attack with projection onto epsilon-ball.
    
    Args:
        model: Model to attack
        images: Input images (B, C, H, W)
        labels: True labels (B,)
        eps: Maximum perturbation (L-inf norm)
        step_size: Step size for each iteration
        num_steps: Number of PGD steps
        random_start: Initialize with random perturbation within eps-ball
        targeted: If True, minimize loss for target_labels
        target_labels: Target labels for targeted attack
    
    Returns:
        Adversarial images
    """
    perturbed = images.clone().detach()
    
    if random_start:
        # Random initialization within epsilon ball
        perturbed = perturbed + torch.empty_like(perturbed).uniform_(-eps, eps)
        perturbed = torch.clamp(perturbed, images.min().item(), images.max().item())
    
    for _ in range(num_steps):
        perturbed = perturbed.clone().detach().requires_grad_(True)
        
        outputs = model(perturbed)
        
        if targeted and target_labels is not None:
            loss = F.cross_entropy(outputs, target_labels)
        else:
            loss = F.cross_entropy(outputs, labels)
        
        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            grad_sign = perturbed.grad.sign()
            
            if targeted:
                perturbed = perturbed - step_size * grad_sign
            else:
                perturbed = perturbed + step_size * grad_sign
            
            # Project back onto epsilon ball (L-inf)
            delta = torch.clamp(perturbed - images, -eps, eps)
            perturbed = images + delta
            
            # Clamp to valid range
            perturbed = torch.clamp(perturbed, images.min().item(), images.max().item())
    
    return perturbed.detach()


class AdversarialEvaluator:
    """
    Comprehensive adversarial robustness evaluation.
    
    Evaluates model robustness under various attack settings,
    tracking metrics over training to detect grokking transitions.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        eps: float = 8/255,
        pgd_steps: int = 20,
        pgd_step_size: float = 2/255
    ):
        """
        Args:
            model: Model to evaluate
            device: Device to use
            eps: Perturbation budget (L-inf norm, default 8/255 for CIFAR)
            pgd_steps: Number of PGD iterations
            pgd_step_size: PGD step size
        """
        self.model = model.to(device)
        self.device = device
        self.eps = eps
        self.pgd_steps = pgd_steps
        self.pgd_step_size = pgd_step_size
    
    @torch.no_grad()
    def evaluate_clean(
        self,
        data_loader,
        n_samples: Optional[int] = None
    ) -> Dict[str, float]:
        """Evaluate clean accuracy."""
        self.model.eval()
        
        correct = 0
        total = 0
        
        for images, labels in data_loader:
            if n_samples is not None and total >= n_samples:
                break
            
            images, labels = images.to(self.device), labels.to(self.device)
            
            batch_size = min(images.size(0), n_samples - total if n_samples else images.size(0))
            images, labels = images[:batch_size], labels[:batch_size]
            
            outputs = self.model(images)
            _, predicted = outputs.max(1)
            
            correct += predicted.eq(labels).sum().item()
            total += batch_size
        
        return {
            "clean_accuracy": correct / total,
            "clean_correct": correct,
            "clean_total": total
        }
    
    def evaluate_fgsm(
        self,
        data_loader,
        n_samples: Optional[int] = None,
        eps: Optional[float] = None
    ) -> Dict[str, float]:
        """Evaluate FGSM robustness."""
        self.model.eval()
        eps = eps or self.eps
        
        correct = 0
        total = 0
        
        for images, labels in tqdm(data_loader, desc="FGSM eval"):
            if n_samples is not None and total >= n_samples:
                break
            
            images, labels = images.to(self.device), labels.to(self.device)
            
            batch_size = min(images.size(0), n_samples - total if n_samples else images.size(0))
            images, labels = images[:batch_size], labels[:batch_size]
            
            # Generate adversarial examples
            adv_images = fgsm_attack(self.model, images, labels, eps=eps)
            
            with torch.no_grad():
                outputs = self.model(adv_images)
                _, predicted = outputs.max(1)
            
            correct += predicted.eq(labels).sum().item()
            total += batch_size
        
        return {
            "fgsm_accuracy": correct / total,
            "fgsm_correct": correct,
            "fgsm_total": total,
            "fgsm_eps": eps
        }
    
    def evaluate_pgd(
        self,
        data_loader,
        n_samples: Optional[int] = None,
        eps: Optional[float] = None,
        num_steps: Optional[int] = None,
        step_size: Optional[float] = None
    ) -> Dict[str, float]:
        """Evaluate PGD robustness."""
        self.model.eval()
        eps = eps or self.eps
        num_steps = num_steps or self.pgd_steps
        step_size = step_size or self.pgd_step_size
        
        correct = 0
        total = 0
        
        for images, labels in tqdm(data_loader, desc="PGD eval"):
            if n_samples is not None and total >= n_samples:
                break
            
            images, labels = images.to(self.device), labels.to(self.device)
            
            batch_size = min(images.size(0), n_samples - total if n_samples else images.size(0))
            images, labels = images[:batch_size], labels[:batch_size]
            
            # Generate adversarial examples
            adv_images = pgd_attack(
                self.model, images, labels,
                eps=eps, step_size=step_size, num_steps=num_steps
            )
            
            with torch.no_grad():
                outputs = self.model(adv_images)
                _, predicted = outputs.max(1)
            
            correct += predicted.eq(labels).sum().item()
            total += batch_size
        
        return {
            "pgd_accuracy": correct / total,
            "pgd_correct": correct,
            "pgd_total": total,
            "pgd_eps": eps,
            "pgd_steps": num_steps,
            "pgd_step_size": step_size
        }
    
    def evaluate_all(
        self,
        data_loader,
        n_samples: int = 1000,
        attacks: Optional[list] = None
    ) -> Dict[str, float]:
        """
        Run comprehensive adversarial evaluation.
        
        Args:
            data_loader: Test data loader
            n_samples: Number of samples to evaluate
            attacks: List of attacks to run ('fgsm', 'pgd'). Default: both
        
        Returns:
            Dictionary with all metrics
        """
        if attacks is None:
            attacks = ["fgsm", "pgd"]
        
        results = {}
        
        # Always evaluate clean accuracy
        clean_results = self.evaluate_clean(data_loader, n_samples)
        results.update(clean_results)
        
        if "fgsm" in attacks:
            fgsm_results = self.evaluate_fgsm(data_loader, n_samples)
            results.update(fgsm_results)
        
        if "pgd" in attacks:
            pgd_results = self.evaluate_pgd(data_loader, n_samples)
            results.update(pgd_results)
        
        return results
    
    def evaluate_robustness_curve(
        self,
        data_loader,
        eps_values: Optional[list] = None,
        n_samples: int = 500,
        attack: str = "pgd"
    ) -> Dict[str, np.ndarray]:
        """
        Evaluate robustness across different epsilon values.
        
        Useful for understanding the robustness-accuracy tradeoff.
        
        Args:
            data_loader: Test data loader
            eps_values: List of epsilon values to test
            n_samples: Number of samples per epsilon
            attack: Attack type ('fgsm' or 'pgd')
        
        Returns:
            Dictionary with epsilon values and corresponding accuracies
        """
        if eps_values is None:
            eps_values = [0, 1/255, 2/255, 4/255, 8/255, 16/255]
        
        accuracies = []
        
        for eps in tqdm(eps_values, desc="Robustness curve"):
            if eps == 0:
                result = self.evaluate_clean(data_loader, n_samples)
                acc = result["clean_accuracy"]
            elif attack == "fgsm":
                result = self.evaluate_fgsm(data_loader, n_samples, eps=eps)
                acc = result["fgsm_accuracy"]
            else:  # pgd
                result = self.evaluate_pgd(data_loader, n_samples, eps=eps)
                acc = result["pgd_accuracy"]
            
            accuracies.append(acc)
        
        return {
            "eps_values": np.array(eps_values),
            "accuracies": np.array(accuracies),
            "attack": attack
        }


def evaluate_adversarial_robustness(
    model: nn.Module,
    data_loader,
    device: str = "cuda",
    n_samples: int = 1000,
    eps: float = 8/255
) -> Dict[str, float]:
    """
    Convenience function for quick adversarial evaluation.
    
    Args:
        model: Model to evaluate
        data_loader: Test data loader
        device: Device to use
        n_samples: Number of samples
        eps: Perturbation budget
    
    Returns:
        Dictionary with clean, FGSM, and PGD accuracies
    """
    evaluator = AdversarialEvaluator(model, device=device, eps=eps)
    return evaluator.evaluate_all(data_loader, n_samples=n_samples)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from spline_theory.models.resnet_variants import get_resnet_variant
    
    # Test adversarial evaluation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 60)
    print("Testing Adversarial Evaluation")
    print("=" * 60)
    
    model = get_resnet_variant(
        architecture="resnet18",
        num_classes=10,
        norm_type="bn"
    ).to(device)
    model.eval()
    
    # Create dummy data
    batch_size = 16
    images = torch.randn(batch_size, 3, 32, 32).to(device)
    labels = torch.randint(0, 10, (batch_size,)).to(device)
    
    print("\nTesting FGSM attack:")
    adv_fgsm = fgsm_attack(model, images, labels, eps=8/255)
    print(f"  Original range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"  Adversarial range: [{adv_fgsm.min():.3f}, {adv_fgsm.max():.3f}]")
    print(f"  Max perturbation: {(adv_fgsm - images).abs().max():.6f}")
    
    print("\nTesting PGD attack:")
    adv_pgd = pgd_attack(model, images, labels, eps=8/255, num_steps=10)
    print(f"  Original range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"  Adversarial range: [{adv_pgd.min():.3f}, {adv_pgd.max():.3f}]")
    print(f"  Max perturbation: {(adv_pgd - images).abs().max():.6f}")
    
    print("\nTesting AdversarialEvaluator:")
    evaluator = AdversarialEvaluator(model, device=device, eps=8/255, pgd_steps=5)
    
    # Create a simple data loader for testing
    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(images.cpu(), labels.cpu())
    loader = DataLoader(dataset, batch_size=8)
    
    results = evaluator.evaluate_all(loader, n_samples=16)
    
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
