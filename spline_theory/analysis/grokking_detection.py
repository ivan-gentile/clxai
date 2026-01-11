"""
Grokking detection and phase transition analysis.

Implements:
- Automatic detection of grokking transitions
- Gradient norm analysis over training
- Phase transition identification

Grokking phenomenon: With minimal regularization and extended training,
models spontaneously reorganize their internal representations, causing
delayed emergence of generalization and adversarial robustness.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import torch
import torch.nn as nn


def compute_gradient_norms(
    model: nn.Module,
    data_loader,
    criterion: nn.Module,
    device: str = "cuda",
    n_batches: int = 10
) -> Dict[str, float]:
    """
    Compute gradient norm statistics on training data.
    
    Large gradient norms indicate the model is still learning actively.
    A sudden drop in gradient norms may indicate a phase transition.
    
    Args:
        model: Model to analyze
        data_loader: Training data loader
        criterion: Loss function
        device: Device to use
        n_batches: Number of batches to average over
    
    Returns:
        Gradient norm statistics
    """
    model.train()
    
    grad_norms = []
    losses = []
    
    for i, (images, labels) in enumerate(data_loader):
        if i >= n_batches:
            break
        
        images, labels = images.to(device), labels.to(device)
        
        model.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Compute gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        
        grad_norms.append(total_norm)
        losses.append(loss.item())
    
    return {
        "mean_grad_norm": np.mean(grad_norms),
        "std_grad_norm": np.std(grad_norms),
        "max_grad_norm": np.max(grad_norms),
        "min_grad_norm": np.min(grad_norms),
        "mean_loss": np.mean(losses)
    }


class GrokkingDetector:
    """
    Detect grokking phase transitions in training history.
    
    Grokking is characterized by:
    1. Training loss reaching near-zero while test accuracy remains low
    2. Delayed sudden improvement in test accuracy
    3. Often accompanied by changes in gradient norm dynamics
    """
    
    def __init__(
        self,
        smooth_window: int = 11,
        threshold_percentile: float = 95
    ):
        """
        Args:
            smooth_window: Window size for smoothing (must be odd)
            threshold_percentile: Percentile for detecting significant changes
        """
        self.smooth_window = smooth_window
        self.threshold_percentile = threshold_percentile
    
    def smooth_curve(self, curve: np.ndarray) -> np.ndarray:
        """Apply Savitzky-Golay smoothing to reduce noise."""
        if len(curve) < self.smooth_window:
            return gaussian_filter1d(curve, sigma=2)
        return savgol_filter(curve, self.smooth_window, 3)
    
    def detect_phase_transition(
        self,
        test_acc: np.ndarray,
        train_acc: np.ndarray,
        epochs: np.ndarray
    ) -> Dict:
        """
        Detect grokking phase transition.
        
        Looks for epochs where:
        - Train accuracy is high (>95%)
        - Test accuracy shows sudden improvement
        
        Args:
            test_acc: Test accuracy curve
            train_acc: Training accuracy curve
            epochs: Epoch numbers
        
        Returns:
            Detection results including transition epoch if found
        """
        # Smooth curves
        test_smooth = self.smooth_curve(test_acc)
        train_smooth = self.smooth_curve(train_acc)
        
        # Compute derivatives
        test_deriv = np.gradient(test_smooth)
        
        # Find epochs where train acc is high
        train_converged_mask = train_smooth > 95.0
        
        # Find significant positive changes in test acc
        threshold = np.percentile(np.abs(test_deriv), self.threshold_percentile)
        significant_improvement = test_deriv > threshold
        
        # Grokking: train converged but test still improving significantly
        grokking_candidates = train_converged_mask & significant_improvement
        
        if np.any(grokking_candidates):
            # Find first grokking epoch
            grokking_indices = np.where(grokking_candidates)[0]
            transition_idx = grokking_indices[0]
            transition_epoch = epochs[transition_idx]
            
            # Compute gap metrics
            gap_at_transition = train_smooth[transition_idx] - test_smooth[transition_idx]
            
            return {
                "grokking_detected": True,
                "transition_epoch": int(transition_epoch),
                "transition_idx": int(transition_idx),
                "gap_at_transition": float(gap_at_transition),
                "test_acc_before": float(test_smooth[max(0, transition_idx - 1)]),
                "test_acc_after": float(test_smooth[min(len(test_smooth) - 1, transition_idx + 10)]),
                "improvement_rate": float(test_deriv[transition_idx])
            }
        
        return {
            "grokking_detected": False,
            "transition_epoch": None,
            "max_gap": float(np.max(train_smooth - test_smooth)),
            "final_gap": float(train_smooth[-1] - test_smooth[-1])
        }
    
    def analyze_training_dynamics(
        self,
        history: Dict[str, List]
    ) -> Dict:
        """
        Comprehensive analysis of training dynamics.
        
        Args:
            history: Training history with keys:
                     'epoch', 'train_acc', 'test_acc', 'train_loss', 
                     'test_loss', 'gradient_norm'
        
        Returns:
            Analysis results
        """
        epochs = np.array(history.get("epoch", []))
        train_acc = np.array(history.get("train_acc", []))
        test_acc = np.array(history.get("test_acc", []))
        train_loss = np.array(history.get("train_loss", []))
        test_loss = np.array(history.get("test_loss", []))
        grad_norms = np.array(history.get("gradient_norm", []))
        
        results = {
            "total_epochs": len(epochs),
            "final_train_acc": float(train_acc[-1]) if len(train_acc) > 0 else None,
            "final_test_acc": float(test_acc[-1]) if len(test_acc) > 0 else None,
            "best_test_acc": float(np.max(test_acc)) if len(test_acc) > 0 else None,
        }
        
        # Detect grokking
        if len(epochs) > 10 and len(train_acc) > 10 and len(test_acc) > 10:
            grokking_results = self.detect_phase_transition(test_acc, train_acc, epochs)
            results.update(grokking_results)
        else:
            results["grokking_detected"] = False
        
        # Analyze gradient norm dynamics
        if len(grad_norms) > 10:
            grad_smooth = self.smooth_curve(grad_norms)
            grad_deriv = np.gradient(grad_smooth)
            
            # Find gradient norm phase changes
            grad_threshold = np.percentile(np.abs(grad_deriv), 90)
            significant_changes = np.where(np.abs(grad_deriv) > grad_threshold)[0]
            
            results["grad_norm_initial"] = float(grad_norms[0])
            results["grad_norm_final"] = float(grad_norms[-1])
            results["grad_norm_ratio"] = float(grad_norms[-1] / (grad_norms[0] + 1e-10))
            results["n_grad_phase_changes"] = len(significant_changes)
        
        # Compute generalization gap over time
        if len(train_acc) > 0 and len(test_acc) > 0:
            gap = train_acc - test_acc
            results["initial_gap"] = float(gap[0])
            results["final_gap"] = float(gap[-1])
            results["max_gap"] = float(np.max(gap))
            results["gap_closed_epochs"] = int(np.argmin(gap)) if np.min(gap) < results["max_gap"] * 0.5 else -1
        
        return results


def detect_grokking_transition(
    history: Dict[str, List],
    smooth_window: int = 11
) -> Dict:
    """
    Convenience function for grokking detection.
    
    Args:
        history: Training history dictionary
        smooth_window: Smoothing window size
    
    Returns:
        Grokking detection results
    """
    detector = GrokkingDetector(smooth_window=smooth_window)
    return detector.analyze_training_dynamics(history)


if __name__ == "__main__":
    # Test grokking detection
    print("=" * 60)
    print("Testing Grokking Detection")
    print("=" * 60)
    
    # Create synthetic training history simulating grokking
    epochs = np.arange(1, 501)
    
    # Train acc converges quickly
    train_acc = 99 * (1 - np.exp(-epochs / 20)) + np.random.randn(500) * 0.5
    train_acc = np.clip(train_acc, 0, 100)
    
    # Test acc shows delayed improvement (grokking)
    test_acc = np.zeros(500)
    test_acc[:100] = 60 + np.random.randn(100) * 2  # Plateau
    test_acc[100:200] = np.linspace(60, 85, 100) + np.random.randn(100) * 2  # Gradual
    test_acc[200:] = 90 * (1 - np.exp(-(epochs[200:] - 200) / 30)) + 10 + np.random.randn(300) * 1  # Fast improvement
    test_acc = np.clip(test_acc, 0, 100)
    
    # Gradient norms decrease over training
    grad_norms = 10 * np.exp(-epochs / 100) + np.random.randn(500) * 0.5
    grad_norms = np.clip(grad_norms, 0.1, 20)
    
    history = {
        "epoch": epochs.tolist(),
        "train_acc": train_acc.tolist(),
        "test_acc": test_acc.tolist(),
        "gradient_norm": grad_norms.tolist()
    }
    
    print("\nAnalyzing synthetic training history...")
    detector = GrokkingDetector()
    results = detector.analyze_training_dynamics(history)
    
    print(f"  Grokking detected: {results['grokking_detected']}")
    if results['grokking_detected']:
        print(f"  Transition epoch: {results['transition_epoch']}")
        print(f"  Gap at transition: {results['gap_at_transition']:.2f}%")
    print(f"  Final train acc: {results['final_train_acc']:.2f}%")
    print(f"  Final test acc: {results['final_test_acc']:.2f}%")
    print(f"  Max generalization gap: {results['max_gap']:.2f}%")
    
    print("\n" + "=" * 60)
    print("Grokking detection test passed!")
    print("=" * 60)
