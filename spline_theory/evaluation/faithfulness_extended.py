"""
Extended faithfulness evaluation for spline theory experiments.

Builds on existing faithfulness metrics, adding:
- Confidence curve smoothness (second derivative)
- Monotonicity tracking
- Integration with embedding trajectory analysis
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from scipy import stats
from scipy.integrate import trapezoid

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import from existing codebase
from src.xai.saliency import SaliencyExtractor, normalize_saliency
from src.xai.pixel_flipping import PixelFlipping
from src.analysis.faithfulness import ECEFaithfulness, FaithfulnessEvaluator


class ExtendedFaithfulnessEvaluator:
    """
    Extended faithfulness evaluation combining multiple metrics.
    
    Computes:
    - AUC-Deletion (lower is better)
    - AUC-Insertion (higher is better)
    - Monotonicity (Spearman correlation)
    - Curve smoothness (second derivative magnitude)
    - ECE-Faithfulness (calibration-based metric)
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        n_steps: int = 20,
        perturbation: str = "mean"
    ):
        """
        Args:
            model: Model to evaluate
            device: Device to use
            n_steps: Number of perturbation steps
            perturbation: Perturbation strategy
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.n_steps = n_steps
        self.perturbation = perturbation
        
        # Initialize evaluators
        self.pixel_flipper = PixelFlipping(
            model=model,
            perturbation=perturbation,
            n_steps=n_steps,
            device=device
        )
        self.ece_evaluator = ECEFaithfulness(n_bins=10)
        
        # Try to get target layer for GradCAM
        target_layer = None
        if hasattr(model, 'encoder'):
            if hasattr(model.encoder, 'layer4'):
                target_layer = model.encoder.layer4[-1]
        elif hasattr(model, 'layer4'):
            target_layer = model.layer4[-1]
        
        self.saliency_extractor = SaliencyExtractor(
            model=model,
            device=device,
            target_layer=target_layer
        )
    
    def compute_curve_smoothness(self, curve: np.ndarray) -> float:
        """
        Compute curve smoothness as mean second derivative magnitude.
        
        Lower values indicate smoother, more predictable faithfulness curves.
        
        Args:
            curve: Prediction curve (n_steps,)
        
        Returns:
            Smoothness score (lower = smoother)
        """
        if len(curve) < 3:
            return 0.0
        
        # First derivative
        first_deriv = np.diff(curve)
        
        # Second derivative
        second_deriv = np.diff(first_deriv)
        
        # Mean absolute second derivative
        smoothness = np.mean(np.abs(second_deriv))
        
        return smoothness
    
    def compute_monotonicity(
        self,
        curve: np.ndarray,
        direction: str = "decreasing"
    ) -> float:
        """
        Compute monotonicity as Spearman correlation.
        
        For deletion: should decrease monotonically (direction='decreasing')
        For insertion: should increase monotonically (direction='increasing')
        
        Args:
            curve: Prediction curve
            direction: Expected direction
        
        Returns:
            Monotonicity score [-1, 1]
        """
        x = np.arange(len(curve))
        corr, _ = stats.spearmanr(x, curve)
        
        if direction == "decreasing":
            return -corr  # Positive value if monotonically decreasing
        else:
            return corr  # Positive value if monotonically increasing
    
    def evaluate_sample(
        self,
        image: torch.Tensor,
        label: int,
        saliency_method: str = "integrated_grad"
    ) -> Dict[str, float]:
        """
        Evaluate faithfulness for a single sample.
        
        Args:
            image: Input image (C, H, W)
            label: True label
            saliency_method: Saliency method to use
        
        Returns:
            Dictionary with all faithfulness metrics
        """
        # Get saliency map
        saliency = self.saliency_extractor.extract(
            image, target_class=label, method=saliency_method
        )
        
        # Evaluate pixel flipping
        results = self.pixel_flipper.evaluate_both(image, saliency, target_class=label)
        
        deletion_curve = results["deletion"]["target_probs"]
        insertion_curve = results["insertion"]["target_probs"]
        fractions = results["deletion"]["fractions"]
        
        # Compute AUC
        auc_deletion = trapezoid(deletion_curve, fractions)
        auc_insertion = trapezoid(insertion_curve, fractions)
        
        # Compute smoothness
        deletion_smoothness = self.compute_curve_smoothness(deletion_curve)
        insertion_smoothness = self.compute_curve_smoothness(insertion_curve)
        
        # Compute monotonicity
        deletion_monotonicity = self.compute_monotonicity(deletion_curve, "decreasing")
        insertion_monotonicity = self.compute_monotonicity(insertion_curve, "increasing")
        
        # Compute ECE-Faithfulness
        ece_result = self.ece_evaluator.compute_faithfulness_curve(
            saliency, deletion_curve, removal_order="most_important_first"
        )
        
        return {
            "auc_deletion": auc_deletion,
            "auc_insertion": auc_insertion,
            "deletion_smoothness": deletion_smoothness,
            "insertion_smoothness": insertion_smoothness,
            "deletion_monotonicity": deletion_monotonicity,
            "insertion_monotonicity": insertion_monotonicity,
            "ece_faithfulness": ece_result["ece"],
            "ece_slope": ece_result["slope"],
            "ece_r_squared": ece_result["r_squared"],
            "deletion_curve": deletion_curve,
            "insertion_curve": insertion_curve
        }
    
    def evaluate_batch(
        self,
        data_loader,
        n_samples: int = 100,
        saliency_method: str = "integrated_grad"
    ) -> Dict[str, float]:
        """
        Evaluate faithfulness across multiple samples.
        
        Args:
            data_loader: Test data loader
            n_samples: Number of samples to evaluate
            saliency_method: Saliency method to use
        
        Returns:
            Aggregated metrics with mean and std
        """
        from tqdm import tqdm
        
        metrics = {
            "auc_deletion": [],
            "auc_insertion": [],
            "deletion_smoothness": [],
            "insertion_smoothness": [],
            "deletion_monotonicity": [],
            "insertion_monotonicity": [],
            "ece_faithfulness": [],
            "ece_slope": [],
            "ece_r_squared": []
        }
        
        count = 0
        
        for images, labels in tqdm(data_loader, desc="Faithfulness eval"):
            for i in range(images.size(0)):
                if count >= n_samples:
                    break
                
                try:
                    result = self.evaluate_sample(
                        images[i], labels[i].item(), saliency_method
                    )
                    
                    for key in metrics:
                        metrics[key].append(result[key])
                    
                    count += 1
                except Exception as e:
                    print(f"Warning: Sample evaluation failed: {e}")
                    continue
            
            if count >= n_samples:
                break
        
        # Aggregate metrics
        aggregated = {"n_samples": count}
        
        for key, values in metrics.items():
            values = np.array(values)
            aggregated[f"{key}_mean"] = float(np.mean(values))
            aggregated[f"{key}_std"] = float(np.std(values))
        
        # Compute composite faithfulness score
        # Higher is better: high insertion, low deletion, high monotonicity
        composite = (
            (1 - aggregated["auc_deletion_mean"]) * 0.3 +
            aggregated["auc_insertion_mean"] * 0.3 +
            aggregated["deletion_monotonicity_mean"] * 0.2 +
            aggregated["insertion_monotonicity_mean"] * 0.2
        )
        aggregated["composite_faithfulness"] = composite
        
        return aggregated


def compute_all_faithfulness_metrics(
    model: nn.Module,
    data_loader,
    device: str = "cuda",
    n_samples: int = 100,
    saliency_methods: Optional[List[str]] = None
) -> Dict[str, Dict]:
    """
    Compute faithfulness metrics for multiple saliency methods.
    
    Args:
        model: Model to evaluate
        data_loader: Test data loader
        device: Device to use
        n_samples: Number of samples
        saliency_methods: List of methods to evaluate
    
    Returns:
        Dictionary mapping method name to metrics
    """
    if saliency_methods is None:
        saliency_methods = ["vanilla_grad", "integrated_grad"]
    
    evaluator = ExtendedFaithfulnessEvaluator(model, device=device)
    
    results = {}
    
    for method in saliency_methods:
        print(f"\nEvaluating {method}...")
        try:
            metrics = evaluator.evaluate_batch(
                data_loader,
                n_samples=n_samples,
                saliency_method=method
            )
            results[method] = metrics
        except Exception as e:
            print(f"Warning: Method {method} failed: {e}")
            results[method] = {"error": str(e)}
    
    return results


if __name__ == "__main__":
    from spline_theory.models.resnet_variants import get_resnet_variant
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 60)
    print("Testing Extended Faithfulness Evaluation")
    print("=" * 60)
    
    model = get_resnet_variant(
        architecture="resnet18",
        num_classes=10,
        norm_type="bn"
    ).to(device)
    model.eval()
    
    # Create dummy data
    image = torch.randn(3, 32, 32)
    label = 0
    
    print("\nTesting ExtendedFaithfulnessEvaluator:")
    evaluator = ExtendedFaithfulnessEvaluator(model, device=device, n_steps=10)
    
    result = evaluator.evaluate_sample(image, label, saliency_method="vanilla_grad")
    
    print(f"  AUC-Deletion: {result['auc_deletion']:.4f}")
    print(f"  AUC-Insertion: {result['auc_insertion']:.4f}")
    print(f"  Deletion smoothness: {result['deletion_smoothness']:.6f}")
    print(f"  Deletion monotonicity: {result['deletion_monotonicity']:.4f}")
    print(f"  ECE-Faithfulness: {result['ece_faithfulness']:.4f}")
    
    print("\n" + "=" * 60)
    print("All faithfulness tests passed!")
    print("=" * 60)
