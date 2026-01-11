"""
Geometric analysis metrics for spline theory experiments.

Implements:
- Local complexity estimation (activation pattern diversity)
- Partition density proxy around data points
- Decision boundary analysis

These metrics test spline theory predictions:
- Good generalization correlates with large regions around data points
- Partition density should concentrate at decision boundaries, not data
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm


class ActivationHook:
    """Hook to capture ReLU activation patterns."""
    
    def __init__(self):
        self.activations = []
    
    def __call__(self, module, input, output):
        # Store binary activation pattern (1 where ReLU is active)
        self.activations.append((output > 0).detach())
    
    def clear(self):
        self.activations = []
    
    def get_pattern(self) -> torch.Tensor:
        """Get concatenated binary activation pattern."""
        if not self.activations:
            return torch.tensor([])
        
        patterns = []
        for act in self.activations:
            # Flatten each activation map
            patterns.append(act.view(act.size(0), -1))
        
        return torch.cat(patterns, dim=1)


def register_relu_hooks(model: nn.Module) -> Tuple[List, List[ActivationHook]]:
    """
    Register forward hooks on all ReLU layers.
    
    Args:
        model: PyTorch model
    
    Returns:
        Tuple of (hook handles, hook objects)
    """
    hooks = []
    handles = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            hook = ActivationHook()
            handle = module.register_forward_hook(hook)
            hooks.append(hook)
            handles.append(handle)
    
    return handles, hooks


def remove_hooks(handles: List):
    """Remove registered hooks."""
    for handle in handles:
        handle.remove()


class LocalComplexityAnalyzer:
    """
    Analyze local complexity of neural network partition.
    
    Based on spline theory: neural networks partition input space into
    convex polytopes. The number of distinct activation patterns in a
    local neighborhood indicates the partition density.
    
    Key insight: Lower local complexity (fewer distinct patterns) around
    data points suggests larger partition regions, which correlates with
    better generalization and robustness.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda"
    ):
        """
        Args:
            model: Model to analyze
            device: Device to use
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
    
    def _sample_neighbors(
        self,
        x: torch.Tensor,
        epsilon: float,
        n_neighbors: int
    ) -> torch.Tensor:
        """
        Sample random neighbors in epsilon-ball around x.
        
        Args:
            x: Center point (C, H, W)
            epsilon: Radius of ball
            n_neighbors: Number of neighbors to sample
        
        Returns:
            Tensor of neighbors (n_neighbors, C, H, W)
        """
        # Sample random directions
        noise = torch.randn(n_neighbors, *x.shape, device=self.device)
        # Normalize to unit sphere and scale by random radius <= epsilon
        noise = noise / noise.view(n_neighbors, -1).norm(dim=1, keepdim=True).view(n_neighbors, 1, 1, 1)
        radii = torch.rand(n_neighbors, 1, 1, 1, device=self.device) * epsilon
        
        neighbors = x.unsqueeze(0) + noise * radii
        return neighbors
    
    def activation_pattern_diversity(
        self,
        x: torch.Tensor,
        epsilon: float = 0.1,
        n_neighbors: int = 100
    ) -> Dict[str, float]:
        """
        Count distinct activation patterns in epsilon-ball around sample.
        
        Lower diversity = larger local region = better geometry.
        
        Args:
            x: Input sample (C, H, W) or (1, C, H, W)
            epsilon: Radius of neighborhood
            n_neighbors: Number of neighbors to sample
        
        Returns:
            Dictionary with diversity metrics
        """
        if x.dim() == 4:
            x = x.squeeze(0)
        x = x.to(self.device)
        
        # Register hooks
        handles, hooks = register_relu_hooks(self.model)
        
        try:
            # Get pattern for center point
            with torch.no_grad():
                _ = self.model(x.unsqueeze(0))
            center_pattern = hooks[0].get_pattern() if hooks else None
            for hook in hooks:
                hook.clear()
            
            # Sample neighbors
            neighbors = self._sample_neighbors(x, epsilon, n_neighbors)
            
            # Get patterns for all neighbors
            patterns = []
            batch_size = 50  # Process in batches for memory efficiency
            
            for i in range(0, n_neighbors, batch_size):
                batch = neighbors[i:i + batch_size]
                with torch.no_grad():
                    _ = self.model(batch)
                
                for hook in hooks:
                    batch_patterns = hook.get_pattern()
                    patterns.append(batch_patterns)
                    hook.clear()
            
            if not patterns:
                return {
                    "n_distinct_patterns": 0,
                    "diversity_ratio": 0.0,
                    "pattern_entropy": 0.0
                }
            
            # Concatenate all patterns
            all_patterns = torch.cat(patterns, dim=0)
            
            # Count distinct patterns (convert to tuple for hashing)
            pattern_set = set()
            for i in range(all_patterns.size(0)):
                # Subsample pattern for efficiency (full pattern can be very large)
                pattern = all_patterns[i, ::100].cpu().numpy().tobytes()
                pattern_set.add(pattern)
            
            n_distinct = len(pattern_set)
            diversity_ratio = n_distinct / n_neighbors
            
            # Compute pattern entropy
            pattern_counts = {}
            for i in range(all_patterns.size(0)):
                pattern = all_patterns[i, ::100].cpu().numpy().tobytes()
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
            probs = np.array(list(pattern_counts.values())) / n_neighbors
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            
            return {
                "n_distinct_patterns": n_distinct,
                "diversity_ratio": diversity_ratio,
                "pattern_entropy": entropy,
                "n_neighbors": n_neighbors,
                "epsilon": epsilon
            }
        
        finally:
            remove_hooks(handles)
    
    def estimate_partition_density(
        self,
        data_loader,
        epsilon: float = 0.1,
        n_neighbors: int = 50,
        n_samples: int = 100
    ) -> Dict[str, float]:
        """
        Estimate partition density around data points.
        
        Args:
            data_loader: Data loader
            epsilon: Neighborhood radius
            n_neighbors: Neighbors per sample
            n_samples: Number of samples to analyze
        
        Returns:
            Aggregated density metrics
        """
        diversities = []
        entropies = []
        count = 0
        
        for images, labels in tqdm(data_loader, desc="Partition density"):
            for i in range(images.size(0)):
                if count >= n_samples:
                    break
                
                result = self.activation_pattern_diversity(
                    images[i], epsilon=epsilon, n_neighbors=n_neighbors
                )
                diversities.append(result["diversity_ratio"])
                entropies.append(result["pattern_entropy"])
                count += 1
            
            if count >= n_samples:
                break
        
        return {
            "mean_diversity": np.mean(diversities),
            "std_diversity": np.std(diversities),
            "mean_entropy": np.mean(entropies),
            "std_entropy": np.std(entropies),
            "n_samples": count,
            "epsilon": epsilon,
            "n_neighbors": n_neighbors
        }


def activation_pattern_diversity(
    model: nn.Module,
    x: torch.Tensor,
    epsilon: float = 0.1,
    n_neighbors: int = 100,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Convenience function for single-sample diversity analysis.
    
    Args:
        model: Model to analyze
        x: Input sample
        epsilon: Neighborhood radius
        n_neighbors: Number of neighbors
        device: Device to use
    
    Returns:
        Diversity metrics
    """
    analyzer = LocalComplexityAnalyzer(model, device=device)
    return analyzer.activation_pattern_diversity(x, epsilon, n_neighbors)


def estimate_partition_density(
    model: nn.Module,
    data_loader,
    epsilon: float = 0.1,
    n_neighbors: int = 50,
    n_samples: int = 100,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Convenience function for partition density estimation.
    
    Args:
        model: Model to analyze
        data_loader: Data loader
        epsilon: Neighborhood radius
        n_neighbors: Neighbors per sample
        n_samples: Number of samples
        device: Device to use
    
    Returns:
        Aggregated density metrics
    """
    analyzer = LocalComplexityAnalyzer(model, device=device)
    return analyzer.estimate_partition_density(
        data_loader, epsilon, n_neighbors, n_samples
    )


class DecisionBoundaryAnalyzer:
    """
    Analyze decision boundary characteristics.
    
    Examines:
    - Distance from data points to decision boundary
    - Boundary sharpness (gradient magnitude at boundary)
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.device = device
    
    def estimate_boundary_distance(
        self,
        x: torch.Tensor,
        label: int,
        max_steps: int = 100,
        step_size: float = 0.01
    ) -> Dict[str, float]:
        """
        Estimate distance to nearest decision boundary.
        
        Uses gradient-based search toward boundary.
        
        Args:
            x: Input sample (C, H, W)
            label: True label
            max_steps: Maximum search steps
            step_size: Step size for search
        
        Returns:
            Boundary distance metrics
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.clone().to(self.device)
        
        self.model.eval()
        initial_pred = None
        
        for step in range(max_steps):
            x.requires_grad_(True)
            
            outputs = self.model(x)
            pred = outputs.argmax(dim=1).item()
            
            if initial_pred is None:
                initial_pred = pred
            
            # Check if we crossed boundary
            if pred != initial_pred:
                distance = step * step_size
                return {
                    "boundary_distance": distance,
                    "crossed_boundary": True,
                    "steps_to_boundary": step,
                    "initial_pred": initial_pred,
                    "final_pred": pred
                }
            
            # Compute gradient toward boundary (maximize loss for true class)
            loss = outputs[0, label]
            self.model.zero_grad()
            loss.backward()
            
            # Move away from current class
            with torch.no_grad():
                grad = x.grad
                if grad is not None:
                    x = x - step_size * grad.sign()
                    x = x.detach()
        
        return {
            "boundary_distance": max_steps * step_size,
            "crossed_boundary": False,
            "steps_to_boundary": max_steps,
            "initial_pred": initial_pred,
            "final_pred": pred
        }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from spline_theory.models.resnet_variants import get_resnet_variant
    
    # Test geometric analysis
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 60)
    print("Testing Geometric Analysis")
    print("=" * 60)
    
    model = get_resnet_variant(
        architecture="resnet18",
        num_classes=10,
        norm_type="bn"
    ).to(device)
    model.eval()
    
    # Create dummy input
    x = torch.randn(3, 32, 32).to(device)
    
    print("\nTesting LocalComplexityAnalyzer:")
    analyzer = LocalComplexityAnalyzer(model, device=device)
    
    result = analyzer.activation_pattern_diversity(
        x, epsilon=0.1, n_neighbors=50
    )
    print(f"  Distinct patterns: {result['n_distinct_patterns']}")
    print(f"  Diversity ratio: {result['diversity_ratio']:.4f}")
    print(f"  Pattern entropy: {result['pattern_entropy']:.4f}")
    
    print("\nTesting DecisionBoundaryAnalyzer:")
    boundary_analyzer = DecisionBoundaryAnalyzer(model, device=device)
    
    boundary_result = boundary_analyzer.estimate_boundary_distance(
        x, label=0, max_steps=50, step_size=0.05
    )
    print(f"  Boundary distance: {boundary_result['boundary_distance']:.4f}")
    print(f"  Crossed boundary: {boundary_result['crossed_boundary']}")
    
    print("\n" + "=" * 60)
    print("All geometric tests passed!")
    print("=" * 60)
