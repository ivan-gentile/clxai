"""
Comparative analysis between CE and CL models.

Implements:
- Geometric comparison (partition density, local complexity)
- Faithfulness comparison
- Adversarial robustness comparison
- Generation of comparison reports

Tests the core hypothesis: CL models achieve geometric properties
that CE models only reach through grokking.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class GeometricComparison:
    """
    Compare geometric properties between models.
    
    Key metrics:
    - Local complexity (activation pattern diversity)
    - Decision boundary distances
    - Embedding space characteristics
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
    
    def compare_local_complexity(
        self,
        model_ce: nn.Module,
        model_cl: nn.Module,
        data_loader,
        epsilon: float = 0.1,
        n_neighbors: int = 50,
        n_samples: int = 50
    ) -> Dict:
        """
        Compare local complexity between CE and CL models.
        
        Args:
            model_ce: Cross-entropy trained model
            model_cl: Contrastive learning trained model
            data_loader: Test data loader
            epsilon: Neighborhood radius
            n_neighbors: Neighbors per sample
            n_samples: Number of samples
        
        Returns:
            Comparison results
        """
        from spline_theory.evaluation.geometric import LocalComplexityAnalyzer
        
        analyzer_ce = LocalComplexityAnalyzer(model_ce, device=self.device)
        analyzer_cl = LocalComplexityAnalyzer(model_cl, device=self.device)
        
        ce_results = analyzer_ce.estimate_partition_density(
            data_loader, epsilon, n_neighbors, n_samples
        )
        cl_results = analyzer_cl.estimate_partition_density(
            data_loader, epsilon, n_neighbors, n_samples
        )
        
        return {
            "ce_diversity": ce_results["mean_diversity"],
            "cl_diversity": cl_results["mean_diversity"],
            "ce_entropy": ce_results["mean_entropy"],
            "cl_entropy": cl_results["mean_entropy"],
            "diversity_ratio": ce_results["mean_diversity"] / (cl_results["mean_diversity"] + 1e-10),
            "entropy_ratio": ce_results["mean_entropy"] / (cl_results["mean_entropy"] + 1e-10),
            "cl_has_lower_complexity": cl_results["mean_diversity"] < ce_results["mean_diversity"]
        }
    
    def compare_adversarial_robustness(
        self,
        model_ce: nn.Module,
        model_cl: nn.Module,
        data_loader,
        n_samples: int = 500
    ) -> Dict:
        """
        Compare adversarial robustness between models.
        
        Args:
            model_ce: Cross-entropy trained model
            model_cl: Contrastive learning trained model
            data_loader: Test data loader
            n_samples: Number of samples
        
        Returns:
            Robustness comparison
        """
        from spline_theory.evaluation.adversarial import AdversarialEvaluator
        
        eval_ce = AdversarialEvaluator(model_ce, device=self.device)
        eval_cl = AdversarialEvaluator(model_cl, device=self.device)
        
        ce_results = eval_ce.evaluate_all(data_loader, n_samples=n_samples)
        cl_results = eval_cl.evaluate_all(data_loader, n_samples=n_samples)
        
        return {
            "ce_clean_acc": ce_results["clean_accuracy"],
            "cl_clean_acc": cl_results["clean_accuracy"],
            "ce_fgsm_acc": ce_results["fgsm_accuracy"],
            "cl_fgsm_acc": cl_results["fgsm_accuracy"],
            "ce_pgd_acc": ce_results["pgd_accuracy"],
            "cl_pgd_acc": cl_results["pgd_accuracy"],
            "fgsm_improvement": cl_results["fgsm_accuracy"] - ce_results["fgsm_accuracy"],
            "pgd_improvement": cl_results["pgd_accuracy"] - ce_results["pgd_accuracy"],
            "cl_more_robust": cl_results["pgd_accuracy"] > ce_results["pgd_accuracy"]
        }
    
    def compare_faithfulness(
        self,
        model_ce: nn.Module,
        model_cl: nn.Module,
        data_loader,
        n_samples: int = 100
    ) -> Dict:
        """
        Compare faithfulness metrics between models.
        
        Args:
            model_ce: Cross-entropy trained model
            model_cl: Contrastive learning trained model
            data_loader: Test data loader
            n_samples: Number of samples
        
        Returns:
            Faithfulness comparison
        """
        from spline_theory.evaluation.faithfulness_extended import ExtendedFaithfulnessEvaluator
        
        eval_ce = ExtendedFaithfulnessEvaluator(model_ce, device=self.device)
        eval_cl = ExtendedFaithfulnessEvaluator(model_cl, device=self.device)
        
        ce_results = eval_ce.evaluate_batch(data_loader, n_samples=n_samples)
        cl_results = eval_cl.evaluate_batch(data_loader, n_samples=n_samples)
        
        return {
            "ce_auc_deletion": ce_results["auc_deletion_mean"],
            "cl_auc_deletion": cl_results["auc_deletion_mean"],
            "ce_auc_insertion": ce_results["auc_insertion_mean"],
            "cl_auc_insertion": cl_results["auc_insertion_mean"],
            "ce_monotonicity": ce_results["deletion_monotonicity_mean"],
            "cl_monotonicity": cl_results["deletion_monotonicity_mean"],
            "ce_composite": ce_results["composite_faithfulness"],
            "cl_composite": cl_results["composite_faithfulness"],
            "deletion_improvement": ce_results["auc_deletion_mean"] - cl_results["auc_deletion_mean"],
            "insertion_improvement": cl_results["auc_insertion_mean"] - ce_results["auc_insertion_mean"],
            "cl_more_faithful": cl_results["composite_faithfulness"] > ce_results["composite_faithfulness"]
        }


def compare_ce_vs_cl(
    model_ce: nn.Module,
    model_cl: nn.Module,
    data_loader,
    device: str = "cuda",
    run_geometric: bool = True,
    run_adversarial: bool = True,
    run_faithfulness: bool = True,
    n_samples: int = 100
) -> Dict:
    """
    Run comprehensive comparison between CE and CL models.
    
    Args:
        model_ce: Cross-entropy trained model
        model_cl: Contrastive learning trained model
        data_loader: Test data loader
        device: Device to use
        run_geometric: Whether to run geometric comparison
        run_adversarial: Whether to run adversarial comparison
        run_faithfulness: Whether to run faithfulness comparison
        n_samples: Number of samples for evaluation
    
    Returns:
        Comprehensive comparison results
    """
    comparison = GeometricComparison(device=device)
    results = {}
    
    if run_geometric:
        print("Running geometric comparison...")
        results["geometric"] = comparison.compare_local_complexity(
            model_ce, model_cl, data_loader, n_samples=min(50, n_samples)
        )
    
    if run_adversarial:
        print("Running adversarial comparison...")
        results["adversarial"] = comparison.compare_adversarial_robustness(
            model_ce, model_cl, data_loader, n_samples=n_samples
        )
    
    if run_faithfulness:
        print("Running faithfulness comparison...")
        results["faithfulness"] = comparison.compare_faithfulness(
            model_ce, model_cl, data_loader, n_samples=n_samples
        )
    
    # Compute overall hypothesis support
    hypothesis_support = {
        "P1_early_robustness": results.get("adversarial", {}).get("cl_more_robust", False),
        "P4_lower_complexity": results.get("geometric", {}).get("cl_has_lower_complexity", False),
        "faithfulness_improvement": results.get("faithfulness", {}).get("cl_more_faithful", False)
    }
    
    n_supported = sum(hypothesis_support.values())
    results["hypothesis_support"] = hypothesis_support
    results["n_hypotheses_supported"] = n_supported
    results["overall_support_ratio"] = n_supported / len(hypothesis_support)
    
    return results


def generate_comparison_report(
    results: Dict,
    output_path: Optional[str] = None
) -> str:
    """
    Generate human-readable comparison report.
    
    Args:
        results: Results from compare_ce_vs_cl
        output_path: Optional path to save report
    
    Returns:
        Report string
    """
    lines = [
        "=" * 70,
        "SPLINE THEORY HYPOTHESIS TEST: CE vs CL COMPARISON",
        "=" * 70,
        ""
    ]
    
    # Geometric comparison
    if "geometric" in results:
        geo = results["geometric"]
        lines.extend([
            "GEOMETRIC ANALYSIS (Local Complexity)",
            "-" * 40,
            f"  CE model diversity: {geo['ce_diversity']:.4f}",
            f"  CL model diversity: {geo['cl_diversity']:.4f}",
            f"  Diversity ratio (CE/CL): {geo['diversity_ratio']:.2f}",
            f"  --> CL has lower complexity: {geo['cl_has_lower_complexity']}",
            ""
        ])
    
    # Adversarial comparison
    if "adversarial" in results:
        adv = results["adversarial"]
        lines.extend([
            "ADVERSARIAL ROBUSTNESS",
            "-" * 40,
            f"  CE clean accuracy: {adv['ce_clean_acc']*100:.2f}%",
            f"  CL clean accuracy: {adv['cl_clean_acc']*100:.2f}%",
            f"  CE PGD accuracy: {adv['ce_pgd_acc']*100:.2f}%",
            f"  CL PGD accuracy: {adv['cl_pgd_acc']*100:.2f}%",
            f"  PGD improvement: {adv['pgd_improvement']*100:+.2f}%",
            f"  --> CL more robust: {adv['cl_more_robust']}",
            ""
        ])
    
    # Faithfulness comparison
    if "faithfulness" in results:
        faith = results["faithfulness"]
        lines.extend([
            "FAITHFULNESS METRICS",
            "-" * 40,
            f"  CE composite faithfulness: {faith['ce_composite']:.4f}",
            f"  CL composite faithfulness: {faith['cl_composite']:.4f}",
            f"  CE AUC-Deletion: {faith['ce_auc_deletion']:.4f}",
            f"  CL AUC-Deletion: {faith['cl_auc_deletion']:.4f}",
            f"  --> CL more faithful: {faith['cl_more_faithful']}",
            ""
        ])
    
    # Hypothesis support summary
    lines.extend([
        "=" * 70,
        "HYPOTHESIS SUPPORT SUMMARY",
        "=" * 70,
    ])
    
    for hyp, supported in results.get("hypothesis_support", {}).items():
        status = "SUPPORTED" if supported else "NOT SUPPORTED"
        lines.append(f"  {hyp}: {status}")
    
    lines.extend([
        "",
        f"Overall support ratio: {results.get('overall_support_ratio', 0):.1%}",
        "=" * 70
    ])
    
    report = "\n".join(lines)
    
    if output_path:
        with open(output_path, "w") as f:
            f.write(report)
        
        # Also save raw results as JSON
        json_path = output_path.replace(".txt", ".json")
        with open(json_path, "w") as f:
            # Convert numpy types for JSON serialization
            def convert(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, dict):
                    return {k: convert(v) for k, v in obj.items()}
                return obj
            
            json.dump(convert(results), f, indent=2)
    
    return report


if __name__ == "__main__":
    from spline_theory.models.resnet_variants import get_resnet_variant
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 60)
    print("Testing CE vs CL Comparison")
    print("=" * 60)
    
    # Create two dummy models (in practice, these would be loaded from checkpoints)
    model_ce = get_resnet_variant(
        architecture="resnet18",
        num_classes=10,
        norm_type="bn"
    ).to(device)
    
    model_cl = get_resnet_variant(
        architecture="resnet18",
        num_classes=10,
        norm_type="bn"
    ).to(device)
    
    # Create dummy data loader
    from torch.utils.data import TensorDataset, DataLoader
    
    dummy_images = torch.randn(32, 3, 32, 32)
    dummy_labels = torch.randint(0, 10, (32,))
    dataset = TensorDataset(dummy_images, dummy_labels)
    loader = DataLoader(dataset, batch_size=8)
    
    print("\nRunning comparison (with dummy models)...")
    
    results = compare_ce_vs_cl(
        model_ce, model_cl, loader,
        device=device,
        run_geometric=False,  # Skip for speed
        run_adversarial=True,
        run_faithfulness=False,  # Skip for speed
        n_samples=16
    )
    
    report = generate_comparison_report(results)
    print(report)
    
    print("\n" + "=" * 60)
    print("Comparison test passed!")
    print("=" * 60)
