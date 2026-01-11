"""Evaluation modules for spline theory experiments."""

from .adversarial import (
    AdversarialEvaluator,
    pgd_attack,
    fgsm_attack,
)
from .geometric import (
    LocalComplexityAnalyzer,
    activation_pattern_diversity,
    estimate_partition_density,
)
from .faithfulness_extended import (
    ExtendedFaithfulnessEvaluator,
    compute_all_faithfulness_metrics,
)

__all__ = [
    "AdversarialEvaluator",
    "pgd_attack",
    "fgsm_attack",
    "LocalComplexityAnalyzer",
    "activation_pattern_diversity",
    "estimate_partition_density",
    "ExtendedFaithfulnessEvaluator",
    "compute_all_faithfulness_metrics",
]
