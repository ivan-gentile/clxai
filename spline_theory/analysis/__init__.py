"""Analysis and visualization for spline theory experiments."""

from .grokking_detection import (
    GrokkingDetector,
    detect_grokking_transition,
    compute_gradient_norms,
)
from .comparison import (
    GeometricComparison,
    compare_ce_vs_cl,
    generate_comparison_report,
)

__all__ = [
    "GrokkingDetector",
    "detect_grokking_transition",
    "compute_gradient_norms",
    "GeometricComparison",
    "compare_ce_vs_cl",
    "generate_comparison_report",
]
