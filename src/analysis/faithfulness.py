"""
Faithfulness metrics computation for XAI evaluation.

Includes:
- Traditional metrics (AUC-Deletion, AUC-Insertion, Monotonicity)
- ECE-Faithfulness (Marco's calibration-based metric)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.integrate import trapezoid


class ECEFaithfulness:
    """
    Expected Calibration Error for XAI Faithfulness.
    
    Marco's innovative idea: Saliency maps should behave like calibrated
    probability estimates. If you remove pixels with cumulative importance X%,
    the model confidence should drop by approximately X%.
    
    Perfect faithfulness = linear relationship between cumulative saliency
    removed and confidence drop.
    
    Key insight: This provides a principled way to evaluate XAI methods
    beyond traditional AUC metrics, treating saliency as a probability
    distribution over pixel importance.
    """
    
    def __init__(self, n_bins: int = 10):
        """
        Args:
            n_bins: Number of bins for ECE computation (default 10)
        """
        self.n_bins = n_bins
    
    def normalize_saliency(self, saliency: np.ndarray) -> np.ndarray:
        """
        Normalize saliency map to sum to 1 (probability distribution).
        
        Args:
            saliency: Raw saliency map of any shape
        
        Returns:
            Normalized saliency (same shape, sums to 1)
        """
        saliency_flat = saliency.flatten()
        # Shift to non-negative
        saliency_pos = saliency_flat - saliency_flat.min()
        total = saliency_pos.sum()
        if total < 1e-10:
            # Uniform if saliency is zero everywhere
            return np.ones_like(saliency_flat) / len(saliency_flat)
        return saliency_pos / total
    
    def compute_expected_confidence_drop(
        self,
        saliency: np.ndarray,
        pixel_order: np.ndarray
    ) -> np.ndarray:
        """
        Compute expected confidence drop at each perturbation step.
        
        If removing pixels with cumulative importance X%, expected confidence
        drop is X% (i.e., remaining confidence = 1 - X).
        
        Args:
            saliency: Normalized saliency map (sums to 1)
            pixel_order: Order in which pixels are removed (indices)
        
        Returns:
            Expected confidence at each step (starting at 1.0)
        """
        cumulative_importance = np.cumsum(saliency[pixel_order])
        expected_confidence = 1.0 - cumulative_importance
        # Prepend 1.0 for the initial state (no pixels removed)
        return np.concatenate([[1.0], expected_confidence])
    
    def compute_ece_faithfulness(
        self,
        confidence_curve: np.ndarray,
        expected_curve: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute ECE-Faithfulness: deviation from calibrated behavior.
        
        Args:
            confidence_curve: Actual model confidence at each step
            expected_curve: Expected confidence (based on saliency)
        
        Returns:
            Dictionary with ECE metrics
        """
        # Normalize both curves to start at the same point
        if confidence_curve[0] > 0:
            actual_normalized = confidence_curve / confidence_curve[0]
        else:
            actual_normalized = confidence_curve
        
        # Compute bin-wise calibration error
        n_steps = len(confidence_curve)
        bin_size = n_steps // self.n_bins
        
        bin_errors = []
        for i in range(self.n_bins):
            start_idx = i * bin_size
            end_idx = start_idx + bin_size if i < self.n_bins - 1 else n_steps
            
            bin_actual = actual_normalized[start_idx:end_idx].mean()
            bin_expected = expected_curve[start_idx:end_idx].mean()
            bin_errors.append(abs(bin_actual - bin_expected))
        
        ece = np.mean(bin_errors)
        max_ce = np.max(bin_errors)
        
        # Compute slope and R² of actual vs expected
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            expected_curve, actual_normalized
        )
        
        # Perfect calibration: slope = 1, R² = 1
        return {
            'ece': ece,
            'max_calibration_error': max_ce,
            'slope': slope,  # Should be ~1.0 for perfect faithfulness
            'r_squared': r_value ** 2,  # Higher = more predictable
            'intercept': intercept,  # Should be ~0 for perfect faithfulness
        }
    
    def compute_faithfulness_curve(
        self,
        saliency: np.ndarray,
        confidence_curve: np.ndarray,
        removal_order: str = 'most_important_first'
    ) -> Dict[str, np.ndarray]:
        """
        Compute full faithfulness analysis for a single sample.
        
        Args:
            saliency: Raw saliency map
            confidence_curve: Model confidence at each perturbation step
            removal_order: 'most_important_first' or 'least_important_first'
        
        Returns:
            Dictionary with curves and metrics
        """
        # Normalize saliency
        saliency_norm = self.normalize_saliency(saliency)
        
        # Get pixel removal order
        if removal_order == 'most_important_first':
            pixel_order = np.argsort(saliency_norm)[::-1]
        else:
            pixel_order = np.argsort(saliency_norm)
        
        # Compute expected curve
        expected_curve = self.compute_expected_confidence_drop(
            saliency_norm, pixel_order
        )
        
        # Truncate to match confidence curve length
        n_steps = len(confidence_curve)
        step_size = len(saliency_norm) // (n_steps - 1)
        expected_sampled = expected_curve[::step_size][:n_steps]
        
        # Ensure same length
        if len(expected_sampled) < n_steps:
            expected_sampled = np.interp(
                np.linspace(0, 1, n_steps),
                np.linspace(0, 1, len(expected_sampled)),
                expected_sampled
            )
        
        # Compute ECE metrics
        metrics = self.compute_ece_faithfulness(confidence_curve, expected_sampled)
        
        return {
            'actual_curve': confidence_curve,
            'expected_curve': expected_sampled,
            'fractions': np.linspace(0, 1, n_steps),
            **metrics
        }
    
    def evaluate_batch(
        self,
        saliencies: List[np.ndarray],
        confidence_curves: List[np.ndarray],
        removal_order: str = 'most_important_first'
    ) -> Dict[str, float]:
        """
        Evaluate ECE-Faithfulness across multiple samples.
        
        Args:
            saliencies: List of saliency maps
            confidence_curves: List of confidence curves
            removal_order: Pixel removal order
        
        Returns:
            Aggregated metrics with mean and std
        """
        all_metrics = {
            'ece': [],
            'max_calibration_error': [],
            'slope': [],
            'r_squared': [],
            'intercept': []
        }
        
        for saliency, conf_curve in zip(saliencies, confidence_curves):
            result = self.compute_faithfulness_curve(
                saliency, conf_curve, removal_order
            )
            for key in all_metrics:
                all_metrics[key].append(result[key])
        
        # Aggregate
        aggregated = {}
        for key, values in all_metrics.items():
            values = np.array(values)
            aggregated[f'{key}_mean'] = float(np.mean(values))
            aggregated[f'{key}_std'] = float(np.std(values))
        
        aggregated['n_samples'] = len(saliencies)
        
        # Compute overall calibration quality score
        # Perfect: slope=1, r²=1, ece=0
        slope_error = abs(1.0 - aggregated['slope_mean'])
        calibration_score = (
            aggregated['r_squared_mean'] 
            * (1 - slope_error) 
            * (1 - aggregated['ece_mean'])
        )
        aggregated['calibration_score'] = float(calibration_score)
        
        return aggregated


class FaithfulnessEvaluator:
    """Compute and compare faithfulness metrics."""
    
    def compute_auc_deletion(self, curve: np.ndarray) -> float:
        x = np.linspace(0, 1, len(curve))
        return np.trapz(curve, x)
    
    def compute_auc_insertion(self, curve: np.ndarray) -> float:
        x = np.linspace(0, 1, len(curve))
        return np.trapz(curve, x)
    
    def compute_monotonicity(self, curve: np.ndarray, direction: str = 'decreasing') -> float:
        x = np.arange(len(curve))
        corr, _ = stats.spearmanr(x, curve)
        return -corr if direction == 'decreasing' else corr
    
    def evaluate(self, deletion_curves: np.ndarray, insertion_curves: np.ndarray) -> Dict:
        del_aucs = [self.compute_auc_deletion(c) for c in deletion_curves]
        ins_aucs = [self.compute_auc_insertion(c) for c in insertion_curves]
        
        return {
            'deletion_auc_mean': np.mean(del_aucs),
            'deletion_auc_std': np.std(del_aucs),
            'insertion_auc_mean': np.mean(ins_aucs),
            'insertion_auc_std': np.std(ins_aucs),
            'n_samples': len(deletion_curves)
        }


def compute_faithfulness_metrics(del_curves: np.ndarray, ins_curves: np.ndarray) -> Dict:
    return FaithfulnessEvaluator().evaluate(del_curves, ins_curves)


def compute_method_agreement(rankings: Dict[str, np.ndarray]):
    methods = list(rankings.keys())
    n = len(methods)
    tau_matrix = np.eye(n)
    
    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            if i < j:
                tau, _ = stats.kendalltau(rankings[m1].flatten(), rankings[m2].flatten())
                tau_matrix[i, j] = tau_matrix[j, i] = tau
    
    return tau_matrix, methods
