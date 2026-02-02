#!/usr/bin/env python3
"""
CLXAI - XAI Results Comparison Script

Compares XAI metrics between CE and SCL models.
Generates summary statistics and comparison plots.

Usage:
    python scripts/compare_xai_results.py --results_dir results/xai_eval
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy import stats


def parse_args():
    parser = argparse.ArgumentParser(description="Compare CE vs SCL XAI results")
    parser.add_argument("--results_dir", type=str, default="results/xai_eval",
                        help="Directory containing evaluation results")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for plots (default: results_dir)")
    parser.add_argument("--metric", type=str, default="all",
                        choices=["all", "pf", "irof", "sparseness", "complexity", 
                                 "robustness", "contrastivity"],
                        help="Metric to analyze")
    return parser.parse_args()


def load_results(results_dir, metric):
    """Load CE and SCL results for a given metric."""
    results_dir = Path(results_dir)
    
    ce_path = results_dir / f"ce_{metric}_scores.csv"
    scl_path = results_dir / f"scl_{metric}_scores.csv"
    
    if not ce_path.exists():
        print(f"  CE results not found: {ce_path}")
        return None, None
    if not scl_path.exists():
        print(f"  SCL results not found: {scl_path}")
        return None, None
    
    ce_df = pd.read_csv(ce_path)
    scl_df = pd.read_csv(scl_path)
    
    return ce_df, scl_df


def filter_correct_predictions(ce_df, scl_df):
    """Filter to samples where both models predict correctly."""
    mask_ce = ce_df["true"] == ce_df["pred"]
    mask_scl = scl_df["true"] == scl_df["pred"]
    final_mask = mask_ce & mask_scl
    
    ce_filtered = ce_df[final_mask].copy().reset_index(drop=True)
    scl_filtered = scl_df[final_mask].copy().reset_index(drop=True)
    
    return ce_filtered, scl_filtered


def compute_statistics(ce_df, scl_df, metric_cols):
    """Compute comparison statistics."""
    results = []
    
    for col in metric_cols:
        ce_vals = ce_df[col].dropna()
        scl_vals = scl_df[col].dropna()
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(ce_vals, scl_vals)
        
        # Effect size (Cohen's d)
        diff = ce_vals - scl_vals
        cohens_d = diff.mean() / diff.std()
        
        results.append({
            "metric": col,
            "ce_mean": ce_vals.mean(),
            "ce_std": ce_vals.std(),
            "scl_mean": scl_vals.mean(),
            "scl_std": scl_vals.std(),
            "diff_mean": diff.mean(),
            "t_stat": t_stat,
            "p_value": p_value,
            "cohens_d": cohens_d,
            "significant": p_value < 0.05
        })
    
    return pd.DataFrame(results)


def plot_comparison(ce_df, scl_df, metric_cols, metric_name, output_dir):
    """Create comparison plot."""
    n_metrics = len(metric_cols)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]
    
    for ax, col in zip(axes, metric_cols):
        ce_vals = ce_df[col].dropna()
        scl_vals = scl_df[col].dropna()
        
        # Box plot
        data = [ce_vals, scl_vals]
        bp = ax.boxplot(data, labels=["CE", "SCL"], patch_artist=True)
        
        colors = ["#3498db", "#e74c3c"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title(col.replace("_", " "), fontsize=12)
        ax.set_ylabel("Score")
        ax.grid(True, alpha=0.3)
        
        # Add mean annotations
        ax.axhline(ce_vals.mean(), color="#3498db", linestyle="--", alpha=0.8, label=f"CE mean: {ce_vals.mean():.3f}")
        ax.axhline(scl_vals.mean(), color="#e74c3c", linestyle="--", alpha=0.8, label=f"SCL mean: {scl_vals.mean():.3f}")
        ax.legend(fontsize=8)
    
    plt.suptitle(f"{metric_name.upper()} Comparison: CE vs SCL", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    output_path = output_dir / f"comparison_{metric_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {output_path}")


def plot_per_sample_comparison(ce_df, scl_df, metric_col, metric_name, output_dir):
    """Create per-sample comparison plot (like in notebook)."""
    ce_vals = ce_df[metric_col].values
    scl_vals = scl_df[metric_col].values
    
    # Sort by difference
    diff = ce_vals - scl_vals
    sorted_indices = np.argsort(diff)
    
    ce_sorted = ce_vals[sorted_indices]
    scl_sorted = scl_vals[sorted_indices]
    diff_sorted = diff[sorted_indices]
    
    plt.figure(figsize=(20, 8))
    x = np.arange(len(ce_sorted))
    
    # Color by which is better
    colors = ["#2ca02c" if d < 0 else "#d62728" for d in diff_sorted]
    
    # Vertical lines connecting points
    plt.vlines(x, ymin=np.minimum(ce_sorted, scl_sorted), 
               ymax=np.maximum(ce_sorted, scl_sorted),
               colors=colors, alpha=0.5, linewidth=1)
    
    # Points
    plt.scatter(x, ce_sorted, color="#3498db", s=20, label=f"CE (mean: {ce_vals.mean():.3f})", zorder=5)
    plt.scatter(x, scl_sorted, color="#e74c3c", s=20, label=f"SCL (mean: {scl_vals.mean():.3f})", zorder=5)
    
    plt.xlabel("Sample (sorted by CE-SCL difference)", fontsize=12)
    plt.ylabel(metric_col.replace("_", " "), fontsize=12)
    plt.title(f"{metric_col}: CE vs SCL Per-Sample Comparison", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    output_path = output_dir / f"per_sample_{metric_name}_{metric_col}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {output_path}")


def analyze_metric(metric_name, results_dir, output_dir):
    """Analyze a single metric."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {metric_name.upper()}")
    print(f"{'='*60}")
    
    ce_df, scl_df = load_results(results_dir, metric_name)
    if ce_df is None or scl_df is None:
        return None
    
    # Filter to correct predictions
    ce_filtered, scl_filtered = filter_correct_predictions(ce_df, scl_df)
    print(f"Samples: {len(ce_df)} total, {len(ce_filtered)} both correct")
    
    # Accuracies
    ce_acc = (ce_df["true"] == ce_df["pred"]).mean() * 100
    scl_acc = (scl_df["true"] == scl_df["pred"]).mean() * 100
    print(f"CE accuracy: {ce_acc:.2f}%")
    print(f"SCL accuracy: {scl_acc:.2f}%")
    
    # Get metric columns
    exclude_cols = ["idx", "true", "pred", "contrast_class"]
    metric_cols = [c for c in ce_filtered.columns if c not in exclude_cols]
    
    # Compute statistics
    stats_df = compute_statistics(ce_filtered, scl_filtered, metric_cols)
    print("\nStatistics:")
    print(stats_df.to_string(index=False))
    
    # Save statistics
    stats_path = output_dir / f"stats_{metric_name}.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"\nSaved: {stats_path}")
    
    # Create plots
    plot_comparison(ce_filtered, scl_filtered, metric_cols, metric_name, output_dir)
    
    # Per-sample plot for first metric column
    if len(metric_cols) > 0:
        plot_per_sample_comparison(ce_filtered, scl_filtered, metric_cols[0], 
                                   metric_name, output_dir)
    
    return stats_df


def main():
    args = parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("CLXAI XAI Results Comparison")
    print(f"{'='*60}")
    print(f"Results dir: {results_dir}")
    print(f"Output dir: {output_dir}")
    
    # Determine which metrics to analyze
    metrics = ["pf", "irof", "sparseness", "complexity", "robustness", "contrastivity"]
    if args.metric != "all":
        metrics = [args.metric]
    
    # Analyze each metric
    all_stats = []
    for metric in metrics:
        stats_df = analyze_metric(metric, results_dir, output_dir)
        if stats_df is not None:
            stats_df["category"] = metric
            all_stats.append(stats_df)
    
    # Combined summary
    if all_stats:
        combined = pd.concat(all_stats, ignore_index=True)
        summary_path = output_dir / "combined_statistics.csv"
        combined.to_csv(summary_path, index=False)
        print(f"\n{'='*60}")
        print(f"Combined statistics saved: {summary_path}")
    
    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
