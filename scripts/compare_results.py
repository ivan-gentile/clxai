#!/usr/bin/env python3
"""
Compare XAI evaluation results across models (CE, SCL, Triplet).

Loads evaluation CSVs, filters to correctly-predicted samples, computes
summary statistics with paired t-tests, and generates comparison plots.

Usage:
    python scripts/compare_results.py \
        --results_dirs results/xai_eval/ce results/xai_eval/scl results/xai_eval/triplet \
        --model_names CE SCL TL \
        --output_dir results/analysis
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats


def load_metric_csv(results_dir, filename):
    path = Path(results_dir) / filename
    if path.exists():
        return pd.read_csv(path)
    return None


def filter_correct_predictions(*dfs):
    """Keep only samples where ALL models predict correctly."""
    mask = None
    for df in dfs:
        m = df["true"] == df["pred"]
        mask = m if mask is None else mask & m
    return [df[mask].reset_index(drop=True) for df in dfs]


def compute_pairwise_stats(dfs, model_names, metric_cols):
    """Compute pairwise t-tests between all model pairs."""
    results = []
    for col in metric_cols:
        for i, name_a in enumerate(model_names):
            for j, name_b in enumerate(model_names):
                if i >= j:
                    continue
                vals_a = dfs[i][col].dropna()
                vals_b = dfs[j][col].dropna()
                n = min(len(vals_a), len(vals_b))
                if n < 2:
                    continue
                t_stat, p_val = stats.ttest_rel(vals_a[:n], vals_b[:n])
                diff = vals_a[:n].values - vals_b[:n].values
                cohens_d = diff.mean() / (diff.std() + 1e-12)
                results.append({
                    "metric": col,
                    "model_a": name_a, "model_b": name_b,
                    "mean_a": vals_a.mean(), "std_a": vals_a.std(),
                    "mean_b": vals_b.mean(), "std_b": vals_b.std(),
                    "t_stat": t_stat, "p_value": p_val,
                    "cohens_d": cohens_d,
                    "significant": p_val < 0.05,
                })
    return pd.DataFrame(results)


def plot_comparison(dfs, model_names, metric_cols, title, output_path):
    """Box plot comparison for each metric column."""
    n = len(metric_cols)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5), squeeze=False)
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]

    for idx, col in enumerate(metric_cols):
        ax = axes[0, idx]
        data = [df[col].dropna().values for df in dfs]
        bp = ax.boxplot(data, labels=model_names, patch_artist=True, widths=0.6)
        for patch, c in zip(bp["boxes"], colors[:len(model_names)]):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        for i, (d, c) in enumerate(zip(data, colors)):
            ax.axhline(np.mean(d), color=c, linestyle="--", alpha=0.7, linewidth=1)
        ax.set_title(col.replace("_", " "), fontsize=11)
        ax.set_ylabel("Score")
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def analyze_metric_file(filename, results_dirs, model_names, output_dir):
    """Load and analyze a single metric CSV across all models."""
    dfs_raw = []
    available_names = []
    for rd, mn in zip(results_dirs, model_names):
        df = load_metric_csv(rd, filename)
        if df is not None:
            dfs_raw.append(df)
            available_names.append(mn)

    if len(dfs_raw) < 2:
        return None

    dfs = filter_correct_predictions(*dfs_raw)
    exclude = {"idx", "true", "pred", "contrast_class"}
    metric_cols = [c for c in dfs[0].columns if c not in exclude]

    metric_name = filename.replace("_scores.csv", "")
    print(f"\n--- {metric_name.upper()} ---")
    print(f"  Samples (all correct): {len(dfs[0])}")
    for mn, df in zip(available_names, dfs):
        for col in metric_cols:
            vals = df[col].dropna()
            print(f"  {mn} {col}: {vals.mean():.4f} +/- {vals.std():.4f}")

    stats_df = compute_pairwise_stats(dfs, available_names, metric_cols)
    if len(stats_df) > 0:
        stats_df.to_csv(output_dir / f"stats_{metric_name}.csv", index=False)
        print(f"  Saved: stats_{metric_name}.csv")

    plot_path = output_dir / f"comparison_{metric_name}.png"
    plot_comparison(dfs, available_names, metric_cols,
                    f"{metric_name.upper()} Comparison", plot_path)
    print(f"  Saved: {plot_path.name}")

    return stats_df


def main():
    parser = argparse.ArgumentParser(description="Compare XAI results across models")
    parser.add_argument("--results_dirs", nargs="+", required=True,
                        help="Directories containing *_scores.csv files for each model")
    parser.add_argument("--model_names", nargs="+", required=True,
                        help="Model names corresponding to each results dir")
    parser.add_argument("--output_dir", type=str, default="results/analysis")
    args = parser.parse_args()

    assert len(args.results_dirs) == len(args.model_names), \
        "Must provide same number of results_dirs and model_names"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_files = [
        "pf_scores.csv", "continuity_scores.csv", "contrastivity_scores.csv",
        "complexity_scores.csv", "sparseness_scores.csv",
    ]

    all_stats = []
    for filename in metric_files:
        sdf = analyze_metric_file(filename, args.results_dirs, args.model_names, output_dir)
        if sdf is not None:
            all_stats.append(sdf)

    if all_stats:
        combined = pd.concat(all_stats, ignore_index=True)
        combined.to_csv(output_dir / "combined_statistics.csv", index=False)
        print(f"\nCombined statistics: {output_dir / 'combined_statistics.csv'}")

    print("\nDone!")


if __name__ == "__main__":
    main()
