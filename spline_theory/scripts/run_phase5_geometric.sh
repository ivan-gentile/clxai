#!/bin/bash
#SBATCH --account=CNHPC_1905882
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64000
#SBATCH --job-name=spline_phase5
#SBATCH --output=logs/slurm/phase5_%j.out
#SBATCH --error=logs/slurm/phase5_%j.err

# ============================================================================
# Phase 5: Geometric Validation
# ============================================================================
# Directly validates spline theory predictions through comprehensive
# geometric analysis of trained models.
# ============================================================================

echo "=========================================="
echo "Spline Theory - Phase 5: Geometric Validation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""

# wandb offline mode
export WANDB_MODE=offline
export WANDB_DIR=/leonardo_scratch/fast/CNHPC_1905882/clxai/wandb

# Load modules
module load profile/deeplrn
module load cineca-ai/4.3.0

# Activate environment
source /leonardo_scratch/fast/CNHPC_1905882/clxai/clxai_env/bin/activate

# Navigate to project
cd /leonardo_scratch/fast/CNHPC_1905882/clxai

# Create output directory
OUTPUT_DIR="spline_theory/results/phase5_geometric"
mkdir -p "${OUTPUT_DIR}" logs/slurm

echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Run geometric validation
python -c "
import sys
sys.path.insert(0, '.')

import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

from src.utils.data import get_data_loaders
from src.models.resnet import ResNet18, ResNet18Encoder
from spline_theory.models.resnet_variants import get_resnet_variant
from spline_theory.evaluation.geometric import LocalComplexityAnalyzer
from spline_theory.evaluation.adversarial import AdversarialEvaluator
from spline_theory.analysis.comparison import compare_ce_vs_cl, generate_comparison_report

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

OUTPUT_DIR = Path('${OUTPUT_DIR}')

# Get test data
_, test_loader = get_data_loaders(
    dataset='cifar10',
    data_dir='/leonardo_scratch/fast/CNHPC_1905882/clxai/data',
    batch_size=128,
    num_workers=4,
    augment=False
)

results = {}

# ============================================================================
# Part 1: Compare existing CE vs CL models
# ============================================================================
print('\n' + '=' * 60)
print('Part 1: CE vs CL Baseline Comparison')
print('=' * 60)

# Load CE model (seed 0)
ce_path = Path('results/models/ce_seed0/best_model.pt')
if ce_path.exists():
    model_ce = ResNet18(num_classes=10).to(device)
    ckpt = torch.load(ce_path, map_location=device)
    model_ce.load_state_dict(ckpt['model_state_dict'])
    model_ce.eval()
    print(f'Loaded CE model from {ce_path}')
else:
    print(f'CE model not found: {ce_path}')
    model_ce = None

# Load CL model (seed 0)
cl_path = Path('results/models/scl_seed0/best_model.pt')
if cl_path.exists():
    model_cl = ResNet18Encoder(embedding_dim=128).to(device)
    ckpt = torch.load(cl_path, map_location=device)
    model_cl.load_state_dict(ckpt['model_state_dict'])
    model_cl.eval()
    print(f'Loaded CL model from {cl_path}')
else:
    print(f'CL model not found: {cl_path}')
    model_cl = None

# Run comparison if both models available
if model_ce is not None and model_cl is not None:
    print('\nRunning CE vs CL comparison...')
    
    comparison = compare_ce_vs_cl(
        model_ce, model_cl, test_loader,
        device=device,
        run_geometric=True,
        run_adversarial=True,
        run_faithfulness=True,
        n_samples=200
    )
    
    results['ce_vs_cl_baseline'] = comparison
    
    # Generate report
    report = generate_comparison_report(
        comparison,
        output_path=str(OUTPUT_DIR / 'ce_vs_cl_comparison.txt')
    )
    print(report)

# ============================================================================
# Part 2: Geometric Analysis Across Training
# ============================================================================
print('\n' + '=' * 60)
print('Part 2: Geometric Analysis Across Training')
print('=' * 60)

# Analyze Phase 2 extended training checkpoints
phase2_results = {}
for variant in ['CE-Extended', 'CE-NoWD', 'CE-NoBN', 'CE-Minimal']:
    variant_dir = Path(f'spline_theory/results/phase2_extended/{variant}_seed0')
    if not variant_dir.exists():
        print(f'  {variant}: directory not found')
        continue
    
    print(f'\n  Analyzing {variant}...')
    variant_results = {}
    
    for epoch in [100, 500, 1000, 5000, 10000]:
        ckpt_path = variant_dir / f'checkpoint_epoch_{epoch}.pt'
        if not ckpt_path.exists():
            continue
        
        # Determine model type
        if 'NoBN' in variant or 'Minimal' in variant:
            model = get_resnet_variant('resnet18', 10, norm_type='id').to(device)
        else:
            model = get_resnet_variant('resnet18', 10, norm_type='bn').to(device)
        
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        
        # Run geometric analysis
        analyzer = LocalComplexityAnalyzer(model, device=device)
        geo = analyzer.estimate_partition_density(
            test_loader, epsilon=0.1, n_neighbors=50, n_samples=50
        )
        
        # Run adversarial eval
        evaluator = AdversarialEvaluator(model, device=device)
        adv = evaluator.evaluate_all(test_loader, n_samples=200)
        
        variant_results[epoch] = {
            'geometric': geo,
            'adversarial': adv
        }
        
        print(f'    Epoch {epoch}: diversity={geo[\"mean_diversity\"]:.4f}, pgd_acc={adv[\"pgd_accuracy\"]*100:.2f}%')
    
    phase2_results[variant] = variant_results

results['phase2_evolution'] = phase2_results

# ============================================================================
# Part 3: Validate Spline Theory Predictions
# ============================================================================
print('\n' + '=' * 60)
print('Part 3: Hypothesis Validation Summary')
print('=' * 60)

predictions_validated = {}

# P1: CL shows adversarial robustness earlier
if 'ce_vs_cl_baseline' in results:
    p1 = results['ce_vs_cl_baseline'].get('adversarial', {}).get('cl_more_robust', False)
    predictions_validated['P1_early_robustness'] = p1
    print(f'P1 (CL early robustness): {\"SUPPORTED\" if p1 else \"NOT SUPPORTED\"}')

# P4: CL has larger regions (lower complexity)
if 'ce_vs_cl_baseline' in results:
    p4 = results['ce_vs_cl_baseline'].get('geometric', {}).get('cl_has_lower_complexity', False)
    predictions_validated['P4_lower_complexity'] = p4
    print(f'P4 (CL lower complexity): {\"SUPPORTED\" if p4 else \"NOT SUPPORTED\"}')

results['predictions_validated'] = predictions_validated

# Save all results
with open(OUTPUT_DIR / 'phase5_results.json', 'w') as f:
    # Convert non-serializable types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    json.dump(convert(results), f, indent=2)

print(f'\nResults saved to {OUTPUT_DIR}')
print('\nPhase 5 geometric validation completed!')
"

echo ""
echo "Phase 5 completed at: $(date)"
echo "=========================================="
