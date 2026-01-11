# Spline Theory Research Extension

This module extends the CLXAI project to test a unified hypothesis connecting **contrastive learning**, **spline theory**, and the **grokking phenomenon**.

## Core Hypothesis

> **Contrastive Learning's training objective naturally induces partition geometry similar to the post-grokking state, achieving robust and interpretable behavior WITHOUT requiring extended training or reduced regularization.**

### Background

1. **CLXAI Original Hypothesis**: Contrastive Learning produces embedding spaces where perturbed inputs move predictably away from class clusters, enabling reliable faithfulness evaluation.

2. **Spline Theory Insight**: Neural networks partition input space into convex polytopes. Good generalization emerges when partition regions are large around data points and concentrated at decision boundaries.

3. **Grokking Phenomenon**: With minimal regularization and extended training, models spontaneously reorganize their partition geometry, causing delayed emergence of adversarial robustness.

### Falsifiable Predictions

| ID | Prediction | If True | If False |
|----|------------|---------|----------|
| P1 | CL models show adversarial robustness earlier in training | CL induces good geometry faster | CL offers no geometric advantage |
| P2 | Extended CE training eventually matches CL faithfulness | Grokking provides alternative path | CL has unique geometric properties |
| P3 | Removing regularization from CE accelerates convergence | Regularization is the main barrier | Architecture/loss is the main factor |
| P4 | CL has larger partition regions around data points | CL's cluster objective creates this | CL geometry is fundamentally different |
| P5 | BatchNorm removal helps CE more than CL | CL is already "BatchNorm-resistant" | Both affected equally |

## Directory Structure

```
spline_theory/
├── __init__.py
├── README.md
│
├── configs/                     # Experiment configurations
│   ├── phase1_diagnostic.yaml
│   ├── phase2_extended_training.yaml
│   ├── phase3_cl_ablation.yaml
│   ├── phase4_augmentation.yaml
│   └── phase5_geometric.yaml
│
├── models/                      # Model variants
│   ├── __init__.py
│   └── resnet_variants.py       # ResNet with configurable normalization
│
├── training/                    # Training utilities
│   ├── __init__.py
│   ├── extended_trainer.py      # Extended training with dense checkpoints
│   └── augmentation.py          # Additional augmentation strategies
│
├── evaluation/                  # Evaluation modules
│   ├── __init__.py
│   ├── adversarial.py           # PGD, FGSM attacks
│   ├── geometric.py             # Local complexity metrics
│   └── faithfulness_extended.py # Extended faithfulness metrics
│
├── analysis/                    # Analysis and visualization
│   ├── __init__.py
│   ├── grokking_detection.py    # Detect phase transitions
│   └── comparison.py            # CE vs CL comparison
│
└── scripts/                     # SLURM job scripts
    ├── run_phase1.sh
    ├── run_phase2_extended.sh
    ├── run_phase3_ablation.sh
    ├── run_phase4_augment.sh
    ├── run_phase5_geometric.sh
    ├── submit_phase2_all.sh
    ├── submit_phase3_all.sh
    └── submit_phase4_all.sh
```

## Experimental Phases

### Phase 1: Diagnostic Analysis (2-4 GPU hours)

Analyze existing trained models (CE, SupCon, Triplet) to establish baselines.

**Run:**
```bash
sbatch spline_theory/scripts/run_phase1.sh
```

**Outputs:**
- Clean and adversarial accuracy for all models
- Faithfulness metrics comparison
- Initial geometric analysis

### Phase 2: Extended Training (~480 GPU hours)

Test grokking hypothesis by training CE models for 10,000 epochs with various configurations.

**Variants:**
- `CE-Extended`: Continue existing CE models
- `CE-NoWD`: Remove weight decay (enable grokking)
- `CE-NoBN`: Remove BatchNorm (fresh start)
- `CE-Minimal`: No WD + No BN (maximum grokking potential)

**Run:**
```bash
# Submit all Phase 2 jobs
bash spline_theory/scripts/submit_phase2_all.sh

# Or submit individual jobs
sbatch --export=VARIANT=CE-NoWD,SEED=0 spline_theory/scripts/run_phase2_extended.sh
```

### Phase 3: CL Ablation (~324 GPU hours)

Full factorial ablation on contrastive learning configurations.

**Factors:**
- Loss: `supcon`, `triplet`
- Regularization: `0.0`, `1e-5`, `1e-4`
- Normalization: `bn`, `gn`, `id`

**Run:**
```bash
# Submit all Phase 3 jobs (54 total)
bash spline_theory/scripts/submit_phase3_all.sh
```

### Phase 4: Augmentation Impact (~90 GPU hours)

Test whether CL makes data augmentation redundant.

**Strategies:**
- `none`: No augmentation
- `standard`: Random crop + flip
- `patch`: Standard + patch occlusion
- `noise`: Standard + Gaussian noise
- `strong`: All augmentations combined

**Run:**
```bash
bash spline_theory/scripts/submit_phase4_all.sh
```

### Phase 5: Geometric Validation (~20 GPU hours)

Directly validate spline theory predictions.

**Run:**
```bash
sbatch spline_theory/scripts/run_phase5_geometric.sh
```

## Key Components

### ResNet Variants

Models with configurable normalization layers:

```python
from spline_theory.models import get_resnet_variant

# Standard BatchNorm
model = get_resnet_variant("resnet18", num_classes=10, norm_type="bn")

# No normalization (for grokking experiments)
model = get_resnet_variant("resnet18", num_classes=10, norm_type="id")

# GroupNorm (batch-size independent)
model = get_resnet_variant("resnet18", num_classes=10, norm_type="gn")
```

### Adversarial Evaluation

```python
from spline_theory.evaluation.adversarial import AdversarialEvaluator

evaluator = AdversarialEvaluator(model, device="cuda", eps=8/255)
results = evaluator.evaluate_all(test_loader, n_samples=1000)
# Returns: clean_accuracy, fgsm_accuracy, pgd_accuracy
```

### Geometric Analysis

```python
from spline_theory.evaluation.geometric import LocalComplexityAnalyzer

analyzer = LocalComplexityAnalyzer(model, device="cuda")
density = analyzer.estimate_partition_density(
    data_loader, epsilon=0.1, n_neighbors=50, n_samples=100
)
# Returns: mean_diversity (lower = larger partition regions = better)
```

### Grokking Detection

```python
from spline_theory.analysis.grokking_detection import detect_grokking_transition

history = {"epoch": [...], "train_acc": [...], "test_acc": [...]}
results = detect_grokking_transition(history)
# Returns: grokking_detected, transition_epoch, gap_at_transition
```

## Compute Budget

| Phase | Configurations | Seeds | Max Epochs | GPU Hours |
|-------|---------------|-------|------------|-----------|
| 1 | 15 (existing) | - | - | 2-4 |
| 2 | 4 | 3 | 10000 | 480 |
| 3 | 18 | 3 | 3000 | 324 |
| 4 | 5 | 3 | 3000 | 90 |
| 5 | - | - | - | 20 |
| **Total** | | | | **~920** |

## Results Directory

Results are saved to `spline_theory/results/`:
- `phase1_diagnostic/`: Diagnostic analysis outputs
- `phase2_extended/{variant}_seed{seed}/`: Extended training checkpoints
- `phase3_ablation/{loss}_{norm}_wd{wd}_seed{seed}/`: Ablation results
- `phase4_augmentation/{aug}_seed{seed}/`: Augmentation study results
- `phase5_geometric/`: Geometric validation reports

## Dependencies

This module uses existing CLXAI infrastructure:
- `src/models/resnet.py`: Base ResNet implementations
- `src/training/losses.py`: SupCon and Triplet losses
- `src/utils/data.py`: CIFAR data loaders
- `src/xai/saliency.py`: Saliency extraction
- `src/analysis/faithfulness.py`: ECE-Faithfulness metrics

## Authors

- CLXAI Research Team (IFAB + RUG collaboration)

## References

1. Khosla et al. "Supervised Contrastive Learning" (NeurIPS 2020)
2. Power et al. "Grokking: Generalization Beyond Overfitting" (2022)
3. Balestriero & LeCun "Spline Theory of Deep Networks" (ICML 2018)
