#!/bin/bash
# =============================================================================
# Submit XAI evaluation jobs for all model combinations
# =============================================================================
# Usage:
#   ./scripts/submit_all_xai_eval.sh           # All models, all seeds
#   ./scripts/submit_all_xai_eval.sh 0         # All models, seed 0 only
#   ./scripts/submit_all_xai_eval.sh 0 pf irof # All models, seed 0, specific metrics
# =============================================================================

SEED_FILTER=${1:-all}
shift 2>/dev/null || true
METRICS=${@:-all}

echo "=================================================="
echo "Submitting XAI Evaluation Jobs"
echo "=================================================="
echo "Seeds: $SEED_FILTER"
echo "Metrics: $METRICS"
echo ""

# Define models and augmentations
MODELS="ce scl triplet"
AUGMENTATIONS="pixel pixel50"

if [ "$SEED_FILTER" == "all" ]; then
    SEEDS="0 1 2 3 4"
else
    SEEDS=$SEED_FILTER
fi

cd /leonardo_scratch/fast/CNHPC_1905882/clxai

for MODEL in $MODELS; do
    for AUG in $AUGMENTATIONS; do
        for SEED in $SEEDS; do
            # Check if model exists
            MODEL_PATH="export_best_models/cifar10/${MODEL}_${AUG}/seed${SEED}/best_model.pt"
            if [ -f "$MODEL_PATH" ]; then
                echo "Submitting: ${MODEL}_${AUG} seed${SEED}"
                sbatch scripts/submit_xai_eval.sh $MODEL $AUG $SEED $METRICS
            else
                echo "Skipping: ${MODEL}_${AUG} seed${SEED} (model not found)"
            fi
        done
    done
done

echo ""
echo "Done! Use 'squeue -u \$USER' to check job status."
