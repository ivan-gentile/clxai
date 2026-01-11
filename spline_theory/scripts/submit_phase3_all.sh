#!/bin/bash
# ============================================================================
# Submit all Phase 3 Ablation Jobs
# ============================================================================
# Submits the full factorial ablation:
# - 2 loss functions (supcon, triplet)
# - 3 weight decay levels (0, 1e-5, 1e-4)
# - 3 normalization types (bn, gn, id)
# - 3 seeds (0, 1, 2)
# Total: 2 x 3 x 3 x 3 = 54 jobs
# ============================================================================

cd /leonardo_scratch/fast/CNHPC_1905882/clxai

# Create log directory
mkdir -p logs/slurm

echo "Submitting Phase 3 Ablation Jobs..."
echo "===================================="

JOB_COUNT=0

for LOSS in supcon triplet; do
    for NORM in bn gn id; do
        for WD in 0.0 0.00001 0.0001; do
            for SEED in 0 1 2; do
                echo "Submitting: loss=${LOSS}, norm=${NORM}, wd=${WD}, seed=${SEED}"
                
                sbatch --export=LOSS=${LOSS},NORM=${NORM},WD=${WD},SEED=${SEED} \
                    spline_theory/scripts/run_phase3_ablation.sh
                
                JOB_COUNT=$((JOB_COUNT + 1))
                
                # Small delay to avoid overwhelming the scheduler
                sleep 0.5
            done
        done
    done
done

echo ""
echo "Submitted ${JOB_COUNT} jobs"
echo "Use 'squeue -u \$USER' to monitor progress"
