#!/bin/bash
# ============================================================================
# Submit all Phase 4 Augmentation Jobs
# ============================================================================
# Submits all augmentation strategies with all seeds:
# - 5 augmentation types (none, standard, patch, noise, strong)
# - 3 seeds (0, 1, 2)
# Total: 5 x 3 = 15 jobs
# ============================================================================

cd /leonardo_scratch/fast/CNHPC_1905882/clxai

# Create log directory
mkdir -p logs/slurm

echo "Submitting Phase 4 Augmentation Jobs..."
echo "========================================"

JOB_COUNT=0

for AUGMENT in none standard patch noise strong; do
    for SEED in 0 1 2; do
        echo "Submitting: augmentation=${AUGMENT}, seed=${SEED}"
        
        sbatch --export=AUGMENT=${AUGMENT},SEED=${SEED} \
            spline_theory/scripts/run_phase4_augment.sh
        
        JOB_COUNT=$((JOB_COUNT + 1))
        
        # Small delay
        sleep 0.5
    done
done

echo ""
echo "Submitted ${JOB_COUNT} jobs"
echo "Use 'squeue -u \$USER' to monitor progress"
