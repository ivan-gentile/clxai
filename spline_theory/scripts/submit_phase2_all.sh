#!/bin/bash
# ============================================================================
# Submit all Phase 2 Extended Training Jobs
# ============================================================================
# Submits all variants with all seeds:
# - 4 variants (CE-Extended, CE-NoWD, CE-NoBN, CE-Minimal)
# - 3 seeds (0, 1, 2)
# Total: 4 x 3 = 12 jobs
# ============================================================================

cd /leonardo_scratch/fast/CNHPC_1905882/clxai

# Create log directory
mkdir -p logs/slurm

echo "Submitting Phase 2 Extended Training Jobs..."
echo "============================================="

JOB_COUNT=0

for VARIANT in CE-Extended CE-NoWD CE-NoBN CE-Minimal; do
    for SEED in 0 1 2; do
        echo "Submitting: variant=${VARIANT}, seed=${SEED}"
        
        sbatch --export=VARIANT=${VARIANT},SEED=${SEED} \
            spline_theory/scripts/run_phase2_extended.sh
        
        JOB_COUNT=$((JOB_COUNT + 1))
        
        # Small delay
        sleep 0.5
    done
done

echo ""
echo "Submitted ${JOB_COUNT} jobs"
echo "Use 'squeue -u \$USER' to monitor progress"
