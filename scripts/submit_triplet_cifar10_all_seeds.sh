#!/bin/bash
# Submit triplet loss training for CIFAR-10 with all 5 seeds
# Uses the fixed training code (unnormalized embeddings)

echo "Submitting Triplet Loss CIFAR-10 training for seeds 0-4..."

for SEED in 0 1 2 3 4; do
    export SEED=$SEED
    export OUTPUT_DIR="results/models/triplet_fixed_seed${SEED}"
    export RUN_NAME="triplet_fixed_seed${SEED}"
    
    sbatch --job-name="triplet_s${SEED}" \
           --output="logs/slurm/triplet_fixed_seed${SEED}_%j.out" \
           --error="logs/slurm/triplet_fixed_seed${SEED}_%j.err" \
           /leonardo_scratch/fast/CNHPC_1905882/clxai/scripts/train_triplet_seed.sh
    
    echo "Submitted seed $SEED"
    sleep 1
done

echo "All 5 seeds submitted!"
echo "Monitor with: squeue -u \$USER"
