#!/bin/bash
# ===========================================
# Submit all augmented training jobs
# 30 jobs total: 6 configurations × 5 seeds
# ===========================================

BASE_DIR="/leonardo_scratch/fast/CNHPC_1905882/clxai"
cd ${BASE_DIR}

# Seeds to train
SEEDS=(0 1 2 3 4)

echo "=========================================="
echo "CLXAI: Submit All Augmented Training Jobs"
echo "=========================================="
echo "Seeds: ${SEEDS[@]}"
echo "Total jobs: 30 (6 configs × 5 seeds)"
echo ""

# Create log directory
mkdir -p logs/slurm

# ============================================
# PATCH AUGMENTATION JOBS (15 total)
# ============================================

echo "--- PATCH AUGMENTATION ---"
echo ""

# Submit CE + Patch jobs (5 seeds × ~4h each)
echo "Submitting 5 CE + Patch jobs..."
for seed in "${SEEDS[@]}"; do
    sbatch --job-name="clxai_ce_patch_s${seed}" \
           --output="logs/slurm/ce_patch_seed${seed}_%j.out" \
           --error="logs/slurm/ce_patch_seed${seed}_%j.err" \
           --export=ALL,SEED=${seed},OUTPUT_DIR="results/models/ce_patch_seed${seed}",RUN_NAME="ce_patch_seed${seed}" \
           scripts/train_ce_patch_seed.sh
    echo "  ✓ CE + Patch seed ${seed}"
    sleep 0.5
done
echo ""

# Submit SCL + Patch jobs (5 seeds × ~6h each)
echo "Submitting 5 SupCon + Patch jobs..."
for seed in "${SEEDS[@]}"; do
    sbatch --job-name="clxai_scl_patch_s${seed}" \
           --output="logs/slurm/scl_patch_seed${seed}_%j.out" \
           --error="logs/slurm/scl_patch_seed${seed}_%j.err" \
           --export=ALL,SEED=${seed},OUTPUT_DIR="results/models/scl_patch_seed${seed}",RUN_NAME="scl_patch_seed${seed}" \
           scripts/train_scl_patch_seed.sh
    echo "  ✓ SupCon + Patch seed ${seed}"
    sleep 0.5
done
echo ""

# Submit Triplet + Patch jobs (5 seeds × ~6h each)
echo "Submitting 5 Triplet + Patch jobs..."
for seed in "${SEEDS[@]}"; do
    sbatch --job-name="clxai_tri_patch_s${seed}" \
           --output="logs/slurm/triplet_patch_seed${seed}_%j.out" \
           --error="logs/slurm/triplet_patch_seed${seed}_%j.err" \
           --export=ALL,SEED=${seed},OUTPUT_DIR="results/models/triplet_patch_seed${seed}",RUN_NAME="triplet_patch_seed${seed}" \
           scripts/train_triplet_patch_seed.sh
    echo "  ✓ Triplet + Patch seed ${seed}"
    sleep 0.5
done
echo ""

# ============================================
# NOISE AUGMENTATION JOBS (15 total)
# ============================================

echo "--- NOISE AUGMENTATION ---"
echo ""

# Submit CE + Noise jobs (5 seeds × ~4h each)
echo "Submitting 5 CE + Noise jobs..."
for seed in "${SEEDS[@]}"; do
    sbatch --job-name="clxai_ce_noise_s${seed}" \
           --output="logs/slurm/ce_noise_seed${seed}_%j.out" \
           --error="logs/slurm/ce_noise_seed${seed}_%j.err" \
           --export=ALL,SEED=${seed},OUTPUT_DIR="results/models/ce_noise_seed${seed}",RUN_NAME="ce_noise_seed${seed}" \
           scripts/train_ce_noise_seed.sh
    echo "  ✓ CE + Noise seed ${seed}"
    sleep 0.5
done
echo ""

# Submit SCL + Noise jobs (5 seeds × ~6h each)
echo "Submitting 5 SupCon + Noise jobs..."
for seed in "${SEEDS[@]}"; do
    sbatch --job-name="clxai_scl_noise_s${seed}" \
           --output="logs/slurm/scl_noise_seed${seed}_%j.out" \
           --error="logs/slurm/scl_noise_seed${seed}_%j.err" \
           --export=ALL,SEED=${seed},OUTPUT_DIR="results/models/scl_noise_seed${seed}",RUN_NAME="scl_noise_seed${seed}" \
           scripts/train_scl_noise_seed.sh
    echo "  ✓ SupCon + Noise seed ${seed}"
    sleep 0.5
done
echo ""

# Submit Triplet + Noise jobs (5 seeds × ~6h each)
echo "Submitting 5 Triplet + Noise jobs..."
for seed in "${SEEDS[@]}"; do
    sbatch --job-name="clxai_tri_noise_s${seed}" \
           --output="logs/slurm/triplet_noise_seed${seed}_%j.out" \
           --error="logs/slurm/triplet_noise_seed${seed}_%j.err" \
           --export=ALL,SEED=${seed},OUTPUT_DIR="results/models/triplet_noise_seed${seed}",RUN_NAME="triplet_noise_seed${seed}" \
           scripts/train_triplet_noise_seed.sh
    echo "  ✓ Triplet + Noise seed ${seed}"
    sleep 0.5
done

echo ""
echo "=========================================="
echo "Submitted 30 jobs total:"
echo ""
echo "PATCH AUGMENTATION (15 jobs):"
echo "  - 5 CE + Patch (seeds 0-4)"
echo "  - 5 SupCon + Patch (seeds 0-4)"
echo "  - 5 Triplet + Patch (seeds 0-4)"
echo ""
echo "NOISE AUGMENTATION (15 jobs):"
echo "  - 5 CE + Noise (seeds 0-4)"
echo "  - 5 SupCon + Noise (seeds 0-4)"
echo "  - 5 Triplet + Noise (seeds 0-4)"
echo ""
echo "Check status: squeue -u \$USER"
echo "=========================================="
