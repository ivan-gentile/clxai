#!/bin/bash
#SBATCH --account=CNHPC_1905882
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64000
#SBATCH --job-name=spline_phase2
#SBATCH --output=logs/slurm/phase2_%j.out
#SBATCH --error=logs/slurm/phase2_%j.err

# ============================================================================
# Phase 2: Extended Training Experiment
# ============================================================================
# Tests grokking hypothesis by training CE models for extended periods
# with various regularization settings.
#
# Usage:
#   sbatch --export=VARIANT=CE-Extended,SEED=0 run_phase2_extended.sh
# ============================================================================

# Get parameters from environment (with defaults)
VARIANT=${VARIANT:-"CE-Extended"}
SEED=${SEED:-0}

echo "=========================================="
echo "Spline Theory - Phase 2: Extended Training"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Variant: ${VARIANT}"
echo "Seed: ${SEED}"
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
OUTPUT_DIR="spline_theory/results/phase2_extended/${VARIANT}_seed${SEED}"
mkdir -p "${OUTPUT_DIR}" logs/slurm

# Set configuration based on variant
case ${VARIANT} in
    "CE-Extended")
        WEIGHT_DECAY=0.0005
        NORM_TYPE="bn"
        RESUME_FROM="results/models/ce_seed${SEED}/best_model.pt"
        ;;
    "CE-NoWD")
        WEIGHT_DECAY=0.0
        NORM_TYPE="bn"
        RESUME_FROM="results/models/ce_seed${SEED}/best_model.pt"
        ;;
    "CE-NoBN")
        WEIGHT_DECAY=0.0005
        NORM_TYPE="id"
        RESUME_FROM=""
        ;;
    "CE-Minimal")
        WEIGHT_DECAY=0.0
        NORM_TYPE="id"
        RESUME_FROM=""
        ;;
    *)
        echo "Unknown variant: ${VARIANT}"
        exit 1
        ;;
esac

echo "Configuration:"
echo "  Weight decay: ${WEIGHT_DECAY}"
echo "  Norm type: ${NORM_TYPE}"
echo "  Resume from: ${RESUME_FROM:-'Fresh start'}"
echo ""

# Run extended training
python spline_theory/training/extended_trainer.py \
    --dataset cifar10 \
    --architecture resnet18 \
    --norm_type "${NORM_TYPE}" \
    --max_epochs 10000 \
    --batch_size 128 \
    --lr 0.1 \
    --weight_decay ${WEIGHT_DECAY} \
    --checkpoint_dir "${OUTPUT_DIR}" \
    --seed ${SEED} \
    ${RESUME_FROM:+--resume "${RESUME_FROM}"} \
    --no_wandb

echo ""
echo "Phase 2 training completed at: $(date)"
echo "Output saved to: ${OUTPUT_DIR}"
echo "=========================================="
