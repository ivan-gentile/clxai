#!/bin/bash
#SBATCH --account=CNHPC_1905882
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32000

# Seed-parameterized Triplet + F-Fidelity Pixel Removal training
# 10% scattered pixel removal (exact F-Fidelity implementation)
# IMPORTANT: Using semi-hard mining to prevent embedding collapse
SEED=${SEED:-0}
OUTPUT_DIR=${OUTPUT_DIR:-"results/models/triplet_pixel_seed${SEED}"}
RUN_NAME=${RUN_NAME:-"triplet_pixel_seed${SEED}"}

echo "=========================================="
echo "CLXAI: Triplet + F-Fidelity Pixel Augmentation (Seed ${SEED})"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Seed: ${SEED}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Mining: semi-hard (prevents embedding collapse)"
echo "Augmentation: F-Fidelity pixel removal (β=0.1, 10% pixels)"
echo "Start time: $(date)"
echo ""

# wandb offline mode (compute nodes have no internet)
export WANDB_MODE=offline
export WANDB_DIR=/leonardo_scratch/fast/CNHPC_1905882/clxai/wandb

# Load modules
module load profile/deeplrn
module load cineca-ai/4.3.0

# Activate environment
source /leonardo_scratch/fast/CNHPC_1905882/clxai/clxai_env/bin/activate

# Navigate to project
cd /leonardo_scratch/fast/CNHPC_1905882/clxai

# Create directories
mkdir -p logs/slurm "${OUTPUT_DIR}" data wandb

# Run training
echo "Starting Triplet + F-Fidelity Pixel training with seed ${SEED}..."
python src/training/train_scl.py \
    --config configs/scl_triplet_pixel_aug.yaml \
    --epochs 500 \
    --batch_size 256 \
    --lr 0.1 \
    --data_dir /leonardo_scratch/fast/CNHPC_1905882/clxai/data \
    --output_dir "${OUTPUT_DIR}" \
    --run_name "${RUN_NAME}" \
    --seed ${SEED} \
    --loss triplet \
    --mining semi-hard \
    --augmentation_type pixel

echo ""
echo "Training completed at: $(date)"
echo "=========================================="
