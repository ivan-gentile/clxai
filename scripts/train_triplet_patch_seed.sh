#!/bin/bash
#SBATCH --account=CNHPC_1905882
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64000

# Triplet Loss + Patch Augmentation Training
# Get seed from environment variable (default to 0)
SEED=${SEED:-0}
OUTPUT_DIR=${OUTPUT_DIR:-"results/models/triplet_patch_seed${SEED}"}
RUN_NAME=${RUN_NAME:-"triplet_patch_seed${SEED}"}

echo "=========================================="
echo "CLXAI: Triplet + Patch Augmentation (Seed ${SEED})"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Seed: ${SEED}"
echo "Output dir: ${OUTPUT_DIR}"
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

# Run training with seed and patch augmentation
echo "Starting Triplet + Patch training with seed ${SEED}..."
python src/training/train_scl.py \
    --config configs/scl_triplet_patch_aug.yaml \
    --epochs 500 \
    --batch_size 256 \
    --lr 0.1 \
    --loss triplet \
    --margin 0.3 \
    --mining semi-hard \
    --data_dir /leonardo_scratch/fast/CNHPC_1905882/clxai/data \
    --output_dir "${OUTPUT_DIR}" \
    --run_name "${RUN_NAME}" \
    --seed ${SEED} \
    --augmentation_type patch

echo ""
echo "Training completed at: $(date)"
echo "=========================================="
