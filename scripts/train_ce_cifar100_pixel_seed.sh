#!/bin/bash
#SBATCH --account=CNHPC_1905882
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64000

# Seed-parameterized CE + F-Fidelity Pixel Removal training for CIFAR-100
# 10% scattered pixel removal (exact F-Fidelity implementation)
SEED=${SEED:-0}
OUTPUT_DIR=${OUTPUT_DIR:-"results/models/cifar100_r152/ce_pixel_seed${SEED}"}
RUN_NAME=${RUN_NAME:-"ce_cifar100_pixel_seed${SEED}"}

echo "=========================================="
echo "CLXAI: CE + F-Fidelity Pixel - CIFAR-100 + ResNet-152 (Seed ${SEED})"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Seed: ${SEED}"
echo "Output dir: ${OUTPUT_DIR}"
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
echo "Starting CE + F-Fidelity Pixel training with seed ${SEED}..."
python src/training/train_ce.py \
    --config configs/ce_cifar100_pixel_aug.yaml \
    --epochs 200 \
    --batch_size 128 \
    --lr 0.1 \
    --data_dir /leonardo_scratch/fast/CNHPC_1905882/clxai/data \
    --output_dir "${OUTPUT_DIR}" \
    --run_name "${RUN_NAME}" \
    --seed ${SEED} \
    --augmentation_type pixel

echo ""
echo "Training completed at: $(date)"
echo "=========================================="
