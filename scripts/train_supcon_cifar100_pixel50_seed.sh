#!/bin/bash
#SBATCH --account=CNHPC_1905882
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=14:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64000

# Seed-parameterized SupCon + F-Fidelity Pixel50 for CIFAR-100
SEED=${SEED:-0}
OUTPUT_DIR=${OUTPUT_DIR:-"results/models/cifar100_r152/supcon_pixel50_seed${SEED}"}
RUN_NAME=${RUN_NAME:-"supcon_cifar100_pixel50_seed${SEED}"}

echo "=========================================="
echo "CLXAI: SupCon + Pixel50 - CIFAR-100 + ResNet-152 (Seed ${SEED})"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Seed: ${SEED}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Augmentation: F-Fidelity pixel removal (β=0.1, 50% probability)"
echo "Start time: $(date)"
echo ""

export WANDB_MODE=offline
export WANDB_DIR=/leonardo_scratch/fast/CNHPC_1905882/clxai/wandb

module load profile/deeplrn
module load cineca-ai/4.3.0

source /leonardo_scratch/fast/CNHPC_1905882/clxai/clxai_env/bin/activate
cd /leonardo_scratch/fast/CNHPC_1905882/clxai

mkdir -p logs/slurm "${OUTPUT_DIR}" data wandb

echo "Starting SupCon + Pixel50 training with seed ${SEED}..."
python src/training/train_scl.py \
    --config configs/supcon_cifar100_pixel_aug.yaml \
    --epochs 500 \
    --batch_size 256 \
    --lr 0.5 \
    --data_dir /leonardo_scratch/fast/CNHPC_1905882/clxai/data \
    --output_dir "${OUTPUT_DIR}" \
    --run_name "${RUN_NAME}" \
    --seed ${SEED} \
    --loss supcon \
    --augmentation_type pixel50

echo ""
echo "Training completed at: $(date)"
echo "=========================================="
