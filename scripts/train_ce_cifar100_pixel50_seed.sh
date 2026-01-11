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

# Seed-parameterized CE + F-Fidelity Pixel50 for CIFAR-100
SEED=${SEED:-0}
OUTPUT_DIR=${OUTPUT_DIR:-"results/models/cifar100_r152/ce_pixel50_seed${SEED}"}
RUN_NAME=${RUN_NAME:-"ce_cifar100_pixel50_seed${SEED}"}

echo "=========================================="
echo "CLXAI: CE + Pixel50 - CIFAR-100 + ResNet-152 (Seed ${SEED})"
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

echo "Starting CE + Pixel50 training with seed ${SEED}..."
python src/training/train_ce.py \
    --config configs/ce_cifar100_pixel_aug.yaml \
    --epochs 200 \
    --batch_size 128 \
    --lr 0.1 \
    --data_dir /leonardo_scratch/fast/CNHPC_1905882/clxai/data \
    --output_dir "${OUTPUT_DIR}" \
    --run_name "${RUN_NAME}" \
    --seed ${SEED} \
    --augmentation_type pixel50

echo ""
echo "Training completed at: $(date)"
echo "=========================================="
