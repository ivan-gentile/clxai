#!/bin/bash
#SBATCH --account=CNHPC_1905882
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64000
#SBATCH --job-name=trip_sh_c100
#SBATCH --output=logs/slurm/triplet_semihard_cifar100_r152_%j.out
#SBATCH --error=logs/slurm/triplet_semihard_cifar100_r152_%j.err

echo "=========================================="
echo "CLXAI: Triplet Loss - CIFAR-100 + ResNet-152 (SEMI-HARD)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
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
mkdir -p logs/slurm results/models/cifar100_r152/triplet_semihard data wandb

# Run training with SEMI-HARD mining
echo "Starting CIFAR-100 ResNet-152 Triplet training (SEMI-HARD)..."
python src/training/train_scl.py \
    --dataset cifar100 \
    --architecture resnet152 \
    --epochs 500 \
    --batch_size 256 \
    --lr 0.1 \
    --loss triplet \
    --margin 0.3 \
    --mining semi-hard \
    --data_dir /leonardo_scratch/fast/CNHPC_1905882/clxai/data \
    --output_dir results/models/cifar100_r152/triplet_semihard \
    --run_name triplet_semihard_cifar100_r152 \
    --seed 42

echo ""
echo "Training completed at: $(date)"
echo "=========================================="
