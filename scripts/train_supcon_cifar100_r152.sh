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
#SBATCH --job-name=supcon_c100_r152
#SBATCH --output=logs/slurm/supcon_cifar100_r152_%j.out
#SBATCH --error=logs/slurm/supcon_cifar100_r152_%j.err

echo "=========================================="
echo "CLXAI: SupCon - CIFAR-100 + ResNet-152"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
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

# Create directories
mkdir -p logs/slurm results/models/cifar100_r152/supcon wandb

# Run training
python src/training/train_scl.py \
    --config configs/supcon_cifar100_r152.yaml

echo ""
echo "Training completed at: $(date)"
echo "=========================================="

