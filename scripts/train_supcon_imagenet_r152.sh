#!/bin/bash
#SBATCH --account=CNHPC_1469675
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=494000
#SBATCH --job-name=supcon_r152_base
#SBATCH --output=logs/slurm/supcon_r152_base_%j.out
#SBATCH --error=logs/slurm/supcon_r152_base_%j.err

echo "=========================================="
echo "SupCon ImageNet R152 - BASE (no custom aug)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"

# WandB offline mode (no internet on compute nodes)
export WANDB_MODE=offline
export WANDB_DIR=/leonardo_scratch/fast/CNHPC_1905882/clxai/logs/wandb

# Load modules
module load profile/deeplrn
module load cineca-ai/4.3.0

# Activate environment
source /leonardo_scratch/fast/CNHPC_1905882/clxai/clxai_env/bin/activate
cd /leonardo_scratch/fast/CNHPC_1905882/clxai

# Create directories
mkdir -p logs/slurm logs/wandb results/models/imagenet_r152/supcon

# Multi-GPU setup
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Run training
python src/training/train_scl_imagenet.py \
    --config configs/supcon_imagenet_r152.yaml \
    --run_name "supcon_r152_base_${SLURM_JOB_ID}"

echo "Completed: $(date)"
