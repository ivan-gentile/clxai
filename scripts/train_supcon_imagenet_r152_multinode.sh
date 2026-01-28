#!/bin/bash
#SBATCH --account=CNHPC_1469675
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=48:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=494000
#SBATCH --job-name=supcon_r152_multi
#SBATCH --output=logs/slurm/supcon_r152_multi_%j.out
#SBATCH --error=logs/slurm/supcon_r152_multi_%j.err

echo "=========================================="
echo "SupCon ImageNet R152 - Multi-Node DDP (4 nodes, 16 GPUs)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Total tasks: $SLURM_NTASKS"
echo "Node list: $SLURM_NODELIST"
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

# NCCL settings for multi-node on Leonardo
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
# Auto-detect network interface (exclude loopback and docker)
export NCCL_SOCKET_IFNAME=^lo,docker
export NCCL_IB_HCA=mlx5

# Master address (first node in allocation)
export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n1)
export MASTER_PORT=29500

echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "World size: $SLURM_NTASKS"
echo "GPUs per node: 4"
echo "Batch per GPU: 128, Effective batch: $((128 * SLURM_NTASKS))"

# Run with srun - each task uses SLURM_LOCALID to select GPU
srun python src/training/train_scl_imagenet_ddp.py \
    --config configs/supcon_imagenet_r152.yaml \
    --run_name "supcon_r152_multi_${SLURM_JOB_ID}"

echo "Completed: $(date)"
