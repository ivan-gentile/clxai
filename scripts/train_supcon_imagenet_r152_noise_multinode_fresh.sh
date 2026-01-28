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
#SBATCH --job-name=supcon_noise_fresh
#SBATCH --output=logs/slurm/supcon_noise_fresh_%j.out
#SBATCH --error=logs/slurm/supcon_noise_fresh_%j.err

echo "=========================================="
echo "SupCon ImageNet R152 NOISE - FRESH START"
echo "temperature=0.07, epochs=420"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "Start: $(date)"

export WANDB_MODE=offline
export WANDB_DIR=/leonardo_scratch/fast/CNHPC_1905882/clxai/logs/wandb

module load profile/deeplrn
module load cineca-ai/4.3.0

source /leonardo_scratch/fast/CNHPC_1905882/clxai/clxai_env/bin/activate
cd /leonardo_scratch/fast/CNHPC_1905882/clxai

mkdir -p logs/slurm logs/wandb results/models/imagenet_r152/supcon_noise

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=^lo,docker
export NCCL_IB_HCA=mlx5

export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n1)
export MASTER_PORT=29502

echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "World size: $SLURM_NTASKS"
echo "Fresh start with temperature=0.07"

srun python src/training/train_scl_imagenet_ddp.py \
    --config configs/supcon_imagenet_r152_noise.yaml \
    --run_name "supcon_noise_t007_${SLURM_JOB_ID}"

echo "Completed: $(date)"
