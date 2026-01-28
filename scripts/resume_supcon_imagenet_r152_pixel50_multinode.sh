#!/bin/bash
#SBATCH --account=CNHPC_1469675
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=96:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=494000
#SBATCH --job-name=supcon_pixel50_resume
#SBATCH --output=logs/slurm/supcon_pixel50_resume_%j.out
#SBATCH --error=logs/slurm/supcon_pixel50_resume_%j.err

echo "=========================================="
echo "SupCon ImageNet R152 PIXEL50 - RESUME from epoch 420"
echo "Target: 840 epochs (4-day run, eval/save every 30)"
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

mkdir -p logs/slurm logs/wandb

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=^lo,docker
export NCCL_IB_HCA=mlx5

export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n1)
export MASTER_PORT=29501

echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "World size: $SLURM_NTASKS"
echo "Resuming from: results/models/imagenet_r152/supcon_pixel50/checkpoint_epoch_420.pt"
echo "Target: 840 epochs (420 more), 4-day walltime"

srun python src/training/train_scl_imagenet_ddp.py \
    --config configs/supcon_imagenet_r152_pixel50.yaml \
    --resume results/models/imagenet_r152/supcon_pixel50/checkpoint_epoch_420.pt \
    --run_name "supcon_pixel50_ep600_${SLURM_JOB_ID}"

echo "Completed: $(date)"
