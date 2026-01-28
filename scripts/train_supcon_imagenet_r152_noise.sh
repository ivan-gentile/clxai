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
#SBATCH --job-name=supcon_r152_noise
#SBATCH --output=logs/slurm/supcon_r152_noise_%j.out
#SBATCH --error=logs/slurm/supcon_r152_noise_%j.err

echo "=========================================="
echo "SupCon ImageNet R152 - NOISE augmentation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"

export WANDB_MODE=offline
export WANDB_DIR=/leonardo_scratch/fast/CNHPC_1905882/clxai/logs/wandb

module load profile/deeplrn
module load cineca-ai/4.3.0

source /leonardo_scratch/fast/CNHPC_1905882/clxai/clxai_env/bin/activate
cd /leonardo_scratch/fast/CNHPC_1905882/clxai

mkdir -p logs/slurm logs/wandb results/models/imagenet_r152/supcon_noise

export CUDA_VISIBLE_DEVICES=0,1,2,3

python src/training/train_scl_imagenet.py \
    --config configs/supcon_imagenet_r152_noise.yaml \
    --run_name "supcon_r152_noise_${SLURM_JOB_ID}"

echo "Completed: $(date)"
