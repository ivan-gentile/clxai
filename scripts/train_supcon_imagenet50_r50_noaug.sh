#!/bin/bash
#SBATCH --account=CNHPC_1905882
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=494000
#SBATCH --job-name=supcon_in50_noaug
#SBATCH --output=logs/slurm/supcon_imagenet50_r50_noaug_%j.out
#SBATCH --error=logs/slurm/supcon_imagenet50_r50_noaug_%j.err

echo "=========================================="
echo "SupCon ImageNet-50 ResNet-50 (NO F-Fidelity Aug)"
echo "Tests: Is SCL naturally robust without pixel removal?"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: 4 (DataParallel)"
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
mkdir -p logs/slurm logs/wandb results/models/imagenet50_r50/supcon_noaug

# Print config summary
echo ""
echo "Configuration:"
echo "  Architecture: ResNet-50 (from scratch)"
echo "  Dataset: ImageNet-50 (50-class subset)"
echo "  Epochs: 160"
echo "  Batch size: 128 per GPU x 4 GPUs = 512 effective"
echo "  Optimizer: RAdam (lr=0.001)"
echo "  Temperature: 0.07"
echo "  Augmentation: NONE (standard contrastive only)"
echo "  Purpose: Test if SCL is naturally robust"
echo ""

# Run training
python src/training/train_scl_imagenet50.py \
    --config configs/supcon_imagenet50_r50_noaug.yaml \
    --run_name "supcon_imagenet50_r50_noaug_${SLURM_JOB_ID}"

echo ""
echo "Completed: $(date)"
