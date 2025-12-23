#!/bin/bash
#SBATCH --account=CNHPC_1905882
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64000
#SBATCH --job-name=cifar100_r152
#SBATCH --output=logs/slurm/cifar100_r152_%j.out
#SBATCH --error=logs/slurm/cifar100_r152_%j.err

echo "=========================================="
echo "CLXAI: CIFAR-100 + ResNet-152 Training"
echo "Scalability Experiments (3 models)"
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
source /leonardo_scratch/fast/CNHPC_1905882/clxai/clxai_env/bin/activate 2>/dev/null || {
    echo "Creating virtual environment..."
    python -m venv /leonardo_scratch/fast/CNHPC_1905882/clxai/clxai_env --system-site-packages
    source /leonardo_scratch/fast/CNHPC_1905882/clxai/clxai_env/bin/activate
    pip install captum pytorch-metric-learning umap-learn
}

# Navigate to project
cd /leonardo_scratch/fast/CNHPC_1905882/clxai

# Create directories
mkdir -p logs/slurm results/models/cifar100_r152/{ce,supcon,triplet} data wandb

# Print GPU info
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# ============================================================================
# Model 1: Cross-Entropy Baseline
# ============================================================================
echo "=========================================="
echo "[1/3] Training CE Baseline on CIFAR-100 + ResNet-152"
echo "Started at: $(date)"
echo "=========================================="

python src/training/train_ce.py \
    --config configs/ce_cifar100_r152.yaml

echo "CE training completed at: $(date)"
echo ""

# ============================================================================
# Model 2: Supervised Contrastive Learning (SupCon)
# ============================================================================
echo "=========================================="
echo "[2/3] Training SupCon on CIFAR-100 + ResNet-152"
echo "Started at: $(date)"
echo "=========================================="

python src/training/train_scl.py \
    --config configs/supcon_cifar100_r152.yaml

echo "SupCon training completed at: $(date)"
echo ""

# ============================================================================
# Model 3: Triplet Loss with Hard Mining
# ============================================================================
echo "=========================================="
echo "[3/3] Training Triplet Loss on CIFAR-100 + ResNet-152"
echo "Started at: $(date)"
echo "=========================================="

python src/training/train_scl.py \
    --config configs/triplet_cifar100_r152.yaml

echo "Triplet training completed at: $(date)"
echo ""

# ============================================================================
# Summary
# ============================================================================
echo "=========================================="
echo "All CIFAR-100 + ResNet-152 training completed!"
echo "End time: $(date)"
echo "=========================================="

echo ""
echo "Results saved to:"
echo "  - results/models/cifar100_r152/ce/"
echo "  - results/models/cifar100_r152/supcon/"
echo "  - results/models/cifar100_r152/triplet/"
echo ""
echo "To sync wandb logs, run:"
echo "  wandb sync wandb/offline-run-*"

