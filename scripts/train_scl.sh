#!/bin/bash
#SBATCH --account=CNHPC_1905882
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64000
#SBATCH --job-name=clxai_scl
#SBATCH --output=logs/slurm/scl_supcon_%j.out
#SBATCH --error=logs/slurm/scl_supcon_%j.err

echo "=========================================="
echo "CLXAI: Supervised Contrastive Learning"
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
mkdir -p logs/slurm results/models/scl data wandb

# Run training
echo "Starting SCL training..."
python src/training/train_scl.py \
    --config configs/scl_supcon.yaml \
    --epochs 500 \
    --batch_size 256 \
    --lr 0.5 \
    --temperature 0.07 \
    --data_dir ./data \
    --output_dir results/models/scl \
    --run_name scl_supcon

echo ""
echo "Training completed at: $(date)"
echo "=========================================="
