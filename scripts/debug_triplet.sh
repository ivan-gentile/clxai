#!/bin/bash
#SBATCH --account=CNHPC_1905882
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64000
#SBATCH --job-name=debug_triplet
#SBATCH --output=logs/slurm/debug_triplet_%j.out
#SBATCH --error=logs/slurm/debug_triplet_%j.err

echo "=========================================="
echo "DEBUG: Triplet Loss Training"
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
mkdir -p logs/slurm results/models/debug_triplet data wandb

# Clear previous debug log
rm -f /leonardo_scratch/fast/CNHPC_1905882/.cursor/debug.log

# Run short training with debug logging enabled
echo "Starting DEBUG Triplet training (30 epochs)..."
python -c "
import sys
sys.path.insert(0, '.')
from src.training.train_scl import train_scl_model

config = {
    'dataset': 'cifar10',
    'architecture': 'resnet18',
    'epochs': 30,
    'batch_size': 256,
    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'loss': 'triplet',
    'triplet': {
        'margin': 0.3,
        'mining': 'hard',
        'squared': False
    },
    'embedding_dim': 128,
    'data_dir': '/leonardo_scratch/fast/CNHPC_1905882/clxai/data',
    'output_dir': 'results/models/debug_triplet',
    'use_wandb': False,
    'run_name': 'debug_triplet',
    'num_workers': 4,
    'save_freq': 100,
    'eval_freq': 10,
    'warmup_epochs': 5,
    'knn_k': 10,
    'seed': 42,
    'debug_log': True  # ENABLE DEBUG LOGGING
}

train_scl_model(config)
"

echo ""
echo "Training completed at: $(date)"
echo "Debug log at: /leonardo_scratch/fast/CNHPC_1905882/.cursor/debug.log"
echo "=========================================="
