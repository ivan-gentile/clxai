#!/bin/bash
#SBATCH --account=CNHPC_1905882
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64000
#SBATCH --job-name=spline_phase3
#SBATCH --output=logs/slurm/phase3_%j.out
#SBATCH --error=logs/slurm/phase3_%j.err

# ============================================================================
# Phase 3: Contrastive Learning Ablation Study
# ============================================================================
# Full factorial ablation across loss functions, regularization, and
# normalization layers.
#
# Usage:
#   sbatch --export=LOSS=supcon,NORM=bn,WD=0.0001,SEED=0 run_phase3_ablation.sh
# ============================================================================

# Get parameters from environment
LOSS=${LOSS:-"supcon"}
NORM=${NORM:-"bn"}
WD=${WD:-"0.0001"}
SEED=${SEED:-0}

echo "=========================================="
echo "Spline Theory - Phase 3: CL Ablation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Loss: ${LOSS}"
echo "Norm: ${NORM}"
echo "Weight Decay: ${WD}"
echo "Seed: ${SEED}"
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

# Create output directory
RUN_NAME="${LOSS}_${NORM}_wd${WD}_seed${SEED}"
OUTPUT_DIR="spline_theory/results/phase3_ablation/${RUN_NAME}"
mkdir -p "${OUTPUT_DIR}" logs/slurm

echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Run training with specified configuration
python -c "
import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from pathlib import Path
import json
import time

from spline_theory.models.resnet_variants import ResNetEncoderVariant
from src.training.losses import SupConLossV2, TripletLoss
from src.utils.data import get_data_loaders
from src.models.classifiers import KNNClassifier

# Configuration
LOSS = '${LOSS}'
NORM = '${NORM}'
WD = float('${WD}')
SEED = int('${SEED}')
OUTPUT_DIR = Path('${OUTPUT_DIR}')

# Set seed
torch.manual_seed(SEED)
np.random.seed(SEED)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')
print(f'Config: loss={LOSS}, norm={NORM}, wd={WD}, seed={SEED}')

# Data loaders
train_loader, test_loader = get_data_loaders(
    dataset='cifar10',
    data_dir='/leonardo_scratch/fast/CNHPC_1905882/clxai/data',
    batch_size=256,
    num_workers=4,
    contrastive=(LOSS == 'supcon'),
    augment=True
)

train_loader_eval, _ = get_data_loaders(
    dataset='cifar10',
    data_dir='/leonardo_scratch/fast/CNHPC_1905882/clxai/data',
    batch_size=256,
    num_workers=4,
    contrastive=False,
    augment=False
)

# Model
model = ResNetEncoderVariant(
    architecture='resnet18',
    embedding_dim=128,
    norm_type=NORM
).to(device)

# Loss
if LOSS == 'supcon':
    criterion = SupConLossV2(temperature=0.07)
else:
    criterion = TripletLoss(margin=0.3, mining='hard')

# Optimizer
optimizer = optim.SGD(
    model.parameters(),
    lr=0.5,
    momentum=0.9,
    weight_decay=WD
)

# Scheduler with warmup
epochs = 3000
warmup_epochs = 10

def lr_lambda(epoch):
    if epoch < warmup_epochs:
        return 0.1 + 0.9 * (epoch / warmup_epochs)
    else:
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))

scheduler = LambdaLR(optimizer, lr_lambda)

# Checkpoint epochs
checkpoint_epochs = [100, 200, 300, 500, 750, 1000, 1500, 2000, 3000]

# Training
best_acc = 0
history = {'epoch': [], 'loss': [], 'knn_acc': []}
start_time = time.time()

for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0
    
    for images, labels in train_loader:
        if isinstance(images, list):
            images = torch.cat([images[0], images[1]], dim=0).to(device)
        else:
            images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        if LOSS == 'supcon':
            embeddings = model(images)
        else:
            embeddings = model.get_embedding(images, normalize=False)
        
        loss = criterion(embeddings, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    
    # Evaluate periodically
    if epoch % 50 == 0 or epoch in checkpoint_epochs:
        model.eval()
        
        # Extract embeddings
        train_emb, train_labels = [], []
        test_emb, test_labels = [], []
        
        with torch.no_grad():
            for images, labels in train_loader_eval:
                emb = model.get_embedding(images.to(device), normalize=True)
                train_emb.append(emb.cpu())
                train_labels.append(labels)
            
            for images, labels in test_loader:
                emb = model.get_embedding(images.to(device), normalize=True)
                test_emb.append(emb.cpu())
                test_labels.append(labels)
        
        train_emb = torch.cat(train_emb).numpy()
        train_labels = torch.cat(train_labels).numpy()
        test_emb = torch.cat(test_emb).numpy()
        test_labels = torch.cat(test_labels).numpy()
        
        # kNN evaluation
        knn = KNNClassifier(k=10, metric='cosine')
        knn.fit(train_emb, train_labels)
        knn_acc = knn.score(test_emb, test_labels) * 100
        
        history['epoch'].append(epoch)
        history['loss'].append(avg_loss)
        history['knn_acc'].append(knn_acc)
        
        elapsed = time.time() - start_time
        print(f'Epoch {epoch}/{epochs} [{elapsed/3600:.1f}h] loss={avg_loss:.4f} kNN={knn_acc:.2f}%')
        
        # Save best
        if knn_acc > best_acc:
            best_acc = knn_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'knn_acc': best_acc
            }, OUTPUT_DIR / 'best_model.pt')
        
        # Save checkpoint
        if epoch in checkpoint_epochs:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'knn_acc': knn_acc
            }, OUTPUT_DIR / f'checkpoint_epoch_{epoch}.pt')

# Save final
torch.save({
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'history': history
}, OUTPUT_DIR / 'final_model.pt')

with open(OUTPUT_DIR / 'history.json', 'w') as f:
    json.dump(history, f)

print(f'Training completed. Best kNN accuracy: {best_acc:.2f}%')
"

echo ""
echo "Phase 3 ablation completed at: $(date)"
echo "Output saved to: ${OUTPUT_DIR}"
echo "=========================================="
