#!/bin/bash
#SBATCH --account=CNHPC_1905882
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64000
#SBATCH --job-name=spline_phase1
#SBATCH --output=logs/slurm/phase1_%j.out
#SBATCH --error=logs/slurm/phase1_%j.err

# ============================================================================
# Phase 1: Diagnostic Analysis of Existing Models
# ============================================================================

echo "=========================================="
echo "Spline Theory - Phase 1: Diagnostic Analysis"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
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

# Create directories
mkdir -p logs/slurm spline_theory/results/phase1_diagnostic

# Run Phase 1 diagnostic analysis
echo "Starting Phase 1 diagnostic analysis..."
python << 'PYTHON_SCRIPT'
import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.models.resnet import ResNet18, ResNet18Encoder
from src.models.classifiers import LinearClassifier
from src.utils.data import get_data_loaders
from spline_theory.evaluation.adversarial import AdversarialEvaluator

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Output directory
output_dir = Path('spline_theory/results/phase1_diagnostic')
output_dir.mkdir(parents=True, exist_ok=True)

# Get data loaders
train_loader, test_loader = get_data_loaders(
    dataset='cifar10',
    data_dir='data',
    batch_size=128,
    num_workers=4,
    augment=False
)

def train_linear_classifier_quick(encoder, train_loader, device, epochs=20):
    """Train a linear classifier on frozen embeddings."""
    encoder.eval()
    
    # Extract embeddings
    print('    Extracting embeddings for linear classifier...')
    train_emb, train_labels = [], []
    with torch.no_grad():
        for images, labels in train_loader:
            emb = encoder.get_embedding(images.to(device), normalize=True)
            train_emb.append(emb.cpu())
            train_labels.append(labels)
    
    train_emb = torch.cat(train_emb)
    train_labels = torch.cat(train_labels)
    
    # Train linear classifier
    clf = LinearClassifier(input_dim=512, num_classes=10).to(device)
    optimizer = optim.Adam(clf.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Create dataset
    from torch.utils.data import TensorDataset, DataLoader
    train_ds = TensorDataset(train_emb, train_labels)
    emb_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    
    for epoch in range(epochs):
        clf.train()
        for emb_batch, label_batch in emb_loader:
            emb_batch, label_batch = emb_batch.to(device), label_batch.to(device)
            optimizer.zero_grad()
            outputs = clf(emb_batch)
            loss = criterion(outputs, label_batch)
            loss.backward()
            optimizer.step()
    
    clf.eval()
    return clf

results = {}

# ============================================================================
# Analyze CE models
# ============================================================================
print('\n' + '=' * 60)
print('Analyzing Cross-Entropy Models')
print('=' * 60)

ce_results = []
for seed in range(5):
    model_path = Path(f'results/models/ce_seed{seed}/best_model.pt')
    if not model_path.exists():
        print(f'  Seed {seed}: NOT FOUND')
        continue
    
    print(f'  Loading CE seed {seed}...')
    model = ResNet18(num_classes=10).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Evaluate
    evaluator = AdversarialEvaluator(model, device=device, eps=8/255, pgd_steps=20)
    metrics = evaluator.evaluate_all(test_loader, n_samples=1000)
    
    ce_results.append({
        'seed': seed,
        'clean_accuracy': metrics['clean_accuracy'],
        'fgsm_accuracy': metrics['fgsm_accuracy'],
        'pgd_accuracy': metrics['pgd_accuracy']
    })
    
    print(f'    Clean: {metrics["clean_accuracy"]*100:.2f}%, FGSM: {metrics["fgsm_accuracy"]*100:.2f}%, PGD: {metrics["pgd_accuracy"]*100:.2f}%')

results['ce_baseline'] = ce_results

# ============================================================================
# Analyze SCL models
# ============================================================================
print('\n' + '=' * 60)
print('Analyzing Supervised Contrastive Learning Models')
print('=' * 60)

class CombinedModel(nn.Module):
    def __init__(self, encoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
    
    def forward(self, x):
        features = self.encoder.get_embedding(x, normalize=True)
        return self.classifier(features)

scl_results = []
for seed in range(5):
    model_path = Path(f'results/models/scl_seed{seed}/best_model.pt')
    if not model_path.exists():
        print(f'  Seed {seed}: NOT FOUND')
        continue
    
    print(f'  Loading SCL seed {seed}...')
    
    encoder = ResNet18Encoder(embedding_dim=128).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    encoder.load_state_dict(checkpoint['model_state_dict'])
    encoder.eval()
    
    # Load or train linear classifier
    linear_path = Path(f'results/models/scl_seed{seed}/linear_classifier.pt')
    if linear_path.exists():
        linear_clf = LinearClassifier(input_dim=512, num_classes=10).to(device)
        linear_clf.load_state_dict(torch.load(linear_path, map_location=device))
        linear_clf.eval()
    else:
        print(f'    Training linear classifier...')
        linear_clf = train_linear_classifier_quick(encoder, train_loader, device)
    
    model = CombinedModel(encoder, linear_clf).to(device)
    
    # Evaluate
    evaluator = AdversarialEvaluator(model, device=device, eps=8/255, pgd_steps=20)
    metrics = evaluator.evaluate_all(test_loader, n_samples=1000)
    
    scl_results.append({
        'seed': seed,
        'clean_accuracy': metrics['clean_accuracy'],
        'fgsm_accuracy': metrics['fgsm_accuracy'],
        'pgd_accuracy': metrics['pgd_accuracy']
    })
    
    print(f'    Clean: {metrics["clean_accuracy"]*100:.2f}%, FGSM: {metrics["fgsm_accuracy"]*100:.2f}%, PGD: {metrics["pgd_accuracy"]*100:.2f}%')

results['scl_supcon'] = scl_results

# ============================================================================
# Analyze Triplet models
# ============================================================================
print('\n' + '=' * 60)
print('Analyzing Triplet Loss Models')
print('=' * 60)

triplet_results = []
for seed in range(5):
    model_path = Path(f'results/models/triplet_fixed_seed{seed}/best_model.pt')
    if not model_path.exists():
        print(f'  Seed {seed}: NOT FOUND')
        continue
    
    print(f'  Loading Triplet seed {seed}...')
    
    encoder = ResNet18Encoder(embedding_dim=128).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    encoder.load_state_dict(checkpoint['model_state_dict'])
    encoder.eval()
    
    # Load or train linear classifier
    linear_path = Path(f'results/models/triplet_fixed_seed{seed}/linear_classifier.pt')
    if linear_path.exists():
        linear_clf = LinearClassifier(input_dim=512, num_classes=10).to(device)
        linear_clf.load_state_dict(torch.load(linear_path, map_location=device))
        linear_clf.eval()
    else:
        print(f'    Training linear classifier...')
        linear_clf = train_linear_classifier_quick(encoder, train_loader, device)
        # Save it for future use
        torch.save(linear_clf.state_dict(), linear_path)
        print(f'    Saved linear classifier to {linear_path}')
    
    model = CombinedModel(encoder, linear_clf).to(device)
    
    # Evaluate
    evaluator = AdversarialEvaluator(model, device=device, eps=8/255, pgd_steps=20)
    metrics = evaluator.evaluate_all(test_loader, n_samples=1000)
    
    triplet_results.append({
        'seed': seed,
        'clean_accuracy': metrics['clean_accuracy'],
        'fgsm_accuracy': metrics['fgsm_accuracy'],
        'pgd_accuracy': metrics['pgd_accuracy']
    })
    
    print(f'    Clean: {metrics["clean_accuracy"]*100:.2f}%, FGSM: {metrics["fgsm_accuracy"]*100:.2f}%, PGD: {metrics["pgd_accuracy"]*100:.2f}%')

results['triplet'] = triplet_results

# ============================================================================
# Summary and Save
# ============================================================================
print('\n' + '=' * 60)
print('PHASE 1 SUMMARY')
print('=' * 60)

summary = {}
for model_type, model_results in results.items():
    if model_results:
        clean_accs = [r['clean_accuracy'] for r in model_results]
        fgsm_accs = [r['fgsm_accuracy'] for r in model_results]
        pgd_accs = [r['pgd_accuracy'] for r in model_results]
        
        summary[model_type] = {
            'clean_mean': np.mean(clean_accs),
            'clean_std': np.std(clean_accs),
            'fgsm_mean': np.mean(fgsm_accs),
            'fgsm_std': np.std(fgsm_accs),
            'pgd_mean': np.mean(pgd_accs),
            'pgd_std': np.std(pgd_accs),
            'n_seeds': len(model_results)
        }
        
        print(f'\n{model_type.upper()} ({len(model_results)} seeds):')
        print(f'  Clean:  {np.mean(clean_accs)*100:.2f}% ± {np.std(clean_accs)*100:.2f}%')
        print(f'  FGSM:   {np.mean(fgsm_accs)*100:.2f}% ± {np.std(fgsm_accs)*100:.2f}%')
        print(f'  PGD:    {np.mean(pgd_accs)*100:.2f}% ± {np.std(pgd_accs)*100:.2f}%')

# Test hypothesis P1: CL more robust than CE
if 'ce_baseline' in summary and 'scl_supcon' in summary:
    ce_pgd = summary['ce_baseline']['pgd_mean']
    scl_pgd = summary['scl_supcon']['pgd_mean']
    print(f'\n--- Hypothesis P1 (CL more robust) ---')
    print(f'  CE PGD:  {ce_pgd*100:.2f}%')
    print(f'  SCL PGD: {scl_pgd*100:.2f}%')
    print(f'  Difference: {(scl_pgd - ce_pgd)*100:+.2f}%')
    print(f'  P1 SUPPORTED: {scl_pgd > ce_pgd}')

# Save results
results['summary'] = summary
with open(output_dir / 'phase1_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f'\nResults saved to {output_dir}/phase1_results.json')
print('\nPhase 1 completed successfully!')
PYTHON_SCRIPT

echo ""
echo "Phase 1 completed at: $(date)"
echo "=========================================="
