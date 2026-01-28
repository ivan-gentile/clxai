#!/bin/bash
# Submit resume jobs for ImageNet SupCon training
# Updated: temperature=0.07, epochs=420

echo "=========================================="
echo "Submitting ImageNet SupCon Resume Jobs"
echo "=========================================="
echo "Changes:"
echo "  - Temperature: 0.1 → 0.07 (SupCon paper optimal)"
echo "  - Epochs: 350 → 420"
echo "  - BASE & PIXEL50: Resume from epoch 210"
echo "  - NOISE: Fresh start (never ran before)"
echo "=========================================="

cd /leonardo_scratch/fast/CNHPC_1905882/clxai

# Make scripts executable
chmod +x scripts/resume_supcon_imagenet_r152_multinode.sh
chmod +x scripts/resume_supcon_imagenet_r152_pixel50_multinode.sh
chmod +x scripts/train_supcon_imagenet_r152_noise_multinode_fresh.sh

echo ""
echo "1. Submitting BASE (resume from epoch 210)..."
JOB1=$(sbatch scripts/resume_supcon_imagenet_r152_multinode.sh | awk '{print $4}')
echo "   Job ID: $JOB1"

echo ""
echo "2. Submitting PIXEL50 (resume from epoch 210)..."
JOB2=$(sbatch scripts/resume_supcon_imagenet_r152_pixel50_multinode.sh | awk '{print $4}')
echo "   Job ID: $JOB2"

echo ""
echo "3. Submitting NOISE (fresh start)..."
JOB3=$(sbatch scripts/train_supcon_imagenet_r152_noise_multinode_fresh.sh | awk '{print $4}')
echo "   Job ID: $JOB3"

echo ""
echo "=========================================="
echo "Submitted 3 jobs:"
echo "  BASE (resume):   $JOB1 - epochs 211→420"
echo "  PIXEL50 (resume): $JOB2 - epochs 211→420"  
echo "  NOISE (fresh):   $JOB3 - epochs 1→420"
echo ""
echo "Expected improvements with temperature=0.07:"
echo "  - Sharper contrastive learning"
echo "  - Better class separation"
echo "  - +10-15% kNN accuracy"
echo ""
echo "Monitor with: squeue -u \$USER"
