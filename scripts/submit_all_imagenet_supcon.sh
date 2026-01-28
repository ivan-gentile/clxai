#!/bin/bash
# Submit all three SupCon ImageNet training jobs

echo "=========================================="
echo "Submitting SupCon ImageNet R152 Trainings"
echo "=========================================="
echo ""

cd /leonardo_scratch/fast/CNHPC_1905882/clxai

# Submit BASE (no custom augmentation)
echo "1. Submitting BASE (no custom aug)..."
JOB1=$(sbatch scripts/train_supcon_imagenet_r152.sh | awk '{print $4}')
echo "   Job ID: $JOB1"

# Submit PIXEL50
echo "2. Submitting PIXEL50..."
JOB2=$(sbatch scripts/train_supcon_imagenet_r152_pixel50.sh | awk '{print $4}')
echo "   Job ID: $JOB2"

# Submit NOISE
echo "3. Submitting NOISE..."
JOB3=$(sbatch scripts/train_supcon_imagenet_r152_noise.sh | awk '{print $4}')
echo "   Job ID: $JOB3"

echo ""
echo "=========================================="
echo "All jobs submitted!"
echo "=========================================="
echo ""
echo "Monitor with: squeue -u \$USER"
echo ""
echo "Training configs:"
echo "  - BASE:    configs/supcon_imagenet_r152.yaml"
echo "  - PIXEL50: configs/supcon_imagenet_r152_pixel50.yaml"
echo "  - NOISE:   configs/supcon_imagenet_r152_noise.yaml"
echo ""
echo "Output directories:"
echo "  - results/models/imagenet_r152/supcon/"
echo "  - results/models/imagenet_r152/supcon_pixel50/"
echo "  - results/models/imagenet_r152/supcon_noise/"
