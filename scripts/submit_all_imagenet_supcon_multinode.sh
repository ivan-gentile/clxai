#!/bin/bash
# Submit all ImageNet SupCon R152 multi-node training jobs

echo "Submitting ImageNet SupCon R152 Multi-Node Training Jobs"
echo "=========================================="
echo "Configuration:"
echo "  - 4 nodes × 4 GPUs = 16 GPUs per job"
echo "  - Batch size: 128/GPU × 16 = 2048 effective"
echo "  - Training: ~4x faster than single node"
echo "=========================================="

cd /leonardo_scratch/fast/CNHPC_1905882/clxai

# Make scripts executable
chmod +x scripts/train_supcon_imagenet_r152_multinode.sh
chmod +x scripts/train_supcon_imagenet_r152_pixel50_multinode.sh
chmod +x scripts/train_supcon_imagenet_r152_noise_multinode.sh

# Submit jobs
echo ""
echo "1. Submitting BASE (no custom aug)..."
JOB1=$(sbatch scripts/train_supcon_imagenet_r152_multinode.sh | awk '{print $4}')
echo "   Job ID: $JOB1"

echo ""
echo "2. Submitting PIXEL50 (F-Fidelity augmentation)..."
JOB2=$(sbatch scripts/train_supcon_imagenet_r152_pixel50_multinode.sh | awk '{print $4}')
echo "   Job ID: $JOB2"

echo ""
echo "3. Submitting NOISE (Gaussian noise augmentation)..."
JOB3=$(sbatch scripts/train_supcon_imagenet_r152_noise_multinode.sh | awk '{print $4}')
echo "   Job ID: $JOB3"

echo ""
echo "=========================================="
echo "Submitted 3 multi-node jobs:"
echo "  BASE:    $JOB1"
echo "  PIXEL50: $JOB2"
echo "  NOISE:   $JOB3"
echo ""
echo "Total compute: 12 nodes (48 GPUs)"
echo "Expected time: ~12 hours for 350 epochs (vs ~48h single node)"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Logs in: logs/slurm/"
