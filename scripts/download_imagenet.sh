#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30800
#SBATCH --partition=lrd_all_serial
#SBATCH --account=CNHPC_1905882
#SBATCH --job-name=imagenet_download
#SBATCH --output=logs/imagenet_download_%j.out
#SBATCH --error=logs/imagenet_download_%j.err

# ============================================================================
# ImageNet-1k Download Script for Leonardo Booster
# ============================================================================
# 
# BEFORE RUNNING THIS SCRIPT:
# 1. Go to https://huggingface.co/datasets/ILSVRC/imagenet-1k
# 2. Log in to your Hugging Face account
# 3. Accept the ImageNet terms of access
# 4. Your HF token must have read access to the dataset
#
# Dataset info:
# - Total size: ~155GB
# - Train: 1,281,167 images
# - Validation: 50,000 images  
# - Test: 100,000 images (labels not available)
#
# Estimated download time: 2-6 hours depending on network
# ============================================================================

echo "=========================================="
echo "ImageNet-1k Download Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Go to project directory
cd /leonardo_scratch/fast/CNHPC_1905882/clxai

# Create logs directory
mkdir -p logs

# Load necessary modules
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0

# Use Python from cineca-ai module (has huggingface_hub and datasets)
echo "Using Python from cineca-ai module"

# Set Hugging Face environment variables
export HF_TOKEN=""
export HUGGING_FACE_HUB_TOKEN=""
export HF_HOME="/leonardo_scratch/fast/CNHPC_1905882/.cache/huggingface"
export HF_HUB_CACHE="/leonardo_scratch/fast/CNHPC_1905882/.cache/huggingface/hub"
export HF_DATASETS_CACHE="/leonardo_scratch/fast/CNHPC_1905882/.cache/huggingface/datasets"

# Disable hf_transfer (not available in cineca-ai module)
export HF_HUB_ENABLE_HF_TRANSFER=0

# Create cache directories
mkdir -p $HF_HOME
mkdir -p $HF_HUB_CACHE
mkdir -p $HF_DATASETS_CACHE

echo ""
echo "Environment:"
echo "  Python: $(which python)"
echo "  HF_HOME: $HF_HOME"
echo "  HF_DATASETS_CACHE: $HF_DATASETS_CACHE"
echo ""

# Check disk space
echo "Disk space before download:"
df -h /leonardo_scratch/fast/CNHPC_1905882/
echo ""

# Note: hf_transfer not available in cineca-ai, using standard download

# Run download script
echo "Starting ImageNet-1k download..."
echo ""

# Download train and validation sets (test set has no labels)
# Use --subset to download specific splits if needed
python scripts/download_imagenet.py

# Check completion status
DOWNLOAD_STATUS=$?

echo ""
echo "=========================================="
if [ $DOWNLOAD_STATUS -eq 0 ]; then
    echo "Download completed successfully!"
else
    echo "Download failed with status: $DOWNLOAD_STATUS"
fi
echo "End time: $(date)"
echo "=========================================="

# Show disk usage
echo ""
echo "Disk space after download:"
df -h /leonardo_scratch/fast/CNHPC_1905882/
echo ""
echo "ImageNet directory size:"
du -sh /leonardo_scratch/fast/CNHPC_1905882/clxai/data/imagenet-1k/ 2>/dev/null || echo "Directory not found"

exit $DOWNLOAD_STATUS
