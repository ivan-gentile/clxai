#!/bin/bash
#SBATCH --account=CNHPC_1469675
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64000
#SBATCH --job-name=test_imagenet_quick
#SBATCH --output=logs/slurm/test_imagenet_quick_%j.out
#SBATCH --error=logs/slurm/test_imagenet_quick_%j.err

echo "=========================================="
echo "Quick Test: ImageNet SCL Pipeline"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Account: $SLURM_JOB_ACCOUNT"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""

# Load modules
module load profile/deeplrn
module load cineca-ai/4.3.0

# Activate environment
source /leonardo_scratch/fast/CNHPC_1905882/clxai/clxai_env/bin/activate

# Navigate to project
cd /leonardo_scratch/fast/CNHPC_1905882/clxai

# Create directories
mkdir -p logs/slurm results

# Test write permissions first
echo "Testing write permissions..."
echo "Test from job $SLURM_JOB_ID at $(date)" > results/test_write_$SLURM_JOB_ID.txt
if [ $? -eq 0 ]; then
    echo "OK: Write permission verified"
    rm results/test_write_$SLURM_JOB_ID.txt
else
    echo "ERROR: Write permission failed!"
    exit 1
fi

# Run quick test
python scripts/test_imagenet_quick.py

echo ""
echo "Test completed at: $(date)"
echo "=========================================="
