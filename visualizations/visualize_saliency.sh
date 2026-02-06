#!/bin/bash
#SBATCH --job-name=viz_saliency
#SBATCH --account=CNHPC_1905882
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/viz_saliency_%j.out
#SBATCH --error=logs/viz_saliency_%j.err

# =============================================================================
# Saliency Map Visualization - CE / SCL / TL on ImageNet-S50
# =============================================================================
# Generates GradCAM + EigenCAM overlay images for a few random validation
# images across all three training objectives.
# =============================================================================

echo "=================================================="
echo "Saliency Map Visualization"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=================================================="

# Load modules
module load profile/deeplrn
module load cineca-ai/4.3.0

# Activate environment
source /leonardo_scratch/fast/CNHPC_1905882/clxai/clxai_env/bin/activate

# Change to project directory
cd /leonardo_scratch/fast/CNHPC_1905882/clxai

# Create log directory if needed
mkdir -p logs

# Run visualization
echo ""
echo "Running saliency visualization..."
echo ""

python visualizations/visualize_saliency.py \
    --num_images 5 \
    --seed 42 \
    --data_dir data/imagenet-s50/ImageNetS50 \
    --output_dir results/saliency_visualizations

EXIT_CODE=$?

echo ""
echo "=================================================="
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=================================================="

exit $EXIT_CODE
