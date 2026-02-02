#!/bin/bash
#SBATCH --job-name=clxai_xai_eval
#SBATCH --account=CNHPC_1905882
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/xai_eval_%j.out
#SBATCH --error=logs/xai_eval_%j.err

# =============================================================================
# CLXAI XAI Evaluation Job
# =============================================================================
# Usage:
#   sbatch scripts/submit_xai_eval.sh                           # CE pixel seed0, all metrics
#   sbatch scripts/submit_xai_eval.sh scl                       # SCL pixel seed0, all metrics
#   sbatch scripts/submit_xai_eval.sh ce pixel50 2              # CE pixel50 seed2
#   sbatch scripts/submit_xai_eval.sh scl pixel 0 pf irof       # SCL pixel seed0, specific metrics
#
# Arguments:
#   $1 = model_version (ce, scl, triplet) - default: ce
#   $2 = augmentation (pixel, pixel50) - default: pixel
#   $3 = seed (0-4) - default: 0
#   $4+ = metrics (pf, irof, sparseness, complexity, robustness, contrastivity, all)
# =============================================================================

# Parse arguments
MODEL_VERSION=${1:-ce}
AUGMENTATION=${2:-pixel}
SEED=${3:-0}
shift 3 2>/dev/null || true
METRICS=${@:-all}

echo "=================================================="
echo "CLXAI XAI Evaluation"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Model: ${MODEL_VERSION}_${AUGMENTATION} (seed ${SEED})"
echo "Metrics: $METRICS"
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

# Run evaluation
echo ""
echo "Running XAI evaluation..."
echo ""

python scripts/run_xai_evaluation.py \
    --model_version $MODEL_VERSION \
    --augmentation $AUGMENTATION \
    --model_seed $SEED \
    --dataset cifar10 \
    --data_dir data \
    --batch_size 128 \
    --metrics $METRICS \
    --output_dir results/xai_eval \
    --seed 42

EXIT_CODE=$?

echo ""
echo "=================================================="
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=================================================="

exit $EXIT_CODE
