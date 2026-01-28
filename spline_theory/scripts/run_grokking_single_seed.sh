#!/bin/bash
#SBATCH --account=CNHPC_1905882
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64000
#SBATCH --job-name=spline_grok_single
#SBATCH --output=logs/slurm/grok_single_%j.out
#SBATCH --error=logs/slurm/grok_single_%j.err

# ============================================================================
# Grokking single-seed runs for CIFAR10 (CE, SupCon, Triplet)
# ============================================================================
# Usage examples:
#   sbatch --export=LOSS=ce,SEED=0 spline_theory/scripts/run_grokking_single_seed.sh
#   sbatch --export=LOSS=supcon,SEED=0 spline_theory/scripts/run_grokking_single_seed.sh
#   sbatch --export=LOSS=triplet,SEED=0 spline_theory/scripts/run_grokking_single_seed.sh
#   sbatch --export=LOSS=all,SEED=0 spline_theory/scripts/run_grokking_single_seed.sh
# ============================================================================

LOSS=${LOSS:-"ce"}
SEED=${SEED:-0}
CONFIG_PATH="/leonardo_scratch/fast/CNHPC_1905882/clxai/spline_theory/configs/grokking_single_seed.yaml"

echo "=========================================="
echo "Spline Theory - Grokking Single Seed Runs"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Loss: ${LOSS}"
echo "Seed: ${SEED}"
echo "Config: ${CONFIG_PATH}"
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

run_ce() {
    local output_dir="spline_theory/results/grokking_single_seed/ce_seed${SEED}"
    mkdir -p "${output_dir}" logs/slurm

    local ce_config_path="${output_dir}/ce_config.yaml"
    python - <<'PY'
import os
import yaml

config_path = os.environ["CONFIG_PATH"]
output_dir = os.environ["OUTPUT_DIR"]
seed = int(os.environ["SEED"])

with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

ce = cfg["ce"]
ce_config = {
    "dataset": "cifar10",
    "architecture": "resnet18",
    "norm_type": ce["norm_type"],
    "max_epochs": ce["max_epochs"],
    "batch_size": ce["batch_size"],
    "lr": ce["lr"],
    "momentum": ce["momentum"],
    "weight_decay": ce["weight_decay"],
    "optimizer": ce.get("optimizer", "adam"),
    "grad_clip": ce.get("grad_clip", 1.0),
    "checkpoint_dir": output_dir,
    "checkpoint_epochs": ce["checkpoint_epochs"],
    "eval_frequency": ce["eval_frequency"],
    "data_dir": ce["data_dir"],
    "num_workers": ce["num_workers"],
    "seed": seed,
    "use_wandb": False,
    "run_name": ce.get("run_name", f"grokking_ce_seed{seed}")
}

with open(os.environ["CE_CONFIG_PATH"], "w") as f:
    yaml.safe_dump(ce_config, f)
PY

    python spline_theory/training/extended_trainer.py \
        --config "${ce_config_path}" \
        --no_wandb
}

run_contrastive() {
    local loss_type=$1
    local output_dir="spline_theory/results/grokking_single_seed/${loss_type}_seed${SEED}"
    local run_name="grokking_${loss_type}_seed${SEED}"
    mkdir -p "${output_dir}" logs/slurm

    python src/training/train_scl.py \
        --config "${CONFIG_PATH}" \
        --loss "${loss_type}" \
        --epochs 10000 \
        --batch_size 256 \
        --lr 0.001 \
        --weight_decay 0.01 \
        --optimizer adam \
        --grad_clip 1.0 \
        --norm_type id \
        --output_dir "${output_dir}" \
        --run_name "${run_name}" \
        --seed "${SEED}" \
        --augmentation_type none \
        --no_wandb
}

case "${LOSS}" in
    ce)
        OUTPUT_DIR="spline_theory/results/grokking_single_seed/ce_seed${SEED}" \
        CE_CONFIG_PATH="spline_theory/results/grokking_single_seed/ce_seed${SEED}/ce_config.yaml" \
        CONFIG_PATH="${CONFIG_PATH}" \
        SEED="${SEED}" \
        run_ce
        ;;
    supcon|scl)
        run_contrastive "supcon"
        ;;
    triplet)
        run_contrastive "triplet"
        ;;
    all)
        OUTPUT_DIR="spline_theory/results/grokking_single_seed/ce_seed${SEED}" \
        CE_CONFIG_PATH="spline_theory/results/grokking_single_seed/ce_seed${SEED}/ce_config.yaml" \
        CONFIG_PATH="${CONFIG_PATH}" \
        SEED="${SEED}" \
        run_ce
        run_contrastive "supcon"
        run_contrastive "triplet"
        ;;
    *)
        echo "Unknown LOSS: ${LOSS} (use ce, supcon, triplet, or all)"
        exit 1
        ;;
esac

echo ""
echo "Grokking single-seed run completed at: $(date)"
echo "=========================================="
