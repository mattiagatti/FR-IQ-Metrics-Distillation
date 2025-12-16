#!/usr/bin/env bash
set -euo pipefail

# ================================================================
# Script to launch sequential trainings for all image quality metrics
# and all models.
# Example usage:
#   ./run_all_metrics_train.sh 0     # Use GPU 0
# ================================================================

# -------------------- CONFIGURATION -------------------- #
CONFIG_PATH="./experiments/train_IMAGENET_our_04.yaml"

# Models to train
MODELS=(mobilevitv2 resnet50v2 vit efficientnet mobilenetv3 tinyvit)

# Metrics to train on
METRICS=(ssim fsim sr_sim vsi dss)

# Path to Python training script
TRAIN_PY="./train.py"
PYTHON_BIN="python3"

# -------------------- GPU SELECTION -------------------- #
if [ $# -lt 1 ]; then
  echo "Usage: $0 <GPU_ID>"
  echo "Example: $0 0   # use GPU 0"
  exit 1
fi

GPU_ID=$1
export CUDA_VISIBLE_DEVICES=${GPU_ID}

echo "==============================================================="
echo "Running sequential trainings on GPU ${GPU_ID}"
echo "Models: ${MODELS[*]}"
echo "Metrics: ${METRICS[*]}"
echo "==============================================================="

# -------------------- TRAINING LOOP -------------------- #
for MODEL in "${MODELS[@]}"; do
  echo "=============================================================="
  echo "=== Starting trainings for model: ${MODEL} ==="
  echo "=============================================================="
  
  for M in "${METRICS[@]}"; do
    echo "=== Training ${MODEL} on metric: ${M} ==="

    ${PYTHON_BIN} "${TRAIN_PY}" \
      --model "${MODEL}" \
      --metric "${M}" \
      --config_path "${CONFIG_PATH}"

    echo "=== Finished ${MODEL} - ${M} ==="
    echo
  done
  
  echo "=== Completed all metrics for model: ${MODEL} ==="
  echo
done

echo "All trainings completed successfully on GPU ${GPU_ID}."
echo "Total combinations: ${#MODELS[@]} models x ${#METRICS[@]} metrics = $((${#MODELS[@]} * ${#METRICS[@]})) trainings"