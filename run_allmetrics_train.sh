#!/usr/bin/env bash
set -euo pipefail

# ================================================================
# Script to launch sequential trainings for all image quality metrics.
# Example usage:
#   ./run_all_metrics_train.sh 0     # Use GPU 0
# ================================================================

# -------------------- CONFIGURATION -------------------- #
MODEL="tinyvit"
CONFIG_PATH="/home/jovyan/python/Neural-No-Reference-SIM/experiments/train_Combined_IQA.yaml"

# Metrics to train on
METRICS=(ssim fsim ms_ssim iw_ssim sr_sim vsi dss haarpsi mdsi)

# Path to Python training script
TRAIN_PY="/home/jovyan/python/Neural-No-Reference-SIM/train.py"
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
echo "Model: ${MODEL}"
echo "==============================================================="

# -------------------- TRAINING LOOP -------------------- #
for M in "${METRICS[@]}"; do
  echo "=== Starting training for metric: ${M} ==="

  ${PYTHON_BIN} "${TRAIN_PY}" \
    --model "${MODEL}" \
    --metric "${M}" \
    --config_path "${CONFIG_PATH}"

  echo "=== Finished training for: ${M} ==="
  echo
done

echo "All trainings completed successfully on GPU ${GPU_ID}."