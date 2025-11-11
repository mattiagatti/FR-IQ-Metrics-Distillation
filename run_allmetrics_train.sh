#!/usr/bin/env bash
set -euo pipefail

# ================================================================
# Script to launch sequential trainings for all image quality metrics.
# Example usage:
#   ./run_all_metrics.sh 0     # Use GPU 0
# ================================================================

# -------------------- CONFIGURATION -------------------- #
MODEL="swinv2"
TRAIN_PATH="/home/jovyan/nfs/datasets/ILSVRC2012_degraded/train"
VAL_PATH="/home/jovyan/nfs/datasets/ILSVRC2012_degraded/val"

EPOCHS=100
BATCH_SIZE=32
IMAGE_SIZE=384
LR=1e-4
TRAIN_TARGET_PER_BIN=100
VAL_TARGET_PER_BIN=100
MIN_SCORE=0.0
MAX_SCORE=1.0
PATIENCE=10
EXPERIMENT_PATH="/home/jovyan/nfs/lsgroi/exp"

# Metrics to train on
METRICS=(ssim fsim ms_ssim iw_ssim sr_sim vsi dss haarpsi mdsi)

# Path to Python training script
TRAIN_PY="/home/jovyan/python/Neural-No-Reference-SIM/train.py"
PYTHON_BIN="python3"

# Logs directory
LOG_DIR="exp/logs_all_metrics"
mkdir -p "${LOG_DIR}"

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
echo "Train path: ${TRAIN_PATH}"
echo "Val path:   ${VAL_PATH:-<none>}"
echo "==============================================================="

# -------------------- TRAINING LOOP -------------------- #
for M in "${METRICS[@]}"; do
  echo "=== Starting training for metric: ${M} ==="
  EXP_NAME="${MODEL}_${M}"
  LOG_FILE="${LOG_DIR}/${EXP_NAME}_gpu${GPU_ID}.log"

  if [ -n "${VAL_PATH}" ]; then
    ${PYTHON_BIN} "${TRAIN_PY}" \
      --model "${MODEL}" \
      --train_path "${TRAIN_PATH}" \
      --val_path "${VAL_PATH}" \
      --target "${M}" \
      --epochs "${EPOCHS}" \
      --batch_size "${BATCH_SIZE}" \
      --image_size "${IMAGE_SIZE}" \
      --lr "${LR}" \
      --train_target_per_bin "${TRAIN_TARGET_PER_BIN}" \
      --val_target_per_bin "${VAL_TARGET_PER_BIN}" \
      --min_score "${MIN_SCORE}" \
      --max_score "${MAX_SCORE}" \
      --patience "${PATIENCE}" \
      --experiment_path "${EXPERIMENT_PATH}" \
      2>&1 | tee "${LOG_FILE}"
  else
    ${PYTHON_BIN} "${TRAIN_PY}" \
      --model "${MODEL}" \
      --train_path "${TRAIN_PATH}" \
      --target "${M}" \
      --epochs "${EPOCHS}" \
      --batch_size "${BATCH_SIZE}" \
      --image_size "${IMAGE_SIZE}" \
      --lr "${LR}" \
      --train_target_per_bin "${TRAIN_TARGET_PER_BIN}" \
      --val_target_per_bin "${VAL_TARGET_PER_BIN}" \
      --min_score "${MIN_SCORE}" \
      --max_score "${MAX_SCORE}" \
      --patience "${PATIENCE}" \
      --experiment_path "${EXPERIMENT_PATH}" \
      2>&1 | tee "${LOG_FILE}"
  fi

  echo "=== Finished training for: ${M} (log: ${LOG_FILE}) ==="
  echo
done

echo "All trainings completed successfully on GPU ${GPU_ID}."