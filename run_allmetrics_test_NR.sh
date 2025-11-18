#!/usr/bin/env bash
set -euo pipefail

# ================================================================
# Script to launch sequential trainings for all image quality metrics.
# Example usage:
#   ./run_all_metrics_test.sh 0     # Use GPU 0
# ================================================================

# -------------------------------
# User configuration
# -------------------------------
MODEL="tinyvit"                             # Model to test
CHECKPOINT_BASE="/home/jovyan/nfs/lsgroi/exp/exp_IMAGENET_New"   # Base dir with checkpoints
TEST_SCRIPT="/home/jovyan/python/Neural-No-Reference-SIM/testNR.py"   # Path to your test script
TEST_PATH="/home/jovyan/nfs/lsgroi/Dataset/KonIQ-10k"       # Dataset path
IMAGE_SIZE=384
BATCH_SIZE=128
MIN_SCORE=0.0
MAX_SAMPLES_PER_BIN=50
OUTPUT_DIR="./test_results/KonIQ-10k_their"

# List of all supported metrics
METRICS=(ssim fsim ms_ssim iw_ssim sr_sim vsi dss haarpsi mdsi)

# Log directory
LOG_DIR="test_logs"
mkdir -p "${LOG_DIR}"

# -------------------- GPU SELECTION -------------------- #

if [ $# -lt 1 ]; then
  echo "Usage: $0 <GPU_ID>"
  echo "Example: $0 0   # use GPU 0"
  exit 1
fi

GPU_ID=$1
export CUDA_VISIBLE_DEVICES=${GPU_ID}

for METRIC in "${METRICS[@]}"; do
  echo "============================================================"
  echo "Starting evaluation for metric: ${METRIC}"
  echo "============================================================"

  # Automatically detect the checkpoint path (model_metric_*.pth)
  CHECKPOINT_PATH=$(find "${CHECKPOINT_BASE}" -type f -name "best.pth" -path "*${MODEL}_${METRIC}_*" | head -n 1 || true)

  if [[ -z "${CHECKPOINT_PATH}" ]]; then
    echo "No checkpoint found for ${MODEL}_${METRIC}, skipping."
    continue
  fi

  echo "Using checkpoint: ${CHECKPOINT_PATH}"

  # Build log file
  LOG_FILE="${LOG_DIR}/${MODEL}_${METRIC}_test.log"

  # Build command
  CMD="python3 ${TEST_SCRIPT} \
      --model ${MODEL} \
      --checkpoint ${CHECKPOINT_PATH} \
      --test_path ${TEST_PATH} \
      --metric ${METRIC} \
      --batch_size ${BATCH_SIZE} \
      --image_size ${IMAGE_SIZE} \
      --min_score ${MIN_SCORE} \
      --output_dir ${OUTPUT_DIR} \
      --max_samples_per_bin ${MAX_SAMPLES_PER_BIN} "

  echo "Running command:"
  echo "${CMD}"
  echo "------------------------------------------------------------"

  # Run and log output
  ${CMD} 2>&1 | tee "${LOG_FILE}"

  echo "Finished evaluation for ${METRIC}. Log saved to ${LOG_FILE}"
  echo
done

echo "============================================================"
echo "All evaluations completed successfully."
echo "Logs saved in: ${LOG_DIR}/"
echo "============================================================"