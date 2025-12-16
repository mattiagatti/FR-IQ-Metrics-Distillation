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
TEST_SCRIPT="/home/jovyan/python/Neural-No-Reference-SIM/testNR.py"   # Path to your test script
CONFIG_PATH="/home/jovyan/python/Neural-No-Reference-SIM/experiments/test_KonIQ10k_KonIQ10k_MOS.yaml"  # Config file path
# List of all supported metrics
METRICS=(mos)


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

  # Build command
  CMD="python3 ${TEST_SCRIPT} \
      --model ${MODEL} \
      --metric ${METRIC} \
      --config_path ${CONFIG_PATH}"

  echo "Running command:"
  echo "${CMD}"
  echo "------------------------------------------------------------"

  # Run and log output
  ${CMD[@]}

  echo "Finished evaluation for ${METRIC}."
  echo
done

echo "============================================================"
echo "All evaluations completed successfully."
echo "Logs saved in: ${LOG_DIR}/"
echo "============================================================"