#!/usr/bin/env bash
set -euo pipefail

# ================================================================
# Script to launch sequential trainings for all image quality metrics.
# Logs to stdout only (no files).
# Example usage:
#   ./run_all_metrics_test.sh 0     # Use GPU 0
# ================================================================

# -------------------------------
# User configuration
# -------------------------------
MODEL="tinyvit"                             # Model to test
TEST_SCRIPT="/home/jovyan/python/Neural-No-Reference-SIM/test.py"   # Path to your test script
# List of all supported metrics
METRICS=(ssim fsim ms_ssim iw_ssim sr_sim vsi dss haarpsi mdsi)
CONFIG_PATH="/home/jovyan/python/Neural-No-Reference-SIM/experiments/test_Combined_IQA_Combined_IQA.yaml"  # Config file path
ALSO_MOS=true                             # Whether to also test with MOS

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

  CMD=(python3 "${TEST_SCRIPT}" \
      --model "${MODEL}" \
      --metric "${METRIC}" \
      --config_path "${CONFIG_PATH}" \
      --mos false)

  echo "Running command:"
  echo "${CMD[@]}"
  echo "------------------------------------------------------------"

  # Run and print output to stdout
  ${CMD[@]}

  echo "Finished evaluation for ${METRIC}."

  # If ALSO_MOS is true, run again with MOS
  if [ "${ALSO_MOS}" = true ]; then
    echo "Starting evaluation for metric: ${METRIC} with MOS"
    CMD=(python3 "${TEST_SCRIPT}" \
      --model "${MODEL}" \
      --metric "${METRIC}" \
      --config_path "${CONFIG_PATH}" \
      --mos true)

    echo "▶️  Running command:"
    echo "${CMD[@]}"
    echo "------------------------------------------------------------"

    # Run and print output to stdout
    ${CMD[@]}

    echo "Finished evaluation for ${METRIC} with MOS."
  fi

  echo
done

echo "============================================================"
echo "All evaluations completed successfully."
echo "============================================================"
