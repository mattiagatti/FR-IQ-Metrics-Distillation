#!/usr/bin/env bash
set -euo pipefail

# ================================================================
# Script to launch sequential tests for all image quality metrics
# and all models.
# Logs to stdout only (no files).
# Example usage:
#   ./run_all_metrics_test.sh 0     # Use GPU 0
# ================================================================

# -------------------------------
# User configuration
# -------------------------------
TEST_SCRIPT="./test.py"   # Path to your test script

# Models to test
MODELS=(mobilevitv2 resnet50v2 vit efficientnet mobilenetv3 tinyvit)

# List of all supported metrics
METRICS=(ssim fsim ms_ssim iw_ssim sr_sim vsi dss haarpsi mdsi)

CONFIG_PATH="./experiments/test_IMAGENETour_IMAGENETour.yaml"  # Config file path
ALSO_MOS=false                           # Whether to also test with MOS

# -------------------- GPU SELECTION -------------------- #

if [ $# -lt 1 ]; then
  echo "Usage: $0 <GPU_ID>"
  echo "Example: $0 0   # use GPU 0"
  exit 1
fi

GPU_ID=$1
export CUDA_VISIBLE_DEVICES=${GPU_ID}

echo "==============================================================="
echo "Running sequential tests on GPU ${GPU_ID}"
echo "Models: ${MODELS[*]}"
echo "Metrics: ${METRICS[*]}"
echo "==============================================================="

# -------------------- TEST LOOP -------------------- #
for MODEL in "${MODELS[@]}"; do
  echo "=============================================================="
  echo "=== Starting tests for model: ${MODEL} ==="
  echo "=============================================================="

  for METRIC in "${METRICS[@]}"; do
    echo "============================================================"
    echo "Starting evaluation for ${MODEL} - metric: ${METRIC}"
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
    "${CMD[@]}"

    echo "Finished evaluation for ${MODEL} - ${METRIC}."

    # If ALSO_MOS is true, run again with MOS
    if [ "${ALSO_MOS}" = true ]; then
      echo "Starting evaluation for ${MODEL} - metric: ${METRIC} with MOS"
      CMD=(python3 "${TEST_SCRIPT}" \
        --model "${MODEL}" \
        --metric "${METRIC}" \
        --config_path "${CONFIG_PATH}" \
        --mos true)

      echo "▶️  Running command:"
      echo "${CMD[@]}"
      echo "------------------------------------------------------------"

      # Run and print output to stdout
      "${CMD[@]}"

      echo "Finished evaluation for ${MODEL} - ${METRIC} with MOS."
    fi

    echo
  done

  echo "=== Completed all metrics for model: ${MODEL} ==="
  echo
done

echo "============================================================"
echo "All evaluations completed successfully on GPU ${GPU_ID}."
echo "Total combinations: ${#MODELS[@]} models x ${#METRICS[@]} metrics = $((${#MODELS[@]} * ${#METRICS[@]})) tests"
echo "============================================================"