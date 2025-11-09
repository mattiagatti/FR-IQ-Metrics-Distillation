#!/usr/bin/env bash
set -euo pipefail

# Script per lanciare training sequenziali su più metriche.
# Modifica le variabili qui sotto: train_path e val_path sono obbligatori.
# Esempio di uso:
#   ./run_all_metrics.sh

# CONFIGURAZIONE (modifica)
MODEL="tinyvit"
TRAIN_PATH="/home/jovyan/nfs/datasets/ILSVRC2012_degraded/train"   # es. /Users/.../train
VAL_PATH="/home/jovyan/nfs/datasets/ILSVRC2012_degraded/val"       # se non vuoi validation esterna lascia vuoto ""
EPOCHS=50
BATCH_SIZE=32
IMAGE_SIZE=384
LR=1e-4
TRAIN_TARGET_PER_BIN=100
VAL_TARGET_PER_BIN=100
MIN_SCORE=0.0
MAX_SCORE=1.0

# Numero di workers/ogg per DataLoader -> lascia come nel file python
# (non necessario qui)

# Lista metriche (presa da generate_dataset.py)
METRICS=(ssim fsim ms_ssim iw_ssim sr_sim vsi dss haarpsi mdsi)

# Path allo script di training (modifica se necessario)
TRAIN_PY="/home/jovyan/python/Paper_SSIM_FSIM/Neural-No-Reference-SIM/train.py"
PYTHON_BIN="python3"

# Directory dove salvare log (crea se non esiste)
LOG_DIR="exp/logs_all_metrics"
mkdir -p "${LOG_DIR}"

for M in "${METRICS[@]}"; do
  echo "=== Avvio training per: ${M} ==="
  EXP_NAME="${M}"
  LOG_FILE="${LOG_DIR}/${EXP_NAME}_train.log"

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
      2>&1 | tee "${LOG_FILE}"
  fi

  echo "=== Fine training per: ${M} (log: ${LOG_FILE}) ==="
  echo
done

echo "Tutti i training completati."