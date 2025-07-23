#!/bin/bash

# Usage check
if [ "$#" -lt 1 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 [ssim|fsim] [min_score (e.g. 0.0)] [max_score (e.g. 1.0)]"
    exit 1
fi

TARGET="$1"
MIN_SCORE="${2:-0.0}"  # Default to 0.0
MAX_SCORE="${3:-1.0}"  # Default to 1.0

# Clean numeric formatting for session name
MIN_SCORE_CLEAN=$(printf "%.2f" "$MIN_SCORE" | tr -d '.')
MAX_SCORE_CLEAN=$(printf "%.2f" "$MAX_SCORE" | tr -d '.')

SESSION_NAME="training_${TARGET}_${MIN_SCORE_CLEAN}${MAX_SCORE_CLEAN}"

# Create detached tmux session
tmux new-session -d -s "${SESSION_NAME}"

# Training commands with both min and max score
declare -a commands=(
  "source .venv/bin/activate && CUDA_VISIBLE_DEVICES=0 python train.py --model mobilevitv2 --train_path /home/jovyan/nfs/datasets/ILSVRC2012_degraded/train --val_path /home/jovyan/nfs/datasets/ILSVRC2012_degraded/val --epochs 50 --batch_size 32 --image_size 384 --lr 1e-4 --target ${TARGET} --min_score ${MIN_SCORE} --max_score ${MAX_SCORE}"
  "source .venv/bin/activate && CUDA_VISIBLE_DEVICES=1 python train.py --model resnet50v2 --train_path /home/jovyan/nfs/datasets/ILSVRC2012_degraded/train --val_path /home/jovyan/nfs/datasets/ILSVRC2012_degraded/val --epochs 50 --batch_size 32 --image_size 384 --lr 1e-4 --target ${TARGET} --min_score ${MIN_SCORE} --max_score ${MAX_SCORE}"
  "source .venv/bin/activate && CUDA_VISIBLE_DEVICES=2 python train.py --model vit --train_path /home/jovyan/nfs/datasets/ILSVRC2012_degraded/train --val_path /home/jovyan/nfs/datasets/ILSVRC2012_degraded/val --epochs 50 --batch_size 32 --image_size 384 --lr 1e-4 --target ${TARGET} --min_score ${MIN_SCORE} --max_score ${MAX_SCORE}"
  "source .venv/bin/activate && CUDA_VISIBLE_DEVICES=3 python train.py --model efficientnet --train_path /home/jovyan/nfs/datasets/ILSVRC2012_degraded/train --val_path /home/jovyan/nfs/datasets/ILSVRC2012_degraded/val --epochs 50 --batch_size 32 --image_size 380 --lr 1e-4 --target ${TARGET} --min_score ${MIN_SCORE} --max_score ${MAX_SCORE}"
  "source .venv/bin/activate && CUDA_VISIBLE_DEVICES=4 python train.py --model mobilenetv3 --train_path /home/jovyan/nfs/datasets/ILSVRC2012_degraded/train --val_path /home/jovyan/nfs/datasets/ILSVRC2012_degraded/val --epochs 50 --batch_size 32 --image_size 384 --lr 1e-4 --target ${TARGET} --min_score ${MIN_SCORE} --max_score ${MAX_SCORE}"
  "source .venv/bin/activate && CUDA_VISIBLE_DEVICES=5 python train.py --model tinyvit --train_path /home/jovyan/nfs/datasets/ILSVRC2012_degraded/train --val_path /home/jovyan/nfs/datasets/ILSVRC2012_degraded/val --epochs 50 --batch_size 32 --image_size 384 --lr 1e-4 --target ${TARGET} --min_score ${MIN_SCORE} --max_score ${MAX_SCORE}"
)

# Launch commands in tmux panes
for i in "${!commands[@]}"; do
  if [ "$i" -eq 0 ]; then
    tmux send-keys -t "${SESSION_NAME}" "${commands[$i]}" C-m
  else
    tmux split-window -t "${SESSION_NAME}" -h
    tmux send-keys -t "${SESSION_NAME}" "${commands[$i]}" C-m
  fi
  tmux select-layout -t "${SESSION_NAME}" tiled
done

# Monitor pane to auto-kill session
tmux split-window -t "${SESSION_NAME}" -v
tmux send-keys -t "${SESSION_NAME}" "while pgrep -f train.py > /dev/null; do sleep 10; done; tmux kill-session -t ${SESSION_NAME}" C-m
tmux select-layout -t "${SESSION_NAME}" tiled

# Attach to session
tmux attach -t "${SESSION_NAME}"