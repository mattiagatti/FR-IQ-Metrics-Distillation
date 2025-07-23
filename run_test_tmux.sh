#!/bin/bash

# Usage check
if [ $# -lt 2 ]; then
  echo "Usage: $0 <test_path> <ssim|fsim>"
  exit 1
fi

TEST_PATH="$1"
TARGET="$2"
FINETUNE=false

# Check for optional --finetune flag
if [ "$3" == "--finetune" ]; then
  FINETUNE=true
fi

SESSION_NAME="testing_${TARGET}"

# Create a new detached tmux session
tmux new-session -d -s $SESSION_NAME

# Define testing commands with dynamic checkpoint paths
declare -a commands=(
  "source .venv/bin/activate && CUDA_VISIBLE_DEVICES=0 python test.py --model mobilevitv2 --checkpoint exp/mobilevitv2_${TARGET}/best.pth --test_path $TEST_PATH --batch_size 32 --image_size 384 --target ${TARGET} --max_samples_per_bin 50"
  "source .venv/bin/activate && CUDA_VISIBLE_DEVICES=1 python test.py --model resnet50v2 --checkpoint exp/resnet50v2_${TARGET}/best.pth --test_path $TEST_PATH --batch_size 32 --image_size 384 --target ${TARGET} --max_samples_per_bin 50"
  "source .venv/bin/activate && CUDA_VISIBLE_DEVICES=2 python test.py --model vit --checkpoint exp/vit_${TARGET}/best.pth --test_path $TEST_PATH --batch_size 32 --image_size 384 --target ${TARGET} --max_samples_per_bin 50"
  "source .venv/bin/activate && CUDA_VISIBLE_DEVICES=3 python test.py --model efficientnet --checkpoint exp/efficientnet_${TARGET}/best.pth --test_path $TEST_PATH --batch_size 32 --image_size 380 --target ${TARGET} --max_samples_per_bin 50"
  "source .venv/bin/activate && CUDA_VISIBLE_DEVICES=4 python test.py --model mobilenetv3 --checkpoint exp/mobilenetv3_${TARGET}/best.pth --test_path $TEST_PATH --batch_size 32 --image_size 384 --target ${TARGET} --max_samples_per_bin 50"
  "source .venv/bin/activate && CUDA_VISIBLE_DEVICES=5 python test.py --model tinyvit --checkpoint exp/tinyvit_${TARGET}/best.pth --test_path $TEST_PATH --batch_size 32 --image_size 384 --target ${TARGET} --max_samples_per_bin 50"
)

# Launch each command in tmux
for i in "${!commands[@]}"; do
  if [ "$i" -eq 0 ]; then
    tmux send-keys -t $SESSION_NAME "${commands[$i]}" C-m
  else
    tmux split-window -t $SESSION_NAME -h
    tmux select-layout -t $SESSION_NAME tiled
    tmux send-keys -t $SESSION_NAME "${commands[$i]}" C-m
  fi
done

# Attach to the session to monitor testing
tmux attach -t $SESSION_NAME