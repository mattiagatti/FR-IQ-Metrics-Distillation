# SSIM Regression on ImageNet

Train deep learning models to **predict SSIM scores** from images using various popular vision backbones.

---

## Requirements

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Script Overview

This script trains a regression model on images and their SSIM ground truth values.

### Command-Line Arguments

| Argument                  | Description                                                                                                  |
| ------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `--model`                 | Model to use (`mobilevit`, `resnet50v2`, `convnext`, `vit`, `efficientnet`, `mobilenetv3`, `tinyvit`) |
| `--train_path`            | Path to training dataset directory                                                                           |
| `--val_split`             | Fraction of training data to use for validation (default: 0.20)                                              |
| `--epochs`                | Number of training epochs                                                                                    |
| `--batch_size`            | Batch size                                                                                                   |
| `--image_size`            | Image size (should be 384 for all models except EfficientNet, which uses 380)                                |
| `--lr`                    | Learning rate                                                                                                |
| `--finetune`              | Enable fine-tuning from a pretrained checkpoint                                                              |
| `--pretrained_checkpoint` | Path to pretrained model checkpoint (default: `./exp/<model>/<model>_best.pth`)                              |
| `--target_per_bin`        | Number of samples to draw from each SSIM score bin during training                                           |


---

## Train From Scratch – Examples for All Models

### MobileViT-v2

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --model mobilevitv2 --train_path /home/jovyan/nfs/datasets/ILSVRC2012_degraded/train --epochs 50 --batch_size 32 --image_size 384 --lr 1e-4 --target_per_bin 100
```

### ResNet50v2

```bash
CUDA_VISIBLE_DEVICES=1 python train.py --model resnet50v2 --train_path /home/jovyan/nfs/datasets/ILSVRC2012_degraded/train --val_path /home/jovyan/nfs/datasets/ILSVRC2012_degraded/val --epochs 50 --batch_size 32 --image_size 384 --lr 1e-4 --target_per_bin 100
```

### ViT

```bash
CUDA_VISIBLE_DEVICES=2 python train.py --model vit --train_path /home/jovyan/nfs/datasets/ILSVRC2012_degraded/train --val_path /home/jovyan/nfs/datasets/ILSVRC2012_degraded/val --epochs 50 --batch_size 32 --image_size 384 --lr 1e-4 --target_per_bin 100
```

### EfficientNet-B4

```bash
CUDA_VISIBLE_DEVICES=3 python train.py --model efficientnet --train_path /home/jovyan/nfs/datasets/ILSVRC2012_degraded/train --val_path /home/jovyan/nfs/datasets/ILSVRC2012_degraded/val --epochs 50 --batch_size 32 --image_size 380 --lr 1e-4 --target_per_bin 100
```

### MobileNetV3

```bash
CUDA_VISIBLE_DEVICES=4 python train.py --model mobilenetv3 --train_path /home/jovyan/nfs/datasets/ILSVRC2012_degraded/train --val_path /home/jovyan/nfs/datasets/ILSVRC2012_degraded/val --epochs 50 --batch_size 32 --image_size 384 --lr 1e-4 --target_per_bin 100
```

### TinyViT

```bash
CUDA_VISIBLE_DEVICES=5 python train.py --model tinyvit --train_path /home/jovyan/nfs/datasets/ILSVRC2012_degraded/train --val_path /home/jovyan/nfs/datasets/ILSVRC2012_degraded/val --epochs 50 --batch_size 32 --image_size 384 --lr 1e-4 --target_per_bin 100
```

---

## Fine-Tune from Pretrained Checkpoint – Examples for All Models

### MobileViT-v2

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --model mobilevitv2 --train_path /home/jovyan/nfs/datasets/DermIQ_degraded/train --epochs 20 --batch_size 32 --image_size 384 --lr 1e-5 --finetune --pretrained_checkpoint ./exp/mobilevitv2/best.pth
```

### ResNet50v2

```bash
CUDA_VISIBLE_DEVICES=1 python train.py --model resnet50v2 --train_path /home/jovyan/nfs/datasets/DermIQ_degraded/train --epochs 20 --batch_size 32 --image_size 384 --lr 1e-5 --finetune --pretrained_checkpoint ./exp/resnet50v2/best.pth
```

### ViT

```bash
CUDA_VISIBLE_DEVICES=2 python train.py --model vit --train_path /home/jovyan/nfs/datasets/DermIQ_degraded/train --epochs 20 --batch_size 32 --image_size 384 --lr 1e-5 --finetune --pretrained_checkpoint ./exp/vit/best.pth
```

### EfficientNet-B4

```bash
CUDA_VISIBLE_DEVICES=3 python train.py --model efficientnet --train_path /home/jovyan/nfs/datasets/DermIQ_degraded/train --epochs 20 --batch_size 32 --image_size 380 --lr 1e-5 --finetune --pretrained_checkpoint ./exp/efficientnet/best.pth
```

### MobileNetV3

```bash
CUDA_VISIBLE_DEVICES=4 python train.py --model mobilenetv3 --train_path /home/jovyan/nfs/datasets/DermIQ_degraded/train --epochs 20 --batch_size 32 --image_size 384 --lr 1e-5 --finetune --pretrained_checkpoint ./exp/mobilenetv3/mobilenetv3_best.pth
```

### TinyViT

```bash
CUDA_VISIBLE_DEVICES=5 python train.py --model tinyvit --train_path /home/jovyan/nfs/datasets/DermIQ_degraded/train --epochs 20 --batch_size 32 --image_size 384 --lr 1e-5 --finetune --pretrained_checkpoint ./exp/tinyvit/best.pth
```

---

## Test – Examples for All Models

Assumes checkpoint is saved as `best.pth` in `exp/<model_name>/`.

### MobileViT-v2

```bash
CUDA_VISIBLE_DEVICES=0 python test.py --model mobilevitv2 --checkpoint exp/mobilevitv2/best.pth --test_path /home/jovyan/nfs/datasets/ILSVRC2012_degraded/test --batch_size 32 --image_size 384
```

### ResNet50v2

```bash
CUDA_VISIBLE_DEVICES=1 python test.py --model resnet50v2 --checkpoint exp/resnet50v2/best.pth --test_path /home/jovyan/nfs/datasets/ILSVRC2012_degraded/test --batch_size 32 --image_size 384
```

### ViT

```bash
CUDA_VISIBLE_DEVICES=2 python test.py --model vit --checkpoint exp/vit/best.pth --test_path /home/jovyan/nfs/datasets/ILSVRC2012_degraded/test --batch_size 32 --image_size 384
```

### EfficientNet-B3

```bash
CUDA_VISIBLE_DEVICES=3 python test.py --model efficientnet --checkpoint exp/efficientnet/best.pth --test_path /home/jovyan/nfs/datasets/ILSVRC2012_degraded/test --batch_size 32 --image_size 380
```

### MobileNetV3

```bash
CUDA_VISIBLE_DEVICES=4 python train.py --model mobilenetv3 --train_path /path/to/dataset --epochs 20 --batch_size 32 --image_size 384 --lr 1e-5 --finetune --pretrained_checkpoint ./exp/mobilenetv3/mobilenetv3_best.pth
```

### TinyViT

```bash
CUDA_VISIBLE_DEVICES=5 python test.py --model tinyvit --checkpoint exp/tinyvit/best.pth --test_path /home/jovyan/nfs/datasets/ILSVRC2012_degraded/test --batch_size 32 --image_size 384
```

---

## Train & Test in parallel

This repository includes scripts to start a tmux session and run multiple training or testing jobs in parallel across different models. Each script automates environment activation, GPU assignment, and dataset handling. Simply pass the appropriate dataset path (when required) as a command-line argument.

### Examples

To train on **ILSVRC2012 degraded train set**:
```bash
./run_train_tmux.sh ssim
./run_train_tmux.sh fsim
```

To test on **ILSVRC2012 degraded test set**:
```bash
./run_test_tmux.sh /home/jovyan/nfs/datasets/ILSVRC2012_degraded/test ssim
./run_test_tmux.sh /home/jovyan/nfs/datasets/ILSVRC2012_degraded/test fsim
```

To finetune on **DermIQ degraded train set**:
```bash
./run_finetune_tmux.sh
```

To test on **DermIQ degraded test set**:
```bash
./run_test_tmux.sh /home/jovyan/nfs/datasets/DermIQ_degraded/test ssim
./run_test_tmux.sh /home/jovyan/nfs/datasets/DermIQ_degraded/test fsim
```

To train on **ILSVRC2012 degraded train set in different ranges**:
```bash
./run_train_tmux.sh ssim 0.4
./run_train_tmux.sh fsim 0.4
./run_train_tmux.sh ssim 0.7
./run_train_tmux.sh fsim 0.7
./run_train_tmux.sh ssim 0.4 0.7
./run_train_tmux.sh fsim 0.4 0.7
```

Then:
```bash
tmux kill-session -t session_name
```


---

## Output Files

Saved in `exp/<model>[_finetune]/`:

* Best model checkpoint
* Last checkpoint
* Hexbin and scatter plots

---

## Metrics

The script tracks:

* R2 Score
* MAE (Mean Absolute Error)
* MSE (Mean Squared Error)
* RMSE (Root Mean Squared Error)