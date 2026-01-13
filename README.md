# Distilling Full-Reference Image Quality Metrics into Neural No-Reference Surrogates

Train deep learning models to **predict FR metrics** from images using various popular vision backbones.

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

### Dataset generation script

```bash
python misc/generate_dataset.py --min-degradations <value> --max-degradations <value> --input_path /path/to/imagenet 
```
Generates degraded datasets from the original ImageNet dataset located at `/path/to/imagenet`. The degraded datasets will be saved in `/path/to/imagenet_degraded/` with `train`, `val`, and `test` subdirectories.
min-degradations and max-degradations specify the range of degradations to apply to each image.

### Command-Line Arguments

| Argument                  | Description                                                                                                  |
| ------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `--model`                 | Model to use (`mobilevit`, `resnet50v2`, `convnext`, `vit`, `efficientnet`, `mobilenetv3`, `tinyvit`) |
| `--metric`            | Path to training dataset directory                                                                           |
| `--config_path`          | Path to the configuration file.                                              |


---

## Train From Scratch – Examples for All Models

### MobileViT-v2

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --model mobilevitv2 --metric SSIM --config_path train_IMAGENET_our.yaml 
```

### ResNet50v2

```bash
CUDA_VISIBLE_DEVICES=1 python train.py --model resnet50v2 --metric SSIM --config_path train_IMAGENET_our.yaml 
```

### ViT

```bash
CUDA_VISIBLE_DEVICES=2 python train.py --model vit --metric SSIM --config_path train_IMAGENET_our.yaml 
```

### EfficientNet-B4

```bash
CUDA_VISIBLE_DEVICES=3 python train.py --model efficientnet --metric SSIM --config_path train_IMAGENET_our.yaml 
```

### MobileNetV3

```bash
CUDA_VISIBLE_DEVICES=4 python train.py --model mobilenetv3 --metric SSIM --config_path train_IMAGENET_our.yaml
```

### TinyViT

```bash
CUDA_VISIBLE_DEVICES=5 python train.py --model tinyvit --metric SSIM --config_path train_IMAGENET_our.yaml
```

--------

## testing
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --model <model_name> --metric SSIM --config_path test_IMAGENETour_IMAGENETour.yaml
```


## Metrics

The script tracks:

* R2 Score
* MAE (Mean Absolute Error)
* Spearman Correlation (SROCC)
* Pearson Correlation (PLCC)
