import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torch
import csv
import random

random.seed(42)

# PIQ similarity metrics
from piq import (
    ssim, fsim, multi_scale_ssim, information_weighted_ssim,
    srsim, vsi, dss, haarpsi, mdsi
)

# ============================================================
#   1. CORE: TID2013 degradation functions
# ============================================================

def gaussian_noise(img, level):
    sigma = [5, 10, 20, 40, 80][level]
    noise = np.random.normal(0, sigma, img.shape)
    out = np.clip(img + noise, 0, 255).astype(np.uint8)
    return out

def additive_color_noise(img, level):
    sigma = [5, 10, 20, 40, 80][level]
    noise = np.random.normal(0, sigma, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)

def spatially_correlated_noise(img, level):
    sigma = [5, 10, 20, 40, 80][level]
    noise = cv2.GaussianBlur(
        np.random.normal(0, sigma, img.shape).astype(np.float32),
        (7,7), sigmaX=3
    )
    return np.clip(img + noise, 0, 255).astype(np.uint8)

def masked_noise(img, level):
    prob = [0.02, 0.05, 0.1, 0.15, 0.25][level]
    mask = np.random.rand(*img.shape[:2]) < prob
    out = img.copy()
    out[mask] = 0
    return out

def high_freq_noise(img, level):
    sigma = [5,10,20,40,80][level]
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noise = noise - cv2.GaussianBlur(noise, (13,13), 5)
    return np.clip(img + noise, 0,255).astype(np.uint8)

def impulse_noise(img, level):
    prob = [0.01,0.03,0.06,0.10,0.15][level]
    out = img.copy()
    mask = np.random.rand(*img.shape[:2]) < prob
    values = np.random.choice([0,255], size=(mask.sum(), 3))
    out[mask] = values
    return out

def quantization_noise(img, level):
    q = [64,32,16,8,4][level]
    return ((img // q) * q).astype(np.uint8)

def gaussian_blur(img, level):
    radius = [1,2,3,5,8][level]
    return cv2.GaussianBlur(img, (0,0), radius)

def brightness_shift(img, level):
    shift = [-40, -20, -10, 20, 40][level]
    img16 = img.astype(np.int16)
    return np.clip(img16 + shift, 0,255).astype(np.uint8)

def contrast_change(img, level):
    factors = [0.5,0.7,0.9,1.3,1.6][level]
    mean = img.mean()
    out = (img - mean) * factors + mean
    return np.clip(out, 0,255).astype(np.uint8)

def saturation_change(img, level):
    factors = [0.3,0.6,0.8,1.2,1.6]
    factor = factors[level]
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[...,1] *= factor
    hsv[...,1] = np.clip(hsv[...,1], 0,255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

def pixelation(img, level):
    s = [0.8,0.6,0.4,0.3,0.2][level]
    h,w = img.shape[:2]
    small = cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w,h), interpolation=cv2.INTER_NEAREST)

def color_quantization(img, level):
    q = [64,32,16,8,4][level]
    return ((img // q) * q).astype(np.uint8)


# List of distortions
DISTORTIONS = [
    gaussian_noise, additive_color_noise, spatially_correlated_noise,
    masked_noise, high_freq_noise, impulse_noise, quantization_noise,
    gaussian_blur, brightness_shift, contrast_change,
    saturation_change, pixelation, color_quantization
]


# ============================================================
#   2. MAIN LOGIC — compute similarity metrics
# ============================================================

def compute_metrics(img_ref, img_deg):
    """Compute all PIQ similarity metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ref_t = torch.tensor(img_ref).permute(2,0,1).unsqueeze(0).float().to(device) / 255
    deg_t = torch.tensor(img_deg).permute(2,0,1).unsqueeze(0).float().to(device) / 255

    return {
        "ssim": ssim(ref_t, deg_t, data_range=1.0).item(),
        "fsim": fsim(ref_t, deg_t, data_range=1.0).item(),
        "ms_ssim": multi_scale_ssim(ref_t, deg_t, data_range=1.0).item(),
        "iw_ssim": information_weighted_ssim(ref_t, deg_t, data_range=1.0).item(),
        "sr_sim": srsim(ref_t, deg_t, data_range=1.0).item(),
        "vsi": vsi(ref_t, deg_t, data_range=1.0).item(),
        "dss": dss(ref_t, deg_t, data_range=1.0).item(),
        "haarpsi": haarpsi(ref_t, deg_t, data_range=1.0).item(),
        "mdsi": mdsi(ref_t, deg_t, data_range=1.0).item(),
    }


# ============================================================
#   3. GENERATE DATASET + METRICS
# ============================================================

def generate_tid_dataset(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_path = input_dir/"valid_paths.txt"
    with open(cache_path, "r") as f:
        valid_paths = [Path(x.strip()) for x in f.readlines() if x.strip()]

    img_paths = [p for p in valid_paths if p.suffix.lower() in [".png",".jpeg",".bmp"]]
    

    random.shuffle(img_paths)
    img_paths = img_paths[:200]

    scores = []

    for img_path in tqdm(img_paths, desc="Processing images"):
        img_ref = np.array(Image.open(img_path).convert("RGB"))

        for d_idx, distortion in enumerate(DISTORTIONS):
            for level in range(5):
                img_deg = distortion(img_ref, level)

                filename = f"{img_path.stem}_D{d_idx+1}_L{level+1}.png"
                Image.fromarray(img_deg).save(output_dir/filename)

                metrics = compute_metrics(img_ref, img_deg)
                metrics["filename"] = filename
                scores.append(metrics)

    # Write scores.csv
    csv_path = output_dir/"scores.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename","ssim","fsim","ms_ssim","iw_ssim",
                        "sr_sim","vsi","dss","haarpsi","mdsi"]
        )
        writer.writeheader()
        writer.writerows(scores)

    print("\nDONE! Saved:", csv_path)


# ============================================================
#   4. RUN SCRIPT
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)

    args = parser.parse_args()
    generate_tid_dataset(args.input_dir, args.output_dir)