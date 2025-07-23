import random
import numpy as np
import torch
import cv2
import tempfile
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from piq import fsim
import os

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Nimbus Roman'],
    'text.usetex': False,
    'font.size': 22
})

# ------------------- Config ------------------- #
imagenet_path = Path("/home/jovyan/nfs/datasets/ILSVRC2012/train")
cache_file = Path(tempfile.gettempdir()) / "imagenet_image_paths.txt"
target_fsims = [0.9, 0.8, 0.7]
tolerance = 0.01
max_trials = 100
num_images = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 45

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ------------------- Degradation Logic ------------------- #
degradation_types = [
    'gaussian_blur', 'motion_blur', 'jpeg_compression', 'brightness', 'contrast',
    'saturation', 'additive_noise', 'chromatic_aberration', 'pixelation', 'color_cast'
]


def get_degradation_params():
    max_n = len(degradation_types)
    num_degradations = random.randint(1, max_n)
    selected = random.sample(degradation_types, num_degradations)
    params = {}
    for degradation in selected:
        if degradation == 'gaussian_blur':
            params[degradation] = random.uniform(0.5, 1.5)
        elif degradation == 'motion_blur':
            params[degradation] = random.randint(3, 7)
        elif degradation == 'jpeg_compression':
            params[degradation] = random.randint(40, 80)
        elif degradation == 'brightness':
            params[degradation] = random.uniform(0.8, 1.2)
        elif degradation == 'contrast':
            params[degradation] = random.uniform(0.8, 1.2)
        elif degradation == 'saturation':
            params[degradation] = random.uniform(0.7, 1.0)
        elif degradation == 'additive_noise':
            params[degradation] = {
                'type': random.choice(['gaussian', 'salt_pepper']),
                'amount': random.uniform(0.01, 0.03)
            }
        elif degradation == 'chromatic_aberration':
            params[degradation] = {
                'shift': random.randint(1, 3),
                'channel': random.randint(0, 2)
            }
        elif degradation == 'pixelation':
            params[degradation] = random.uniform(0.9, 0.98)
        elif degradation == 'color_cast':
            params[degradation] = {
                'channel': random.randint(0, 2),
                'factor': random.uniform(1.05, 1.15)
            }
    return params


def apply_degradation(img, params):
    degraded = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    h, w = degraded.shape[:2]
    for degradation, param in params.items():
        if degradation == 'gaussian_blur':
            ksize = max(3, int(param) | 1)
            degraded = cv2.GaussianBlur(degraded, (ksize, ksize), sigmaX=param)
        elif degradation == 'motion_blur':
            k = param
            kernel = np.zeros((k, k))
            kernel[k // 2, :] = np.ones(k)
            kernel /= k
            degraded = cv2.filter2D(degraded, -1, kernel)
        elif degradation == 'jpeg_compression':
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), param]
            _, enc = cv2.imencode('.jpg', degraded, encode_param)
            degraded = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        elif degradation == 'brightness':
            degraded = cv2.convertScaleAbs(degraded, alpha=param, beta=0)
        elif degradation == 'contrast':
            mean = np.mean(degraded)
            degraded = cv2.convertScaleAbs(degraded, alpha=param, beta=(1 - param) * mean)
        elif degradation == 'saturation':
            hsv = cv2.cvtColor(degraded, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[..., 1] *= param
            hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
            degraded = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        elif degradation == 'additive_noise':
            if param['type'] == 'gaussian':
                noise = np.random.normal(0, param['amount'] * 255, degraded.shape).astype(np.float32)
                degraded = np.clip(degraded.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            elif param['type'] == 'salt_pepper':
                coords = np.where(np.random.rand(h, w) < param['amount'])
                for c in range(3):
                    degraded[coords[0], coords[1], c] = np.random.choice([0, 255], size=len(coords[0]))
        elif degradation == 'chromatic_aberration':
            shift = param['shift']
            channel = param['channel']
            axis = random.choice([0, 1])
            degraded[..., channel] = np.roll(degraded[..., channel], shift, axis=axis)
        elif degradation == 'pixelation':
            new_w = max(1, int(w * param))
            new_h = max(1, int(h * param))
            degraded = cv2.resize(degraded, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            degraded = cv2.resize(degraded, (w, h), interpolation=cv2.INTER_NEAREST)
        elif degradation == 'color_cast':
            channel, factor = param['channel'], param['factor']
            degraded[..., channel] = np.clip(degraded[..., channel].astype(np.float32) * factor, 0, 255).astype(np.uint8)
    degraded_rgb = cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB)
    return Image.fromarray(degraded_rgb)


def fsim_between(img1, img2):
    t1 = torch.tensor(np.array(img1)).permute(2, 0, 1).unsqueeze(0).float() / 255
    t2 = torch.tensor(np.array(img2)).permute(2, 0, 1).unsqueeze(0).float() / 255
    return fsim(t1.to(device), t2.to(device), data_range=1.0).item()


def progressively_degrade(img, targets, tolerance, max_trials, log_prefix="Image"):
    results = [img]
    base = img
    current = img

    for target_idx, target in enumerate(targets):
        print(f"{log_prefix} - Target FSIM: {target:.2f}")
        
        for trial in range(1, max_trials + 1):
            params = get_degradation_params()
            degraded = apply_degradation(current, params)
            score = fsim_between(base, degraded)

            print(f"  Trial {trial:03d}: FSIM = {score:.4f} | Degradations: {params}")

            current = degraded

            if abs(score - target) <= tolerance:
                print(f"  Reached FSIM={score:.4f} at trial {trial}")
                break

            if score < target - tolerance:
                print(f"  FSIM dropped below target ({score:.4f} < {target}), stopping further degradation")
                break

        results.append(current)

    return results


def progressively_degrade(img, targets, tolerance, max_trials, log_prefix="Image"):
    results = [img]
    base = img
    current = img

    for target_idx, target in enumerate(targets):
        print(f"{log_prefix} - Target FSIM: {target:.2f}")

        trial = 0
        best_img = current
        best_score = fsim_between(base, current)

        while trial < max_trials:
            trial += 1
            params = get_degradation_params()
            degraded = apply_degradation(current, params)
            score = fsim_between(base, degraded)

            print(f"  Trial {trial:03d}: FSIM = {score:.4f} | Degradations: {params}")

            if abs(score - target) <= tolerance:
                print(f"  Reached FSIM={score:.4f} at trial {trial}")
                current = degraded
                break

            if score > target:
                # FSIM is still too high: continue degrading this version
                current = degraded
            else:
                # FSIM dropped below target: revert to best known good one
                print(f"  FSIM {score:.4f} below target ({target}), reverting to previous image")
                current = best_img
                continue

            best_img = current
            best_score = score

        results.append(current)

    return results


# ------------------- Execution ------------------- #
if cache_file.exists():
    print(f"Loading image paths from cache: {cache_file}")
    with open(cache_file, "r") as f:
        all_images = [Path(line.strip()) for line in f]
else:
    print(f"Scanning ImageNet directory... This may take a while.")
    all_images = list(imagenet_path.rglob("*.JPEG"))

    # Save to cache
    with open(cache_file, "w") as f:
        for path in all_images:
            f.write(str(path) + "\n")
    print(f"Saved image paths to cache: {cache_file}")

# selected_images = random.sample(all_images, num_images)

picks = ["n02859443/n02859443_12212.JPEG", "n02108000/n02108000_1843.JPEG", "n04252225/n04252225_28945.JPEG"]
selected_images = [imagenet_path / x for x in picks]

fig, axes = plt.subplots(nrows=num_images, ncols=len(target_fsims) + 1, figsize=(18, 4 * num_images))

for row, img_path in enumerate(selected_images):
    img = Image.open(img_path).convert("RGB").resize((384, 384))
    degraded_imgs = progressively_degrade(img, target_fsims, tolerance, max_trials)

    image_id = img_path.stem  # e.g., "n02859443_12212"
    save_dir = Path("plots/samples/images") / image_id
    save_dir.mkdir(parents=True, exist_ok=True)

    for col, d_img in enumerate(degraded_imgs):
        title = "Original" if col == 0 else f"FSIM = {target_fsims[col - 1]}"
        axes[row, col].imshow(d_img)
        axes[row, col].set_title(title)
        axes[row, col].axis("off")

        # Save each image to disk
        safe_title = title.lower().replace("=", "").replace(" ", "").replace(".", "_")
        image_filename = save_dir / f"{safe_title}.png"
        d_img.save(image_filename)

output_path = Path("plots/samples/degraded_examples.pdf")
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0)
plt.show()

print(f"Saved figure to {output_path.resolve()}")