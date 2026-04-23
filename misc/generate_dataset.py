import cv2
import random
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from piq import ssim, fsim, multi_scale_ssim, information_weighted_ssim, vif_p, srsim, gmsd, multi_scale_gmsd, vsi, dss, haarpsi, mdsi
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import argparse
from collections import defaultdict

#python3 misc/generate_dataset.py --root-dir /tmp --input-txt train.txt --output-dir /home/jovyan/nfs/lsgroi/Datasets/Medical_degraded/train --min-degradations 1 --max-degradations 5 --num-threads 16

mappingPath= {
    "DERM7pt-clinic": Path("/home/jovyan/nfs/igallo/datasets/OOD/clinic/DERM7pt-clinic"),
    "DERM7pt-derm": Path("/home/jovyan/nfs/igallo/datasets/OOD/dermatoscopic/DERM7pt-clinic"),
    "MCR-SL": Path("/home/jovyan/nfs/igallo/datasets/OOD/clinic+derm/MCR-SL"),
    "MRA-MIDAS": Path("/home/jovyan/nfs/igallo/datasets/OOD/clinic+derm/MRA-MIDAS"),
    "PAD-UFES-20": Path("/home/jovyan/nfs/igallo/datasets/OOD/clinic/PAD-UFES-20"),
    "bcn20k": Path("/home/jovyan/nfs/igallo/datasets/OOD/dermatoscopic/bcn20k"),
    "ham10k": Path("/home/jovyan/nfs/igallo/datasets/OOD/dermatoscopic/ham10k"),
    "derm12345": Path("/home/jovyan/nfs/igallo/datasets/OOD/dermatoscopic/derm12345"),
    "isic_archive_d3": Path("/home/jovyan/nfs/igallo/datasets/OOD/dermatoscopic/isic_archive_d3"),
    "isic_archive_d4": Path("/home/jovyan/nfs/igallo/datasets/OOD/dermatoscopic/isic_archive_d4"),
}


class Degrader:
    def __init__(self, root_dir, paths_txt, degradation_types=None, min_resolution=(384, 384), min_degradations=None, max_degradations=None):
        self.root_dir = Path(root_dir)
        self.paths_txt = Path(paths_txt)
        self.min_degradations = min_degradations
        self.max_degradations = max_degradations

        if not self.paths_txt.exists():
            raise FileNotFoundError(f"Input txt file not found: {self.paths_txt}")

        def _extract_path_from_line(line: str):
            line = line.strip()
            if not line or line.startswith('#'):
                return None

            # Expected format: "relative/path/to/image.jpg <class_id>"
            # Keep full path even when it contains spaces, drop trailing numeric class label.
            split_line = line.rsplit(maxsplit=1)
            if len(split_line) == 2 and split_line[1].lstrip("+-").isdigit():
                return split_line[0]
            return line

        def _resolve_mapped_path(path_str: str):
            img_path = Path(path_str)
            if img_path.is_absolute():
                return "ABSOLUTE", img_path

            parts = img_path.parts

            # Format example:
            # medical_combined/DERM7pt-clinic/val/.../image.jpg
            if len(parts) >= 2 and parts[0] == "medical_combined":
                dataset_name = parts[1]
                if dataset_name not in mappingPath:
                    raise KeyError(
                        f"Dataset '{dataset_name}' not found in mappingPath for input line path: {path_str}"
                    )
                return dataset_name, mappingPath[dataset_name].joinpath(*parts[2:])

            # Also support paths starting directly with dataset key.
            if len(parts) >= 1 and parts[0] in mappingPath:
                dataset_name = parts[0]
                return dataset_name, mappingPath[dataset_name].joinpath(*parts[1:])

            # Fallback to ROOT_DIR-based resolution for non-mapped inputs.
            return "ROOT_DIR", self.root_dir / img_path

        print(f"[Load] Reading image paths from {self.paths_txt}")
        listed_entries = []
        with open(self.paths_txt, 'r', encoding='utf-8') as f:
            for line in f:
                rel_path = _extract_path_from_line(line)
                if rel_path is None:
                    continue
                dataset_name, img_path = _resolve_mapped_path(rel_path)
                listed_entries.append((dataset_name, rel_path, img_path))

        listed_entries = sorted(listed_entries, key=lambda x: str(x[2]))

        valid_paths = []
        missing_by_dataset = defaultdict(list)
        for dataset_name, original_rel_path, path in tqdm(listed_entries, desc=f"Filtering {self.root_dir.name}"):
            if path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            if not path.exists() or not path.is_file():
                missing_by_dataset[dataset_name].append((original_rel_path, path))
                continue
            try:
                with Image.open(path) as img:
                    if img.width >= min_resolution[0] and img.height >= min_resolution[1]:
                        valid_paths.append(path)
            except Exception:
                continue

        if missing_by_dataset:
            print("\n[Missing] Dataset e istanze mancanti:")
            total_missing = 0
            for dataset_name in sorted(missing_by_dataset.keys()):
                missing_instances = missing_by_dataset[dataset_name]
                total_missing += len(missing_instances)
                print(f"- {dataset_name}: {len(missing_instances)}")
                for original_rel_path, resolved_path in missing_instances:
                    print(f"    rel: {original_rel_path}")
                    print(f"    abs: {resolved_path}")

            raise FileNotFoundError(
                f"Found {total_missing} missing image(s) referenced in {self.paths_txt}. See grouped list above."
            )

        if not valid_paths:
            raise ValueError(
                f"No valid images found from list {self.paths_txt} under root {self.root_dir} after filtering."
            )

        self.image_paths = valid_paths
        self.degradation_types = degradation_types or [
            'gaussian_blur', 'motion_blur', 'jpeg_compression', 'brightness',
            'contrast', 'saturation', 'additive_noise', 'chromatic_aberration',
            'pixelation', 'color_cast'
        ]

    def _get_degradation_params(self):
        max_n = len(self.degradation_types)

        # Determine how many degradations to apply
        if self.min_degradations is not None and self.max_degradations is not None:
            lower = max(1, min(self.min_degradations, max_n))
            upper = min(self.max_degradations, max_n)
            num_degradations = random.randint(lower, upper)
        elif self.max_degradations is not None:
            upper = min(self.max_degradations, max_n)
            num_degradations = random.randint(1, upper)
        else:
            num_degradations = random.randint(1, max_n)

        selected = random.sample(self.degradation_types, num_degradations)
        params = {}
        for degradation in selected:
            if degradation == 'gaussian_blur':
                params[degradation] = random.uniform(2.0, 5.0)
            elif degradation == 'motion_blur':
                params[degradation] = random.randint(10, 30)
            elif degradation == 'jpeg_compression':
                params[degradation] = random.randint(1, 20)
            elif degradation == 'brightness':
                params[degradation] = random.uniform(0.2, 0.8)
            elif degradation == 'contrast':
                params[degradation] = random.uniform(0.2, 0.7)
            elif degradation == 'saturation':
                params[degradation] = random.uniform(0.0, 0.5)
            elif degradation == 'additive_noise':
                params[degradation] = {
                    'type': random.choice(['gaussian', 'salt_pepper']),
                    'amount': random.uniform(0.05, 0.15)
                }
            elif degradation == 'chromatic_aberration':
                params[degradation] = {
                    'shift': random.randint(3, 8),
                    'channel': random.randint(0, 2)
                }
            elif degradation == 'pixelation':
                params[degradation] = random.uniform(0.05, 0.2)
            elif degradation == 'color_cast':
                params[degradation] = {
                    'channel': random.randint(0, 2),
                    'factor': random.uniform(1.5, 2.0)
                }
        return params

    def _apply_degradation(self, img, params):
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
                channel, shift = param['channel'], param['shift']
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
        return degraded_rgb

    def degrade_to_metrics(self, img: Image.Image):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img_np = np.array(img)
        sharp_tensor = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0).float() / 255
        sharp_tensor = sharp_tensor.to(device)

        if random.random() < 0.005:
            return img, {
                'ssim': 1.0,
                'fsim': 1.0,
                'iw_ssim': 1.0,
                'sr_sim': 1.0
            }

        params = self._get_degradation_params()
        degraded_np = self._apply_degradation(img, params)

        degraded_tensor = torch.tensor(degraded_np).permute(2, 0, 1).unsqueeze(0).float() / 255
        degraded_tensor = degraded_tensor.to(device)

        ssim_score = ssim(sharp_tensor, degraded_tensor, data_range=1.0).item()
        fsim_score = fsim(sharp_tensor, degraded_tensor, data_range=1.0).item()
        iw_ssim_score = information_weighted_ssim(sharp_tensor, degraded_tensor, data_range=1.0).item()
        sr_sim_score = srsim(sharp_tensor, degraded_tensor, data_range=1.0).item()
        

        metrics = {
            'ssim': ssim_score,
            'fsim': fsim_score,
            'iw_ssim': iw_ssim_score,
            'sr_sim': sr_sim_score
        }

        return Image.fromarray(degraded_np), metrics


def _process_single_image(path, degrader, output_dir):
    try:
        image_name = path.stem + ".png"
        sharp_img = Image.open(path).convert('RGB')
        degraded_img, metrics = degrader.degrade_to_metrics(sharp_img)

        save_path = output_dir / image_name
        degraded_img.save(save_path)

        return image_name, metrics

    except Exception as e:
        print(f"Failed to process {path.name}: {e}")
        return None


def save_metrics_and_csv(degrader, output_dir, num_threads=16):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scores = []

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {
            executor.submit(_process_single_image, path, degrader, output_dir): path
            for path in degrader.image_paths
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {degrader.root_dir.name}"):
            result = future.result()
            if result:
                scores.append(result)
    

    metrics_list = ['ssim', 'fsim', 'iw_ssim', 'sr_sim']
    csv_path = output_dir / "scores.csv"
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename"] + metrics_list)
        for filename, metrics in scores:
            writer.writerow([filename] + [metrics.get(key, "") for key in metrics_list])

    import collections

    def fine_bin(score):
        bin_floor = int(score * 100)
        return f"{bin_floor:02d}-{bin_floor + 1:02d}"  # e.g., "85-86"

    ssim_hist = collections.Counter()
    fsim_hist = collections.Counter()
    iw_ssim_hist = collections.Counter()
    sr_sim_hist = collections.Counter()

    for _, metrics in scores:
        ssim_hist[fine_bin(metrics['ssim'])] += 1
        fsim_hist[fine_bin(metrics['fsim'])] += 1
        iw_ssim_hist[fine_bin(metrics['iw_ssim'])] += 1
        sr_sim_hist[fine_bin(metrics['sr_sim'])] += 1

    metrics_hists = [
        ("SSIM", ssim_hist),
        ("FSIM", fsim_hist),
        ("IW-SSIM", iw_ssim_hist),
        ("SR-SIM", sr_sim_hist)
    ]

    # Save histograms to a text file (and also print to stdout)
    hist_path = output_dir / "histograms.txt"
    with open(hist_path, "w") as f:
        for title, hist in metrics_hists:
            header = f"\n--- {title} Distribution (0.01 bins) ---\n"
            #print(header.strip())
            f.write(header)
            for i in range(0, 100):
                bin_label = f"{i:02d}-{i + 1:02d}"
                count = hist.get(bin_label, 0)
                line = f"{bin_label}: {count}\n"
                #print(line.strip())
                f.write(line)
    print(f"\n[Saved] Histograms written to {hist_path}")
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run image degradation pipelines.")
    parser.add_argument("--min-degradations", type=int, default=None,
                    help="Minimum number of degradations to apply.")
    parser.add_argument("--max-degradations", type=int, default=None,
                    help="Maximum number of degradations to apply.")
    parser.add_argument("--root-dir", type=str, required=True,
                    help="Root directory used to resolve relative image paths listed in --input-txt.")
    parser.add_argument("--input-txt", type=str, required=True,
                    help="Txt file containing one relative image path per line (relative to --root-dir).")
    parser.add_argument("--output-dir", type=str, required=True,
                    help="Output directory where degraded images and scores.csv will be saved.")
    parser.add_argument("--num-threads", type=int, default=16,
                    help="Number of worker threads used for processing images.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists() and (output_dir / "scores.csv").exists():
        print(f"[Skip] Output already exists at {output_dir}, skipping.")
    else:
        degrader = Degrader(
            root_dir=args.root_dir,
            paths_txt=args.input_txt,
            min_resolution=(384, 384),
            min_degradations=args.min_degradations,
            max_degradations=args.max_degradations,
        )

        print(f"Processing image list with {len(degrader.image_paths)} images...")
        save_metrics_and_csv(degrader, output_dir=output_dir, num_threads=args.num_threads)