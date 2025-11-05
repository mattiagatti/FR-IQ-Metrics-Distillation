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



def degrade_to_metrics(img_ref: Image.Image, img_degraded: Image.Image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_ref_np = np.array(img_ref)
    ref_tensor = torch.tensor(img_ref_np).permute(2, 0, 1).unsqueeze(0).float() / 255
    ref_tensor = ref_tensor.to(device)

    img_degraded_np = np.array(img_degraded)
    degraded_tensor = torch.tensor(img_degraded_np).permute(2, 0, 1).unsqueeze(0).float() / 255
    degraded_tensor = degraded_tensor.to(device)

    ssim_score = ssim(ref_tensor, degraded_tensor, data_range=1.0).item()
    fsim_score = fsim(ref_tensor, degraded_tensor, data_range=1.0).item()
    ms_ssim_score = multi_scale_ssim(ref_tensor, degraded_tensor, data_range=1.0).item()
    iw_ssim_score = information_weighted_ssim(ref_tensor, degraded_tensor, data_range=1.0).item()
    vif_p_score = vif_p(ref_tensor, degraded_tensor, data_range=1.0).item()
    sr_sim_score = srsim(ref_tensor, degraded_tensor, data_range=1.0).item()
    gmsd_score = gmsd(ref_tensor, degraded_tensor, data_range=1.0).item()
    ms_gmsd_score = multi_scale_gmsd(ref_tensor, degraded_tensor, data_range=1.0).item()
    vsi_score = vsi(ref_tensor, degraded_tensor, data_range=1.0).item()
    dss_score = dss(ref_tensor, degraded_tensor, data_range=1.0).item()
    haarpsi_score = haarpsi(ref_tensor, degraded_tensor, data_range=1.0).item()
    mdsi_score = mdsi(ref_tensor, degraded_tensor, data_range=1.0).item()

    metrics = {
        'ssim': ssim_score,
        'fsim': fsim_score,
        'ms_ssim': ms_ssim_score,
        'iw_ssim': iw_ssim_score,
        'vif_p': vif_p_score,
        'sr_sim': sr_sim_score,
        'gmsd': gmsd_score,
        'ms_gmsd': ms_gmsd_score,
        'vsi': vsi_score,
        'dss': dss_score,
        'haarpsi': haarpsi_score,
        'mdsi': mdsi_score
    }

    return metrics


def _process_single_image(path_degraded, ref_image_dir):
    try:
        ref_name = path_degraded.name.split('_')[0].upper() + ".BMP"
        ref_img = Image.open(ref_image_dir / ref_name).convert('RGB')
        degr_img = Image.open(path_degraded).convert('RGB')
        metrics = degrade_to_metrics(ref_img, degr_img)

        return path_degraded.name, metrics

    except Exception as e:
        print(f"Failed to process {path_degraded.name}: {e}")
        return None


def save_metrics_and_csv(input_path, output_dir):
    base = Path(input_path)
    degraded_path = base / "distorted_images"
    ref_path = base / "reference_images"

    # list .bmp files (case-insensitive) directly in degraded_path
    degraded_image_paths = []
    if degraded_path.exists():
        degraded_image_paths = sorted(
            [p for p in degraded_path.iterdir() if p.is_file() and p.suffix.lower() == ".bmp"]
        )
    else:
        print(f"No such directory: {degraded_path}")
    

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scores = []

    for path in tqdm(degraded_image_paths, desc=f"Processing degraded images"):
        result = _process_single_image(path, ref_path)
        if result:
            scores.append(result)
    

    metrics_list = ['ssim', 'fsim', 'ms_ssim', 'iw_ssim', 'vif_p', 'sr_sim', 'gmsd', 'ms_gmsd', 'vsi', 'dss', 'haarpsi', 'mdsi']
    csv_path = output_dir / "scores.csv"
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename"] + metrics_list)
        for filename, metrics in scores:
            writer.writerow([filename] + [metrics.get(key, "") for key in metrics_list])

    import collections
    import math

    def fine_bin(score):
        bin_floor = int(score * 100)
        return f"{bin_floor:02d}-{bin_floor + 1:02d}"  # e.g., "85-86"

    ssim_hist = collections.Counter()
    fsim_hist = collections.Counter()
    ms_ssim_hist = collections.Counter()
    iw_ssim_hist = collections.Counter()
    vif_p_hist = collections.Counter()
    sr_sim_hist = collections.Counter()
    gmsd_hist = collections.Counter()
    ms_gmsd_hist = collections.Counter()
    vsi_hist = collections.Counter()
    dss_hist = collections.Counter()
    haarpsi_hist = collections.Counter()
    mdsi_hist = collections.Counter()

    for _, metrics in scores:
        ssim_hist[fine_bin(metrics['ssim'])] += 1
        fsim_hist[fine_bin(metrics['fsim'])] += 1
        ms_ssim_hist[fine_bin(metrics['ms_ssim'])] += 1
        iw_ssim_hist[fine_bin(metrics['iw_ssim'])] += 1
        vif_p_hist[fine_bin(metrics['vif_p'])] += 1
        sr_sim_hist[fine_bin(metrics['sr_sim'])] += 1
        gmsd_hist[fine_bin(metrics['gmsd'])] += 1
        ms_gmsd_hist[fine_bin(metrics['ms_gmsd'])] += 1
        vsi_hist[fine_bin(metrics['vsi'])] += 1
        dss_hist[fine_bin(metrics['dss'])] += 1
        haarpsi_hist[fine_bin(metrics['haarpsi'])] += 1
        mdsi_hist[fine_bin(metrics['mdsi'])] += 1

    metrics_hists = [
        ("SSIM", ssim_hist),
        ("FSIM", fsim_hist),
        ("MS-SSIM", ms_ssim_hist),
        ("IW-SSIM", iw_ssim_hist),
        ("VIF-P", vif_p_hist),
        ("SR-SIM", sr_sim_hist),
        ("GMSD", gmsd_hist),
        ("MS-GMSD", ms_gmsd_hist),
        ("VSI", vsi_hist),
        ("DSS", dss_hist),
        ("HaarPSI", haarpsi_hist),
        ("MDSI", mdsi_hist),
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
    parser.add_argument("--dataset_path", type=str, default=None,required=True,
                    help="Path to the dataset.")
    args = parser.parse_args()

    output_dir = Path(args.dataset_path)

    if output_dir.exists() and (output_dir / "scores.csv").exists():
        print(f"[Skip] scores.csv already exists at {output_dir}, skipping.")
        exit(0)

    print(f"Processing Dataset {output_dir.name}...")
    save_metrics_and_csv(args.dataset_path, output_dir=output_dir)