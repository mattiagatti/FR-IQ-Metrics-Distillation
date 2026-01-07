"""
Script per confrontare i tempi di calcolo tra:
- Metriche classiche Full-Reference (usando piq)
- Modelli Deep Learning No-Reference

Metriche analizzate: SSIM, FSIM, IW-SSIM, SR-SIM
"""

import argparse
import time
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from PIL import Image
import pandas as pd

# Import per metriche classiche
from piq import ssim as piq_ssim
from piq import fsim as piq_fsim
from piq import information_weighted_ssim as piq_iw_ssim
from piq import srsim as piq_sr_sim

# Import per modelli DL
from model import RegressionModel
from dataset import SIMDataset


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# --------------------------------------------------------------------------- #
# Modelli DL
# --------------------------------------------------------------------------- #
model_map = {
    'mobilevitv2': 'mobilevitv2_175.cvnets_in22k_ft_in1k_384',
    'resnet50v2': 'resnetv2_50.a1h_in1k',
    'vit': 'vit_small_patch16_384.augreg_in1k',
    'efficientnet': 'efficientnet_b3.ra2_in1k',
    'mobilenetv3': 'mobilenetv3_large_100.ra_in1k',
    'tinyvit': 'tiny_vit_21m_384.dist_in22k_ft_in1k'
}


def load_dl_model(model_name, checkpoint_path):
    """Carica un modello deep learning pre-addestrato."""
    model = RegressionModel(model_map[model_name]).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Timing delle metriche classiche (Full-Reference)
# --------------------------------------------------------------------------- #
def benchmark_classical_metric(metric_name, images_distorted, images_reference, num_runs=1):
    """
    Misura il tempo di calcolo di una metrica classica FR.
    
    Args:
        metric_name: nome della metrica ('ssim', 'fsim', 'iw_ssim', 'sr_sim')
        images_distorted: tensor [N, C, H, W] di immagini degradate
        images_reference: tensor [N, C, H, W] di immagini reference
        num_runs: numero di ripetizioni per la media
    """
    metric_fn = {
        'ssim': piq_ssim,
        'fsim': piq_fsim,
        'iw_ssim': piq_iw_ssim,
        'sr_sim': piq_sr_sim
    }[metric_name]
    
    times = []
    
    # Warmup
    with torch.no_grad():
        _ = metric_fn(images_distorted[:1].to(device), images_reference[:1].to(device))
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    for _ in range(num_runs):
        start = time.perf_counter()
        
        with torch.no_grad():
            for i in range(len(images_distorted)):
                _ = metric_fn(
                    images_distorted[i:i+1].to(device), 
                    images_reference[i:i+1].to(device)
                )
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    return np.mean(times), np.std(times)


# --------------------------------------------------------------------------- #
# Timing dei modelli DL (No-Reference)
# --------------------------------------------------------------------------- #
def benchmark_dl_model(model, dataloader, num_runs=1):
    """
    Misura il tempo di inferenza di un modello DL.
    
    Args:
        model: modello PyTorch
        dataloader: DataLoader con le immagini
        num_runs: numero di ripetizioni per la media
    """
    times = []
    
    # Warmup
    with torch.no_grad():
        for batch in dataloader:
            images, _ = batch
            _ = model(images[:1].to(device))
            break
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    for _ in range(num_runs):
        start = time.perf_counter()
        
        with torch.no_grad():
            for batch in dataloader:
                images, _ = batch
                _ = model(images.to(device))
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    return np.mean(times), np.std(times)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Benchmark timing: Classical FR vs DL NR")
    parser.add_argument('--test_path', type=str, required=True,
                        help='Path to test dataset')
    parser.add_argument('--reference_path', type=str, required=True,
                        help='Path to reference images for classical metrics')
    parser.add_argument('--checkpoint_base', type=str, required=True,
                        help='Base path where model checkpoints are stored')
    parser.add_argument('--model', type=str, default='tinyvit',
                        choices=['swinv2', 'mobilevitv2', 'resnet50v2', 'vit',
                                 'efficientnet', 'mobilenetv3', 'tinyvit'],
                        help='Model architecture to use')
    parser.add_argument('--image_size', type=int, default=384,
                        help='Image size for models')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for DL models')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to use for benchmarking')
    parser.add_argument('--num_runs', type=int, default=3,
                        help='Number of runs to average')
    parser.add_argument('--output_file', type=str, default='timing_comparison.csv',
                        help='Output CSV file')
    
    args = parser.parse_args()
    
    metrics = ['ssim', 'fsim', 'iw_ssim', 'sr_sim']
    
    # Preparazione dataset
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor()
    ])
    
    test_dataset_full = SIMDataset(Path(args.test_path), transform=transform, MOS=False)
    
    # Seleziona un subset di immagini
    num_samples = min(args.num_samples, len(test_dataset_full))
    indices = np.random.choice(len(test_dataset_full), num_samples, replace=False)
    test_dataset = Subset(test_dataset_full, indices)
    
    # DataLoader per modelli DL
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=4)
    
    # Carica le immagini per le metriche classiche
    print("\nLoading images for classical metrics...")
    images_distorted = []
    images_reference = []
    reference_path = Path(args.reference_path)
    
    for idx in tqdm(indices, desc="Loading images"):
        # Immagine degradata
        dist_path = test_dataset_full.image_paths[idx]
        dist_img = Image.open(dist_path).convert('RGB')
        dist_tensor = transform(dist_img)
        images_distorted.append(dist_tensor)
        
        # Immagine reference (assumiamo stesso nome nella cartella reference)
        ref_path = reference_path / dist_path.name
        if not ref_path.exists():
            # Cerca in sottocartelle
            matches = list(reference_path.rglob(dist_path.name))
            if matches:
                ref_path = matches[0]
            else:
                print(f"Warning: Reference image not found for {dist_path.name}")
                # Usa l'immagine stessa come reference (caso peggiore)
                ref_path = dist_path
        
        ref_img = Image.open(ref_path).convert('RGB')
        ref_tensor = transform(ref_img)
        images_reference.append(ref_tensor)
    
    images_distorted = torch.stack(images_distorted)
    images_reference = torch.stack(images_reference)
    
    print(f"\nBenchmarking on {num_samples} images, {args.num_runs} runs each")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"Batch size (DL): {args.batch_size}")
    print("="*70)
    
    results = []
    
    # Benchmark per ogni metrica
    for metric in metrics:
        print(f"\n{'='*70}")
        print(f"Metric: {metric.upper()}")
        print(f"{'='*70}")
        
        # 1. Classical metric (Full-Reference)
        print(f"\n[Classical FR] Computing {metric.upper()}...")
        classical_time, classical_std = benchmark_classical_metric(
            metric, images_distorted, images_reference, args.num_runs
        )
        classical_per_image = classical_time / num_samples
        
        print(f"  Total time: {classical_time:.4f} ± {classical_std:.4f} seconds")
        print(f"  Time per image: {classical_per_image*1000:.2f} ms")
        
        # 2. Deep Learning model (No-Reference)
        checkpoint_path = Path(args.checkpoint_base) / f"{args.model}_{metric}_000100" / "best.pth"
        
        if not checkpoint_path.exists():
            print(f"  [DL NR] Checkpoint not found: {checkpoint_path}")
            print(f"  Skipping DL model for {metric}")
            dl_time = np.nan
            dl_std = np.nan
            dl_per_image = np.nan
            speedup = np.nan
        else:
            print(f"\n[DL NR] Loading model from {checkpoint_path}")
            model = load_dl_model(args.model, checkpoint_path)
            
            print(f"[DL NR] Running inference...")
            dl_time, dl_std = benchmark_dl_model(model, test_loader, args.num_runs)
            dl_per_image = dl_time / num_samples
            speedup = classical_time / dl_time
            
            print(f"  Total time: {dl_time:.4f} ± {dl_std:.4f} seconds")
            print(f"  Time per image: {dl_per_image*1000:.2f} ms")
            print(f"  Speedup: {speedup:.2f}x")
            
            del model
            torch.cuda.empty_cache()
        
        results.append({
            'metric': metric.upper(),
            'classical_total_time': classical_time,
            'classical_std': classical_std,
            'classical_per_image_ms': classical_per_image * 1000,
            'dl_total_time': dl_time,
            'dl_std': dl_std,
            'dl_per_image_ms': dl_per_image * 1000,
            'speedup': speedup
        })
    
    # Salva risultati
    df = pd.DataFrame(results)
    output_path = Path(args.output_file)
    df.to_csv(output_path, index=False, float_format='%.4f')
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")
    print(df.to_string(index=False))
    print(f"\nResults saved to: {output_path}")
    
    # Crea anche una tabella LaTeX
    latex_path = output_path.with_suffix('.tex')
    with open(latex_path, 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Confronto tempi di calcolo: Metriche Classiche FR vs Modelli DL NR}\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\hline\n")
        f.write("Metrica & Classical (ms) & DL NR (ms) & Speedup & Tempo Totale Risparmiato \\\\\n")
        f.write("\\hline\n")
        
        for _, row in df.iterrows():
            saved_time = row['classical_total_time'] - row['dl_total_time']
            f.write(f"{row['metric']} & "
                   f"{row['classical_per_image_ms']:.2f} & "
                   f"{row['dl_per_image_ms']:.2f} & "
                   f"{row['speedup']:.2f}x & "
                   f"{saved_time:.2f}s \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write(f"\\label{{tab:timing_comparison}}\n")
        f.write("\\end{table}\n")
    
    print(f"LaTeX table saved to: {latex_path}")


if __name__ == "__main__":
    main()
