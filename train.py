"""
Train SIM-predictor.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

from collections import defaultdict
from dataset import SIMDataset
from model import RegressionModel
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from torchvision import transforms


# --------------------------------------------------------------------------- #
# 1.  Utility
# --------------------------------------------------------------------------- #
def set_global_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def get_balanced_subset_indices(dataset,
                                indices,
                                epoch_seed,
                                target_per_bin=50,
                                target="ssim",
                                num_bins=100):
    """
    Sample up to `target_per_bin` examples per bin using uniform weights.
    """
    rng = random.Random(epoch_seed)
    base_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
    scores = base_dataset.ssim_scores if target == "ssim" else base_dataset.fsim_scores
    bin_size = 1.0 / num_bins

    # Organize indices by bin
    bin_to_indices = defaultdict(list)
    for i in indices:
        score = scores[i].item()
        bin_idx = min(int(score / bin_size), num_bins - 1)
        bin_to_indices[bin_idx].append(i)

    # Compute weights (uniform here)
    total_available_bins = sum(1 for b in bin_to_indices.values() if b)
    if total_available_bins == 0:
        raise ValueError("No bins contain data — check input scores or filtering.")

    bin_weights = np.array([1.0 if bin_to_indices[b] else 0.0 for b in range(num_bins)])
    bin_weights /= bin_weights.sum()

    # Sample from each bin
    subset = []
    for b, weight in enumerate(bin_weights):
        candidates = bin_to_indices[b]
        n = len(candidates)
        take = min(int(round(weight * target_per_bin * num_bins)), n)
        if take > 0:
            subset.extend(rng.sample(candidates, take))

    print(f"[Epoch] Sampled {len(subset)} examples.")
    return subset


# --------------------------------------------------------------------------- #
# 2.  CLI
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument('--model',  default='tinyvit',
                    choices=['swinv2','mobilevitv2','resnet50v2','vit',
                             'efficientnet','mobilenetv3','tinyvit'])
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--image_size', type=int, default=384)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--train_path', required=True)
parser.add_argument('--val_path',   default=None)
parser.add_argument('--train_target_per_bin', type=int, default=100)
parser.add_argument('--val_target_per_bin',   type=int, default=100)
parser.add_argument('--target', choices=['ssim','fsim'], required=True)
parser.add_argument('--min_score', type=float, default=0.0,
                    help="Minimum SSIM/FSIM score to include in training/validation")
parser.add_argument('--max_score', type=float, default=1.0,
                    help="Maximum SSIM/FSIM score to include in training/validation")
args = parser.parse_args()
# --------------------------------------------------------------------------- #

set_global_seed(42)
g = torch.Generator().manual_seed(42)

# === Model Map ===
model_map = {
    'swinv2': 'swinv2_base_window12to24_192to384.ms_in22k_ft_in1k',
    'mobilevitv2': 'mobilevitv2_175.cvnets_in22k_ft_in1k_384',
    'resnet50v2': 'resnetv2_50.a1h_in1k',
    'vit': 'vit_small_patch16_384.augreg_in1k',
    'efficientnet': 'efficientnet_b3.ra2_in1k',
    'mobilenetv3': 'mobilenetv3_large_100.ra_in1k',
    'tinyvit': 'tiny_vit_21m_384.dist_in22k_ft_in1k'
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([transforms.Resize((args.image_size,)*2),
                                transforms.ToTensor()])
min_score = args.min_score
max_score = args.max_score
min_score_str = f"{int(min_score * 100):03d}"
max_score_str = f"{int(max_score * 100):03d}"
range_suffix = f"{min_score_str}{max_score_str}"

# === Dataset Paths ===
train_path = Path(args.train_path)
train_path = train_path.parent / f"{train_path.stem}_{range_suffix}"

train_dataset_full = SIMDataset(train_path, transform=transform)
train_scores = train_dataset_full.ssim_scores if args.target == "ssim" else train_dataset_full.fsim_scores
filtered_train_indices = [i for i, s in enumerate(train_scores) if args.min_score <= s <= args.max_score]
print(f"Filtered training set to {len(filtered_train_indices)} samples with {args.min_score} <= {args.target.upper()} <= {args.max_score}")
train_dataset = Subset(train_dataset_full, filtered_train_indices)

if args.val_path:
    val_path = Path(args.val_path)
    val_path = val_path.parent / f"{val_path.stem}_{range_suffix}"

    val_dataset_full = SIMDataset(val_path, transform=transform)
    if args.min_score > 0.0:
        val_scores = val_dataset_full.ssim_scores if args.target == "ssim" else val_dataset_full.fsim_scores
        filtered_val_indices = [i for i, s in enumerate(val_scores) if args.min_score <= s <= args.max_score]
        print(f"Filtered validation set to {len(filtered_val_indices)} samples with {args.min_score} <= {args.target.upper()} <= {args.max_score}")
        val_dataset = Subset(val_dataset_full, filtered_val_indices)
    else:
        val_dataset = val_dataset_full

    # Group indices into bins for balanced val sampling
    if isinstance(val_dataset, Subset):
        val_scores_full = val_dataset.dataset.ssim_scores if args.target == "ssim" else val_dataset.dataset.fsim_scores
        val_indices_base = val_dataset.indices
    else:
        val_scores_full = val_dataset.ssim_scores if args.target == "ssim" else val_dataset.fsim_scores
        val_indices_base = list(range(len(val_dataset)))
    
    val_scores = [val_scores_full[i] for i in val_indices_base]

    val_indices = []

    num_bins = 100
    bin_size = 1.0 / num_bins
    bin_to_indices = defaultdict(list)

    for idx, score in enumerate(val_scores):
        bin_idx = min(int(score / bin_size), num_bins - 1)
        bin_to_indices[bin_idx].append(idx)

    total_bins_used = 0
    for bin_idx, bin_idxs in bin_to_indices.items():
        if not bin_idxs:
            continue
        sampled = sorted(
            bin_idxs,
            key=lambda i: val_dataset_full.image_paths[val_indices_base[i]].name
        )[:args.val_target_per_bin]
        val_indices.extend(sampled)
        total_bins_used += 1

        if len(bin_idxs) < args.val_target_per_bin:
            scores = [val_scores[i].item() for i in bin_idxs]
            print(f"  Bin {bin_idx:02d} (range {bin_idx * bin_size:.2f}–{(bin_idx + 1) * bin_size:.2f}): "
                  f"{len(bin_idxs)} samples -> using all available, below target ({args.val_target_per_bin})")

    print(f"[Validation] Using {len(val_indices)} samples from {total_bins_used} non-empty bins")
    train_indices = list(range(len(train_dataset)))
    val_subset = Subset(val_dataset, val_indices)

else:
    # Automatic validation split from train_dataset
    val_indices = []
    train_indices = []

    num_bins = 100
    bin_size = 1.0 / num_bins
    scores = train_dataset.ssim_scores if args.target == "ssim" else train_dataset.fsim_scores

    bin_to_indices = defaultdict(list)
    for idx in range(len(train_dataset)):
        score = scores[idx].item()
        bin_idx = min(int(score / bin_size), num_bins - 1)
        bin_to_indices[bin_idx].append(idx)

    for bin_idx, bin_idxs in bin_to_indices.items():
        if len(bin_idxs) >= args.val_target_per_bin:
            sampled_val = random.sample(bin_idxs, args.val_target_per_bin)
            val_indices.extend(sampled_val)

    if len(val_indices) == 0:
        raise ValueError("No validation bins had enough samples. Reduce --val_target_per_bin or check dataset.")

    all_indices = set(range(len(train_dataset)))
    train_indices = list(all_indices - set(val_indices))
    val_subset = Subset(train_dataset, val_indices)

train_subset = Subset(train_dataset, train_indices)

val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True, generator=g)

print(f"Train size: {len(train_subset)}, Validation size: {len(val_subset)}")

model = RegressionModel(model_map[args.model]).to(device)

backbone_params = sum(p.numel() for p in model.backbone.parameters())
trainable_backbone_params = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)

print(f"Backbone total parameters: {backbone_params}")
print(f"Backbone trainable parameters: {trainable_backbone_params}")

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

# === Training & Validation ===
r2_scores = []
all_y_true = []
all_y_pred = []


# Build base name
checkpoint_name = f"{args.model}_{args.target}_{range_suffix}"

checkpoint_dir = Path("exp") / checkpoint_name
checkpoint_dir.mkdir(parents=True, exist_ok=True)

best_model_path = None

best_r2 = -np.inf

last_checkpoint_path = checkpoint_dir / f"last.pth"

start_epoch = 0  # Always define start_epoch

if last_checkpoint_path.exists():
    print(f"Found checkpoint: {last_checkpoint_path}")
    checkpoint = torch.load(last_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    best_r2 = checkpoint.get('best_r2', -np.inf)
    start_epoch = checkpoint.get('epoch', 0)
    print(f"Resuming from epoch {start_epoch}, best R2: {best_r2:.4f}")

for epoch in range(start_epoch, args.epochs):
    epoch_seed = 42 + epoch
    train_subset_base = train_subset.dataset if isinstance(train_subset, Subset) else train_subset
    subset_indices = get_balanced_subset_indices(
        dataset=train_subset_base,
        indices=train_subset.indices,
        epoch_seed=epoch_seed,
        target_per_bin=args.train_target_per_bin,
        target=args.target
    )
    subset_dataset = Subset(train_subset.dataset, subset_indices)

    train_loader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, generator=g)

    model.train()
    pbar = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{args.epochs}] Training", unit="batch")
    running_loss = 0.0
    
    for images, ssim_scores, fsim_scores in pbar:
        images = images.to(device)
        scores = ssim_scores if args.target == 'ssim' else fsim_scores
        scores = (scores - min_score) / (max_score - min_score)
        scores = scores.to(device)
    
        preds = model(images)

        assert preds.shape == scores.shape, f"Shape mismatch: preds={preds.shape}, scores={scores.shape}"
        
        loss = criterion(preds, scores)
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        running_loss += loss.item() * images.size(0)
        avg_loss = running_loss / ((pbar.n + 1) * args.batch_size)
        pbar.set_postfix(avg_loss=avg_loss)

    scheduler.step()
    print(f"Train Loss: {running_loss / len(train_loader.dataset):.4f}")

    # === Validation ===
    model.eval()

    y_true, y_pred = [], []
    original_scores = []
    
    with torch.no_grad():
        for images, ssim_scores, fsim_scores in tqdm(val_loader, desc="[Validation]", unit="batch"):
            images = images.to(device)
            raw_scores = ssim_scores if args.target == 'ssim' else fsim_scores  # Keep original for later
            normalized_scores = (raw_scores - min_score) / (max_score - min_score)
    
            raw_scores_np = raw_scores.cpu().numpy().flatten()
            normalized_scores_np = normalized_scores.cpu().numpy().flatten()
            preds = model(images).cpu().numpy().flatten()
    
            y_true.extend(normalized_scores_np.tolist())
            y_pred.extend(preds.tolist())
            original_scores.extend(raw_scores_np.tolist())


    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Unfiltered metrics
    r2 = r2_score(y_true, y_pred)
    best_mae = mean_absolute_error(y_true, y_pred)
    best_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    print(f"Validation Metrics: R2 {r2:.4f}  MAE {best_mae:.4f}  RMSE {best_rmse:.4f}")
    
    r2_scores.append(r2)


    if r2 > best_r2:
        if best_model_path and best_model_path.exists():
            best_model_path.unlink()
            print(f"Deleted previous best model: {best_model_path.name}")
    
        best_r2 = r2
        best_model_path = checkpoint_dir / "best.pth"
        torch.save(model.state_dict(), best_model_path)
        print(f"New best model saved at: {best_model_path.name}")
    
        best_epoch = epoch + 1
    
        # Save plots for best epoch
        plt.figure()
        plt.hexbin(y_true, y_pred, gridsize=60, cmap='viridis', mincnt=1)
        plt.colorbar(label='Count')
        plt.xlabel(f"True {args.target.upper()}")
        plt.ylabel(f"Predicted {args.target.upper()}")
        plt.title(f"{args.target.upper()} vs Predicted {args.target.upper()}")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        hexbin_path = checkpoint_dir / f"hexbin.pdf"
        plt.savefig(hexbin_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved hexbin plot to {hexbin_path}")
    
        max_points = 1000
        if len(y_true) > max_points:
            indices = np.random.choice(len(y_true), size=max_points, replace=False)
            y_true_plot = y_true[indices]
            y_pred_plot = y_pred[indices]
        else:
            y_true_plot = y_true
            y_pred_plot = y_pred
    
        plt.figure()
        plt.scatter(y_true_plot, y_pred_plot, alpha=0.5)
        plt.xlabel(f"True {args.target.upper()}")
        plt.ylabel(f"Predicted {args.target.upper()}")
        plt.title(f"{args.target.upper()} vs Predicted {args.target.upper()}")
        plt.grid(True)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        scatter_path = checkpoint_dir / f"scatter.pdf"
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved scatter plot to {scatter_path}")

        # Save predictions and ground truths from best epoch to CSV
        denorm_y_pred = (np.array(y_pred) * (max_score - min_score) + min_score)
        
        csv_data = {
            f"true_{args.target}": y_true,
            f"predicted_{args.target}": y_pred,
            f"denorm_true_{args.target}": original_scores,
            f"denorm_predicted_{args.target}": denorm_y_pred
        }
        
        csv_df = pd.DataFrame(csv_data)
        csv_path = checkpoint_dir / f"predictions.csv"
        csv_df.to_csv(csv_path, index=False)
        print(f"Saved predictions to {csv_path}")
    
    # Save full training state for resuming (always)
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_r2': best_r2,
    }, last_checkpoint_path)
    print(f"Saved last checkpoint to {last_checkpoint_path}")
    
    # Always update R2-over-time plot
    plt.figure()
    plt.plot(range(len(r2_scores)), r2_scores, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("R2 Score")
    plt.title("R2 Score Over Epochs")
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    plt.grid(True)
    r2_plot_path = checkpoint_dir / f"r2_over_epochs_{args.target}.pdf"
    plt.savefig(r2_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

# === Save best validation result and epoch to file ===
results_txt = checkpoint_dir / "best_results.txt"

with open(results_txt, "w") as f:
    f.write(f"Best Epoch: {best_epoch}\n")
    f.write(f"Best R2 Score : {best_r2:.4f}\n")
    f.write(f"Best MAE      : {best_mae:.4f}\n")
    f.write(f"Best RMSE     : {best_rmse:.4f}\n")