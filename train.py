"""
Train SIM-predictor (single-target training supporting all metrics).
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
import json
import yaml
from scipy.stats import spearmanr


# --------------------------------------------------------------------------- #
# 1.  Utilities
# --------------------------------------------------------------------------- #
def set_global_seed(seed: int = 42):
    """Ensure full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _get_scores_tensor_from_dataset(base_dataset: SIMDataset, target: str) -> torch.Tensor:
    """
    Retrieve the score tensor for the given metric from the dataset.
    Expects an attribute named '<target>_scores' to exist.
    """
    attr = f"{target}_scores"
    if not hasattr(base_dataset, attr):
        raise AttributeError(f"Dataset does not have '{attr}'. "
                             f"Ensure scores.csv contains a '{target}' column.")
    return getattr(base_dataset, attr)


def get_balanced_subset_indices(dataset,
                                indices,
                                epoch_seed,
                                target_per_bin=50,
                                target="ssim",
                                num_bins=100):
    """
    Sample up to `target_per_bin` examples per score bin (uniform weighting).
    Ensures balanced sampling across the full metric range.
    """
    rng = random.Random(epoch_seed)
    base_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset

    scores_full = _get_scores_tensor_from_dataset(base_dataset, target)
    bin_size = 1.0 / num_bins

    # Group indices by bin
    bin_to_indices = defaultdict(list)
    for i in indices:
        score = float(scores_full[i].item())
        bin_idx = min(int(score / bin_size), num_bins - 1)
        bin_to_indices[bin_idx].append(i)

    # Compute uniform bin weights
    non_empty = [b for b in range(num_bins) if bin_to_indices[b]]
    if not non_empty:
        raise ValueError("No bins contain data — check input scores or filtering.")

    bin_weights = np.zeros(num_bins, dtype=float)
    bin_weights[non_empty] = 1.0
    bin_weights /= bin_weights.sum()

    # Perform balanced sampling
    subset = []
    for b, weight in enumerate(bin_weights):
        candidates = bin_to_indices[b]
        if not candidates:
            continue
        take = min(int(round(weight * target_per_bin * num_bins)), len(candidates))
        if take > 0:
            subset.extend(rng.sample(candidates, take))

    print(f"[Epoch] Sampled {len(subset)} examples for target='{target}'.")
    return subset


# --------------------------------------------------------------------------- #
# 2.  CLI
# --------------------------------------------------------------------------- #
ALL_METRICS = ['ssim', 'fsim', 'ms_ssim', 'iw_ssim',
               'sr_sim', 'vsi', 'dss', 'haarpsi', 'mdsi','mos']

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='tinyvit',
                    choices=['swinv2', 'mobilevitv2', 'resnet50v2', 'vit',
                             'efficientnet', 'mobilenetv3', 'tinyvit'])
parser.add_argument('--metric', choices=ALL_METRICS, required=True,
                    help="Metric to predict (model outputs one scalar).")
parser.add_argument('--config_path', required=True,
                    help="Path to the configuration file.")
args = parser.parse_args()

# Load training configuration from the config_path and inject into args

conf_path = Path(args.config_path)
if not conf_path.exists():
    raise FileNotFoundError(f"Config file not found: {conf_path}")

# Try JSON first, then YAML if available
cfg = None
with conf_path.open('r') as f:
    cfg = yaml.safe_load(f)
# Defaults (previous CLI defaults)
_defaults = {
    'epochs': 50,
    'batch_size': 32,
    'image_size': 384,
    'lr': 1e-4,
    'train_path': None,
    'val_path': None,
    'train_target_per_bin': 100,
    'val_target_per_bin': 100,
    'min_score': 0.0,
    'max_score': 1.0,
    'patience': 10,
    'experiment_path': "/exp"
}

# Inject config values into args (config takes precedence over defaults)
for key, default in _defaults.items():
    if key.upper() in cfg:
        val = cfg[key.upper()]
    else:
        val = default
    setattr(args, key, val)
args.lr=float(args.lr)
# Basic validation
if args.train_path is None:
    raise ValueError("train_path must be provided in the config file.")

print(f"Loaded config from {conf_path}")



# --------------------------------------------------------------------------- #
# 3.  Setup
# --------------------------------------------------------------------------- #
set_global_seed(42)
g = torch.Generator().manual_seed(42)
epochs_without_improvement = 0

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
transform = transforms.Compose([
    transforms.Resize((args.image_size,) * 2),
    transforms.ToTensor()
])

min_score, max_score = args.min_score, args.max_score
range_suffix = f"{int(min_score * 100):03d}{int(max_score * 100):03d}"

# --------------------------------------------------------------------------- #
# 4.  Dataset Loading and Filtering
# --------------------------------------------------------------------------- #
train_path = Path(args.train_path)
if args.metric == 'mos':
    train_subset = SIMDataset(train_path, transform=transform, MOS=True)
else:
    train_dataset_full = SIMDataset(train_path, transform=transform)
    # Filter training data based on selected metric
    scores_full = _get_scores_tensor_from_dataset(train_dataset_full, args.metric)
    filtered_train_indices = [i for i, s in enumerate(scores_full)
                            if min_score <= float(s.item()) <= max_score]
    print(f"Filtered training set to {len(filtered_train_indices)} samples "
        f"with {min_score} <= {args.metric.upper()} <= {max_score}")
    train_subset = Subset(train_dataset_full, filtered_train_indices)

# --- Validation split ---
if args.val_path:
    val_path = Path(args.val_path)
    if args.metric == 'mos':
        val_subset = SIMDataset(val_path, transform=transform, MOS=True)

    else:
        val_dataset_full = SIMDataset(val_path, transform=transform)

        val_scores_full = _get_scores_tensor_from_dataset(val_dataset_full, args.metric)
        filtered_val_indices = [i for i, s in enumerate(val_scores_full)
                                if min_score <= float(s.item()) <= max_score]
        print(f"Filtered validation set to {len(filtered_val_indices)} samples "
            f"with {min_score} <= {args.metric.upper()} <= {max_score}")
        val_dataset = Subset(val_dataset_full, filtered_val_indices)

        # Balanced validation sampling
        num_bins = 100
        bin_size = 1.0 / num_bins
        bin_to_indices = defaultdict(list)

        for local_idx, global_idx in enumerate(val_dataset.indices):
            score = float(val_scores_full[global_idx].item())
            bin_idx = min(int(score / bin_size), num_bins - 1)
            bin_to_indices[bin_idx].append(local_idx)

        val_indices = []
        for bin_idx, bin_idxs in bin_to_indices.items():
            if not bin_idxs:
                continue
            sampled = sorted(
                bin_idxs,
                key=lambda i: val_dataset_full.image_paths[val_dataset.indices[i]].name
            )[:args.val_target_per_bin]
            val_indices.extend(sampled)

        print(f"[Validation] Using {len(val_indices)} balanced samples.")
        val_subset = Subset(val_dataset, val_indices)

else:
    # Automatic validation split from training set
    num_bins = 100
    bin_size = 1.0 / num_bins

    bin_to_indices = defaultdict(list)
    for idx in range(len(train_dataset_full)):
        global_idx = train_dataset_full.indices[idx]
        score = float(scores_full[global_idx].item())
        bin_idx = min(int(score / bin_size), num_bins - 1)
        bin_to_indices[bin_idx].append(idx)

    val_indices = []
    for bin_idx, bin_idxs in bin_to_indices.items():
        if len(bin_idxs) >= args.val_target_per_bin:
            val_indices.extend(random.sample(bin_idxs, args.val_target_per_bin))

    if len(val_indices) == 0:
        raise ValueError("No validation bins had enough samples.")

    all_indices = set(range(len(train_dataset_full)))
    train_indices = list(all_indices - set(val_indices))
    val_subset = Subset(train_dataset_full, val_indices)

    train_subset = Subset(train_dataset_full, train_indices)
val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False,
                        num_workers=8, pin_memory=True, generator=g)

print(f"Train size: {len(train_subset)}, Validation size: {len(val_subset)}")

# --------------------------------------------------------------------------- #
# 5.  Model and Training Setup
# --------------------------------------------------------------------------- #
model = RegressionModel(model_map[args.model]).to(device)

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

r2_scores = []
checkpoint_name = f"{args.model}_{args.metric}_{range_suffix}"
checkpoint_dir = Path(args.experiment_path) / checkpoint_name
checkpoint_dir.mkdir(parents=True, exist_ok=True)

best_model_path = None
best_r2 = -np.inf
last_checkpoint_path = checkpoint_dir / "last.pth"
start_epoch = 0

# Resume if checkpoint exists
if last_checkpoint_path.exists():
    print ("-----The experiment was already done-------")
    exit()
    print(f"Found checkpoint: {last_checkpoint_path}")
    checkpoint = torch.load(last_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    print (checkpoint.keys())
    best_r2 = checkpoint.get('best_r2', -np.inf)
    start_epoch = checkpoint.get('epoch', 0)
    print(f"Resuming from epoch {start_epoch}, best R2: {best_r2:.4f}")

# --------------------------------------------------------------------------- #
# 6.  Training Loop
# --------------------------------------------------------------------------- #
for epoch in range(start_epoch, args.epochs):
    epoch_seed = 42 + epoch
    train_subset_base = train_subset.dataset if isinstance(train_subset, Subset) else train_subset

    subset_indices = get_balanced_subset_indices(
        dataset=train_subset_base,
        indices=train_subset.indices if isinstance(train_subset, Subset) else list(range(len(train_subset))),
        epoch_seed=epoch_seed,
        target_per_bin=args.train_target_per_bin,
        target=args.metric
    )
    subset_dataset = Subset(train_subset_base, subset_indices)

    train_loader = DataLoader(subset_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, generator=g)

    model.train()
    pbar = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{args.epochs}] Training ({args.metric})", unit="batch")
    running_loss = 0.0
    
    for images, metrics_dict in pbar:
        images = images.to(device)

        # Select only the chosen metric
        scores = metrics_dict[args.metric].unsqueeze(1).to(device)
        scores = (scores - min_score) / (max_score - min_score)

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

    # ----------------------------------------------------------------------- #
    # Validation
    # ----------------------------------------------------------------------- #
    model.eval()
    y_true, y_pred, original_scores = [], [], []

    with torch.no_grad():
        for images, metrics_dict in tqdm(val_loader, desc=f"[Validation] ({args.metric})", unit="batch"):
            images = images.to(device)
            raw_scores = metrics_dict[args.metric].unsqueeze(1)
            normalized_scores = (raw_scores - min_score) / (max_score - min_score)

            preds = model(images).cpu().numpy().flatten()
            y_true.extend(normalized_scores.cpu().numpy().flatten().tolist())
            y_pred.extend(preds.tolist())
            original_scores.extend(raw_scores.cpu().numpy().flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    spearman_corr, _ = spearmanr(y_true, y_pred)
    print(f"Validation ({args.metric}): R2={r2:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f} SROCC={spearman_corr:.4f}")
    r2_scores.append(r2)

    # ----------------------------------------------------------------------- #
    # Save Best Model and Plots
    # ----------------------------------------------------------------------- #
    best_epoch = epoch + 1
    if r2 > best_r2:
        best_r2 = r2
        epochs_without_improvement = 0
        best_model_path = checkpoint_dir / "best.pth"
        torch.save(model.state_dict(), best_model_path)
        print(f"New best model saved at {best_model_path.name}")

        # Hexbin plot
        plt.figure()
        plt.hexbin(y_true, y_pred, gridsize=60, cmap='viridis', mincnt=1)
        plt.colorbar(label='Count')
        plt.xlabel(f"True {args.metric.upper()}")
        plt.ylabel(f"Predicted {args.metric.upper()}")
        plt.title(f"{args.metric.upper()} vs Predicted")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.savefig(checkpoint_dir / "hexbin.pdf", dpi=300, bbox_inches='tight')
        plt.close()

        # Scatter plot
        max_points = 1000
        idxs = np.random.choice(len(y_true), min(len(y_true), max_points), replace=False)
        plt.figure()
        plt.scatter(y_true[idxs], y_pred[idxs], alpha=0.5)
        plt.xlabel(f"True {args.metric.upper()}")
        plt.ylabel(f"Predicted {args.metric.upper()}")
        plt.title(f"{args.metric.upper()} vs Predicted")
        plt.grid(True)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.savefig(checkpoint_dir / "scatter.pdf", dpi=300, bbox_inches='tight')
        plt.close()

        # Save predictions to CSV
        denorm_y_pred = y_pred * (max_score - min_score) + min_score
        csv_data = {
            f"true_{args.metric}": y_true,
            f"pred_{args.metric}": y_pred,
            f"denorm_true_{args.metric}": original_scores,
            f"denorm_pred_{args.metric}": denorm_y_pred
        }
        pd.DataFrame(csv_data).to_csv(checkpoint_dir / "predictions.csv", index=False)
    else:
        epochs_without_improvement += 1
        print(f"No improvement in validation r2. counter {epochs_without_improvement}/{args.patience}")
    
    if epochs_without_improvement >= args.patience:
        print("Early stopping: no improvement for 10 epochs.")
        break
    # ----------------------------------------------------------------------- #
    # Save Checkpoint and R2 Progress
    # ----------------------------------------------------------------------- #
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_r2': best_r2,
    }, last_checkpoint_path)

    plt.figure()
    plt.plot(range(len(r2_scores)), r2_scores, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("R2 Score")
    plt.title(f"R2 Over Epochs ({args.metric})")
    plt.grid(True)
    plt.savefig(checkpoint_dir / f"r2_over_epochs_{args.metric}.pdf", dpi=300, bbox_inches='tight')
    plt.close()

# --------------------------------------------------------------------------- #
# 7.  Save Best Results
# --------------------------------------------------------------------------- #
results_txt = checkpoint_dir / "best_results.txt"
with open(results_txt, "w") as f:
    f.write(f"Best Epoch: {best_epoch}\n")
    f.write(f"Best R2 Score : {best_r2:.4f}\n")
    f.write(f"Best MAE      : {mae:.4f}\n")
    f.write(f"Best RMSE     : {rmse:.4f}\n")
    f.write(f"Best SROCC    : {spearman_corr:.4f}\n")