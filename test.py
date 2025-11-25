import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr, kendalltau

from model import RegressionModel
from dataset import SIMDataset


# --------------------------------------------------------------------------- #
# 1. Setup utilities
# --------------------------------------------------------------------------- #
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed()


# --------------------------------------------------------------------------- #
# 2. CLI arguments and yaml config
# --------------------------------------------------------------------------- #
ALL_METRICS = ['ssim', 'fsim', 'ms_ssim', 'iw_ssim',
               'sr_sim', 'vsi', 'dss', 'haarpsi', 'mdsi', 'mos']

parser = argparse.ArgumentParser(description="Evaluate regression model on IQA metrics")
parser.add_argument('--model', type=str, default='tinyvit',
                    choices=['swinv2', 'mobilevitv2', 'resnet50v2', 'vit',
                             'efficientnet', 'mobilenetv3', 'tinyvit'],
                    help='Model architecture to use')
parser.add_argument('--config_path', type=str, required=True,
                    help='Path to YAML config file with test settings')
parser.add_argument('--metric', type=str, required=True, choices=ALL_METRICS, help='Metric to predict')
parser.add_argument(
    '--mos', type=lambda x: str(x).lower() in ['true', '1', 'yes'],
    default=False,
    help='Use MOS scores instead of FR metrics (true/false)'
)

args = parser.parse_args()

if args.config_path:
    cfg_path = Path(args.config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open('r') as f:
        cfg = yaml.safe_load(f) 
    # normalize keys to lowercase and map to args if present
    for k, v in cfg.items():
        key = k.lower()
        if key == 'checkpoint_base':
            base = Path(v)
            if not base.exists():
                raise FileNotFoundError(f"Checkpoint base not found: {base}")
            subdir = f"{args.model}_{args.metric}_000100"
            checkpoint_path = base / subdir / "best.pth"
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Expected checkpoint not found: {checkpoint_path}")
            setattr(args, 'checkpoint', str(checkpoint_path))
        else:
            setattr(args, key, v)



# --------------------------------------------------------------------------- #
# 3. Model loading
# --------------------------------------------------------------------------- #
model_map = {
    'mobilevitv2': 'mobilevitv2_175.cvnets_in22k_ft_in1k_384',
    'resnet50v2': 'resnetv2_50.a1h_in1k',
    'vit': 'vit_small_patch16_384.augreg_in1k',
    'efficientnet': 'efficientnet_b3.ra2_in1k',
    'mobilenetv3': 'mobilenetv3_large_100.ra_in1k',
    'tinyvit': 'tiny_vit_21m_384.dist_in22k_ft_in1k'
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = RegressionModel(model_map[args.model]).to(device)

# Load model checkpoint
state = torch.load(args.checkpoint, map_location=device)
print(f"checkpoint loaded from {args.checkpoint}")
if "model_state_dict" in state:
    model.load_state_dict(state["model_state_dict"])
else:
    model.load_state_dict(state)
model.eval()


# --------------------------------------------------------------------------- #
# 4. Dataset loading
# --------------------------------------------------------------------------- #
test_path = Path(args.test_path)
transform = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor()
])
if args.metric=='mos':
    test_dataset_full = SIMDataset(test_path, transform=transform, MOS=True)
else:
    test_dataset_full = SIMDataset(test_path, transform=transform, MOS=args.mos)

# Dynamically get the selected metric tensor
attr = f"{args.metric}_scores"
if args.mos:
    print("Using MOS scores for evaluation.")
    attr1 = "mos_scores"
    if not hasattr(test_dataset_full, attr1):
        raise AttributeError(f"Dataset does not have metric MOS ")

if not hasattr(test_dataset_full, attr):
    raise AttributeError(f"Dataset does not have metric '{args.metric}' — "
                         f"available: {', '.join(ALL_METRICS)}")

all_scores = getattr(test_dataset_full, attr)

# Filter dataset by minimum score
# selected_indices = [i for i, s in enumerate(all_scores) if s >= args.min_score]
# print(f"Filtered test set to {len(selected_indices)} samples with {args.metric.upper()} >= {args.min_score}")

# test_dataset = Subset(test_dataset_full, selected_indices)
test_loader = DataLoader(test_dataset_full, batch_size=args.batch_size, shuffle=False, num_workers=4)


# --------------------------------------------------------------------------- #
# 5. Evaluation
# --------------------------------------------------------------------------- #
y_true, y_pred = [], []

with torch.no_grad():
    for batch in tqdm(test_loader, desc=f"Evaluating {args.model} on {args.metric}", unit="batch"):
        images, metrics_dict = batch
        images = images.to(device)
        if args.mos:
            targets = metrics_dict["mos"].unsqueeze(1).to(device)
        else:
            targets = metrics_dict[args.metric].unsqueeze(1).to(device)

        preds = model(images).cpu().numpy().flatten()
        targets_np = targets.cpu().numpy().flatten()

        y_pred.extend(preds)
        y_true.extend(targets_np)

y_true = np.array(y_true)
y_pred = np.array(y_pred)


# --------------------------------------------------------------------------- #
# 6. Sampling for plotting
# --------------------------------------------------------------------------- #
bin_size = 0.01
bins = np.floor(y_true / bin_size) * bin_size
max_samples_per_bin = args.max_samples_per_bin

sampled_indices = []
for b in np.unique(bins):
    bin_idx = np.where(bins == b)[0]
    if len(bin_idx) > max_samples_per_bin:
        bin_idx = np.random.choice(bin_idx, max_samples_per_bin, replace=False)
    sampled_indices.extend(bin_idx.tolist())

y_true_plot = y_true[sampled_indices]
y_pred_plot = y_pred[sampled_indices]


# --------------------------------------------------------------------------- #
# 7. Metrics
# --------------------------------------------------------------------------- #
r2 = r2_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
# Rank-based correlations (non-parametric)
spearman_corr, _ = spearmanr(y_true, y_pred)
kendall_corr, _ = kendalltau(y_true, y_pred)

# --------------------------------------------------------------------------- #
# 8. Visualization
# --------------------------------------------------------------------------- #
plot_dir = Path(args.output_dir)
plot_dir.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(6, 6))
hb = plt.hexbin(y_true_plot, y_pred_plot, gridsize=50, cmap='viridis', mincnt=1)
plt.colorbar(hb, label='Count')
if args.mos:
    plt.xlabel(f"MOS")
else:
    plt.xlabel(f"True {args.metric.upper()}")
plt.ylabel(f"Predicted {args.metric.upper()}")
if args.mos:
    plt.title(f"{args.model} on {args.metric.upper()} AGAINST MOS  |  R²={r2:.3f}")
else:
    plt.title(f"{args.model} on {args.metric.upper()}  |  R²={r2:.3f}")
plt.plot([args.min_score, 1], [args.min_score, 1], 'r--', linewidth=1)
plt.tight_layout()

if args.mos:
    plot_path = plot_dir / f"hexbin_{args.model}_{args.metric}_MOS_min{args.min_score}.pdf"
else:
    plot_path = plot_dir / f"hexbin_{args.model}_{args.metric}_min{args.min_score}.pdf"
plt.savefig(plot_path, dpi=300)
plt.show()

# --------------------------------------------------------------------------- #
# 9. Print and save results
# --------------------------------------------------------------------------- #
print(f"\n=== Test Results for {args.model} on {args.metric.upper()} ===")
print(f"R2 Score       : {r2:.4f}")
print(f"MAE            : {mae:.4f}")
print(f"MSE            : {mse:.4f}")
print(f"RMSE           : {rmse:.4f}")
print(f"Spearman ρ     : {spearman_corr:.4f}")
print(f"Kendall τ      : {kendall_corr:.4f}")

# Save results to text file
if args.mos:
    results_path = plot_dir / f"results_{args.model}_{args.metric}_MOS.txt"
else:
    results_path = plot_dir / f"results_{args.model}_{args.metric}.txt"
with open(results_path, "w") as f:
    f.write(f"Model       : {args.model}\n")
    f.write(f"Target      : {args.metric}\n")
    f.write(f"R2 Score    : {r2:.4f}\n")
    f.write(f"MAE         : {mae:.4f}\n")
    f.write(f"MSE         : {mse:.4f}\n")
    f.write(f"RMSE        : {rmse:.4f}\n")
    f.write(f"Spearman ρ  : {spearman_corr:.4f}\n")
    f.write(f"Kendall τ   : {kendall_corr:.4f}\n")

print(f"Results saved to: {results_path}")