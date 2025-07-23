import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch

from pathlib import Path
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from model import RegressionModel
from dataset import SIMDataset


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed()

# === CLI Arguments ===
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='tinyvit',
                    choices=['swinv2', 'mobilevitv2', 'resnet50v2', 'vit', 'efficientnet', 'mobilenetv3', 'tinyvit'],
                    help='Model to use')
parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--image_size', type=int, default=384, help='Input image size')
parser.add_argument('--test_path', type=str, required=True, help='Path to test dataset directory')
parser.add_argument('--target', type=str, required=True, choices=['ssim', 'fsim'], help='Target metric to evaluate')
parser.add_argument('--min_score', type=float, default=0.7, help='Minimum SSIM/FSIM score to include in evaluation')
parser.add_argument('--max_samples_per_bin', type=int, default=50,
                    help='Maximum number of samples to include per bin in the hexbin plot')
args = parser.parse_args()

# === Model Map ===
model_map = {
    'mobilevitv2': 'mobilevitv2_175.cvnets_in22k_ft_in1k_384',
    'resnet50v2': 'resnetv2_50.a1h_in1k',
    'vit': 'vit_small_patch16_384.augreg_in1k',
    'efficientnet': 'efficientnet_b3.ra2_in1k',
    'mobilenetv3': 'mobilenetv3_large_100.ra_in1k',
    'tinyvit': 'tiny_vit_21m_384.dist_in22k_ft_in1k'
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Load Model ===
model = RegressionModel(model_map[args.model]).to(device)
state = torch.load(args.checkpoint, map_location=device)
if "model_state_dict" in state:
    model.load_state_dict(state["model_state_dict"])
else:
    model.load_state_dict(state)
model.eval()

# === Load Dataset ===
test_path = Path(args.test_path)
transform = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor()
])

test_dataset_full = SIMDataset(test_path, transform=transform)
all_scores = test_dataset_full.ssim_scores if args.target == "ssim" else test_dataset_full.fsim_scores

# === Filter dataset based on min_score ===
selected_indices = [i for i, s in enumerate(all_scores) if s >= args.min_score]
print(f"Filtered test set to {len(selected_indices)} samples with {args.target.upper()} >= {args.min_score}")
test_dataset = Subset(test_dataset_full, selected_indices)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

# === Evaluation ===
y_true, y_pred = [], []

with torch.no_grad():
    for images, ssim_scores, fsim_scores in tqdm(test_loader, desc="Testing", unit="batch"):
        images = images.to(device)
        scores = ssim_scores if args.target == "ssim" else fsim_scores
        preds = model(images).cpu().numpy().flatten()
        targets = scores.numpy().flatten()
        y_pred.extend(preds)
        y_true.extend(targets)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# === Bin Sampling (for plotting and metrics) ===
bin_size = 0.01
bins = np.floor(y_true / bin_size) * bin_size
max_samples_per_bin = args.max_samples_per_bin

sampled_indices = []
for b in np.unique(bins):
    bin_idx = np.where(bins == b)[0]
    if len(bin_idx) > max_samples_per_bin:
        bin_idx = np.random.choice(bin_idx, max_samples_per_bin, replace=False)
    sampled_indices.extend(bin_idx.tolist())

# Subset data for both metrics and plotting
y_true_plot = y_true[sampled_indices]
y_pred_plot = y_pred[sampled_indices]

# === Metrics on sampled set ===
r2 = r2_score(y_true_plot, y_pred_plot)
mae = mean_absolute_error(y_true_plot, y_pred_plot)
mse = mean_squared_error(y_true_plot, y_pred_plot)
rmse = np.sqrt(mse)

# Use sampled data for plotting
y_true_plot = y_true[sampled_indices]
y_pred_plot = y_pred[sampled_indices]

# === Create output folder for plots ===
plot_dir = Path("test_plots")
plot_dir.mkdir(parents=True, exist_ok=True)

# === Hexbin Plot ===
hb = plt.hexbin(y_true_plot, y_pred_plot, gridsize=50, cmap='viridis', mincnt=1)
plt.colorbar(hb, label='Count')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title(f'Hexbin Plot: {args.model} on {args.target.upper()} (min_score={args.min_score})')
plt.plot([args.min_score, 1], [args.min_score, 1], 'r--', linewidth=1)
plt.tight_layout()

# Save plot
plot_path = plot_dir / f"hexbin_{args.model}_{args.target}_min{args.min_score}.pdf"
plt.savefig(plot_path, dpi=300)
plt.show()

# === Print Results ===
print(f"\nTest Results for {args.model} on {args.target.upper()} (min_score={args.min_score}):")
print(f"R2 Score : {r2:.4f}")
print(f"MAE      : {mae:.4f}")
print(f"MSE      : {mse:.4f}")
print(f"RMSE     : {rmse:.4f}")