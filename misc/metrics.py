import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
from collections import defaultdict

# Base directories
base_dir = Path("exp")
plots_base = Path("plots")
table_dir = Path("tables")
table_dir.mkdir(exist_ok=True)

model_name_map = {
    "efficientnet": "EfficientNet-B3",
    "mobilenetv3": "MobileNetV3",
    "resnet50v2": "ResNet50v2",
    "mobilevitv2": "MobileViT-v2",
    "tinyvit": "TinyViT",
    "vit": "ViT"
}

# Results dict: {score_str: {model: {'ssim': metrics, 'fsim': metrics}}}
results = {}

exp_folders = [p for p in base_dir.iterdir() if p.is_dir() and re.search(r'_\d{3,}$', p.name)]

for folder in exp_folders:
    csv_path = folder / "predictions.csv"

    if not csv_path.exists():
        print(f"[Skip] No predictions.csv in {folder}")
        continue

    # Determine metric
    if "fsim" in folder.name.lower():
        true_col, pred_col = "true_fsim", "predicted_fsim"
        metric = "fsim"
    elif "ssim" in folder.name.lower():
        true_col, pred_col = "true_ssim", "predicted_ssim"
        metric = "ssim"
    else:
        print(f"[Skip] Unknown metric in folder name: {folder.name}")
        continue

    # Extract score range (e.g., 040070)
    match = re.search(r"_(\d{3})(\d{3})$", folder.name)
    if match:
        min_score_str, max_score_str = match.groups()
        score_str = f"{min_score_str}{max_score_str}"
    else:
        print(f"[Skip] Could not parse score range in {folder.name}")
        continue

    # Extract model name (before first "_")
    raw_model = folder.name.split("_")[0].lower()
    model_name = model_name_map.get(raw_model, raw_model)

    # Create plot directory
    plot_dir = plots_base / metric / score_str
    plot_dir.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_csv(csv_path)

        # Prefer denormalized columns if available
        denorm_true_col = f"denorm_{true_col}"
        denorm_pred_col = f"denorm_{pred_col}"
        
        if denorm_true_col in df.columns and denorm_pred_col in df.columns:
            y_true = df[denorm_true_col]
            y_pred = df[denorm_pred_col]
            print(f"[Info] Using denormalized columns: {denorm_true_col}, {denorm_pred_col}")
        elif true_col in df.columns and pred_col in df.columns:
            y_true = df[true_col]
            y_pred = df[pred_col]
        else:
            print(f"[Skip] Missing required columns in {csv_path}")
            continue

        # Store metrics for table
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)

        results.setdefault(score_str, {}).setdefault(model_name, {})[metric] = {
            "R2": r2,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse
        }

        # Plotting
        target_name = true_col.split("_")[1].upper()
        
        # Extract min_score from folder name (e.g., _040100 → 0.40)
        match = re.search(r"_(\d{3})(\d{3})$", folder.name)
        if match:
            min_score_str, _ = match.groups()
            min_score = int(min_score_str) / 100
        else:
            raise ValueError(f"Could not extract min_score from folder name: {folder.name}")


        # Scatter
        plt.scatter(y_true, y_pred, alpha=0.3, s=10)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel(f"True")
        plt.ylabel(f"Predicted")
        # plt.title(f"{target_name} vs Predicted {target_name}")
        plt.xlim(min_score, 1.0)
        plt.ylim(min_score, 1.0)
        plt.grid(True)
        plt.tight_layout()
        scatter_path = plot_dir / f"{model_name}_scatter.pdf"
        plt.savefig(scatter_path, dpi=300)
        plt.close()
        print(f"[Saved] Scatter plot to {scatter_path}")

        # Hexbin
        plt.hexbin(y_true, y_pred, gridsize=60, cmap='viridis', mincnt=1)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.colorbar(label='Count')
        plt.xlabel(f"True")
        plt.ylabel(f"Predicted")
        # plt.title(f"{target_name} Prediction")
        plt.xlim(min_score, 1.0)
        plt.ylim(min_score, 1.0)
        plt.grid(True)
        plt.tight_layout()
        hexbin_path = plot_dir / f"{model_name}_hexbin.pdf"
        plt.savefig(hexbin_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"[Saved] Hexbin plot to {hexbin_path}")

    except Exception as e:
        print(f"[Error] {folder.name}: {e}")

# -------- Generate LaTeX Tables with Bold Best Values --------
for score_str, models in results.items():
    metric_keys = ["R2", "MAE", "MSE", "RMSE"]
    best_values = {
        "ssim": {k: None for k in metric_keys},
        "fsim": {k: None for k in metric_keys}
    }

    # First pass: find best values
    for model, metrics in models.items():
        for metric_type in ["ssim", "fsim"]:
            m = metrics.get(metric_type, {})
            for k in metric_keys:
                v = m.get(k)
                if v is None:
                    continue
                if best_values[metric_type][k] is None:
                    best_values[metric_type][k] = v
                else:
                    if k == "R2":
                        best_values[metric_type][k] = max(best_values[metric_type][k], v)
                    else:
                        best_values[metric_type][k] = min(best_values[metric_type][k], v)

    # Format rows using model_name_map order
    rows = []
    for raw_model in model_name_map:
        model = model_name_map[raw_model]
        metrics = models.get(model, {})
        row = model
        for metric_type in ["ssim", "fsim"]:
            for k in metric_keys:
                v = metrics.get(metric_type, {}).get(k)
                if v is None:
                    row += " & ---"
                else:
                    is_best = np.isclose(v, best_values[metric_type][k])
                    value_str = f"{v:.4f}"
                    if is_best:
                        row += f" & \\textbf{{{value_str}}}"
                    else:
                        row += f" & {value_str}"
        rows.append(row)

    table_lines = [
        "\\begin{tabular}{l|cccc|cccc}",
        "\\toprule",
        "\\multirow{2}{*}{\\textbf{Model}} & \\multicolumn{4}{c|}{\\textbf{SSIM Prediction}} & \\multicolumn{4}{c}{\\textbf{FSIM Prediction}} \\\\",
        " & \\textbf{R2} & \\textbf{MAE} & \\textbf{MSE} & \\textbf{RMSE} & \\textbf{R2} & \\textbf{MAE} & \\textbf{MSE} & \\textbf{RMSE} \\\\",
        "\\midrule"
    ]


    for i, row in enumerate(rows):
        table_lines.append(row + " \\\\")
        if i == 2:
            table_lines.append("\\midrule")

    table_lines += ["\\bottomrule", "\\end{tabular}"]

    table_path = table_dir / f"table_{score_str}.txt"
    with open(table_path, "w") as f:
        f.write("\n".join(table_lines))

    print(f"[Saved] Bold LaTeX table to {table_path}")