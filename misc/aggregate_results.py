import os
import re
import pandas as pd
from pathlib import Path

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
EXP_DIR = Path("exp")  # root directory containing all experiment folders
OUTPUT_DIR = Path("exp_summary")
OUTPUT_DIR.mkdir(exist_ok=True)

# Regex to extract model and metric from folder name: model_metric_range
pattern = re.compile(r"^(?P<model>[^_]+)_(?P<metric>[^_]+)")

# Prepare containers
data_epoch = {}
data_r2 = {}
data_mae = {}
data_rmse = {}

# -------------------------------------------------------------------
# Parse all experiments
# -------------------------------------------------------------------
for exp_folder in sorted(EXP_DIR.iterdir()):
    if not exp_folder.is_dir():
        continue
    result_file = exp_folder / "best_results.txt"
    if not result_file.exists():
        print(f"⚠️ Skipping {exp_folder.name}: no best_results.txt")
        continue

    # Extract model + metric from folder name
    m = pattern.match(exp_folder.name)
    if not m:
        print(f"⚠️ Skipping {exp_folder.name}: unexpected name format")
        continue

    model = m.group("model")
    metric = m.group("metric")

    # Read file contents
    content = result_file.read_text().strip().splitlines()
    values = {}
    for line in content:
        if "Best Epoch" in line:
            values["epoch"] = int(line.split(":")[1].strip())
        elif "Best R2" in line:
            values["r2"] = float(line.split(":")[1].strip())
        elif "Best MAE" in line:
            values["mae"] = float(line.split(":")[1].strip())
        elif "Best RMSE" in line:
            values["rmse"] = float(line.split(":")[1].strip())

    # Store in dictionaries (rows=metrics, columns=models)
    data_epoch.setdefault(metric, {})[model] = values.get("epoch", None)
    data_r2.setdefault(metric, {})[model] = values.get("r2", None)
    data_mae.setdefault(metric, {})[model] = values.get("mae", None)
    data_rmse.setdefault(metric, {})[model] = values.get("rmse", None)

# -------------------------------------------------------------------
# Convert to DataFrames and export
# -------------------------------------------------------------------
def save_table(data, name):
    df = pd.DataFrame(data).sort_index().T.sort_index(axis=1)
    out_path = OUTPUT_DIR / f"{name}.csv"
    df.to_csv(out_path)
    print(f"✅ Saved {out_path}")

save_table(data_epoch, "best_epoch")
save_table(data_r2, "best_r2")
save_table(data_mae, "best_mae")
save_table(data_rmse, "best_rmse")

print("✅ Aggregation completed successfully.")