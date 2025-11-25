import os
import re
import pandas as pd
from pathlib import Path

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
RESULTS_DIRS_TO_COMBINE = [
    Path("/home/jovyan/python/Neural-No-Reference-SIM/test_results/Live_Combined_IQA"),
    Path("/home/jovyan/python/Neural-No-Reference-SIM/test_results/Live_our"),
    Path("/home/jovyan/python/Neural-No-Reference-SIM/test_results/Live_their"),
    Path("/home/jovyan/python/Neural-No-Reference-SIM/test_results/Live_TIDtheir"),
    Path("/home/jovyan/python/Neural-No-Reference-SIM/test_results/Live_Combined_IQA_MOS")
]

OUTPUT_DIR = Path("/home/jovyan/python/Neural-No-Reference-SIM/test_results/Live_summary")
OUTPUT_DIR.mkdir(exist_ok=True)

# pattern per estrarre la metrica
pattern = r"results_[^_]+_([^_]+(?:_[^_]+)?)_MOS\.txt"

# Tabelle finali: righe = experiment_name, colonne = metric
r2_table = {}
spearman_table = {}
kendall_table = {}

# -------------------------------------------------------------------
# Parse all experiments
# -------------------------------------------------------------------
for exp in RESULTS_DIRS_TO_COMBINE:

    experiment_name = exp.name       # esempio: KonIQ-10k_our
    r2_table.setdefault(experiment_name, {})
    spearman_table.setdefault(experiment_name, {})
    kendall_table.setdefault(experiment_name, {})

    for result_file in sorted(exp.iterdir()):
        if not result_file.name.startswith("results"):
            continue

        m = re.search(pattern, result_file.name)
        if not m:
            print(f"⚠️ Skipping {result_file.name}: unexpected format")
            continue

        metric = m.group(1)  # esempio: ssim, ms_ssim, vsi etc.

        # Leggi valori
        content = result_file.read_text().splitlines()

        r2 = spearman = kendall = None
        for line in content:
            if "R2 Score" in line:
                r2 = float(line.split(":")[1].strip())
            elif "Spearman" in line:
                spearman = float(line.split(":")[1].strip())
            elif "Kendall" in line:
                kendall = float(line.split(":")[1].strip())

        r2_table[experiment_name][metric] = r2
        spearman_table[experiment_name][metric] = spearman
        kendall_table[experiment_name][metric] = kendall

# -------------------------------------------------------------------
# Convert tables to DataFrames and export CSV
# -------------------------------------------------------------------
def save_table(data_dict, filename):
    df = pd.DataFrame.from_dict(data_dict, orient="index")
    df = df.sort_index(axis=0).sort_index(axis=1)  # ordina righe e colonne
    df.to_csv(OUTPUT_DIR / filename)
    print(f"✅ Saved {filename}")

save_table(r2_table, "R2_table.csv")
save_table(spearman_table, "Spearman_table.csv")
save_table(kendall_table, "Kendall_table.csv")

print("🎉 Aggregation completed successfully!")