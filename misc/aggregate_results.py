import os
import re
import pandas as pd
from pathlib import Path

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
RESULTS_DIRS_TO_COMBINE = [
    Path("./test_results/CSIQ_our"),
]

OUTPUT_DIR = Path("./test_results/CSIQ_summary")
OUTPUT_DIR.mkdir(exist_ok=True)

# pattern per estrarre la metrica
# pattern per estrarre modello e metrica (es: results_efficientnet_dss.txt)
pattern = r"results_([^_]+)_([^_]+(?:_[^_]+)?)\_MOS.txt"

# Tabelle finali per modello: righe = experiment_name, colonne = evaluation metric
# Struttura: tables_by_model[model][experiment_name][metric] = value
r2_tables_by_model = {}
spearman_tables_by_model = {}
kendall_tables_by_model = {}
MSE_tables_by_model = {}

# -------------------------------------------------------------------
# Parse all experiments
# -------------------------------------------------------------------
for exp in RESULTS_DIRS_TO_COMBINE:

    experiment_name = exp.name       # esempio: KonIQ-10k_our

    for result_file in sorted(exp.iterdir()):
        if not result_file.name.startswith("results"):
            continue

        m = re.search(pattern, result_file.name)
        if not m:
            print(f"⚠️ Skipping {result_file.name}: unexpected format")
            continue

        model = m.group(1)   # esempio: efficientnet
        metric = m.group(2)  # esempio: ssim, ms_ssim, vsi etc.

        if metric == "haarpsi" or metric == "mdsi":
            continue
        # Leggi valori
        content = result_file.read_text().splitlines()

        r2 = spearman = kendall = mse = None
        for line in content:
            if "R2 Score" in line:
                r2 = float(line.split(":")[1].strip())
            elif "Spearman" in line:
                spearman = float(line.split(":")[1].strip())
            elif "Kendall" in line:
                kendall = float(line.split(":")[1].strip())
            elif line.startswith("MSE"):
                mse = float(line.split(":")[1].strip())

        # Inizializza le tabelle per il modello se non presenti
        r2_tables_by_model.setdefault(model, {}).setdefault(experiment_name, {})[metric] = r2
        spearman_tables_by_model.setdefault(model, {}).setdefault(experiment_name, {})[metric] = spearman
        kendall_tables_by_model.setdefault(model, {}).setdefault(experiment_name, {})[metric] = kendall
        MSE_tables_by_model.setdefault(model, {}).setdefault(experiment_name, {})[metric] = mse

# -------------------------------------------------------------------
# Convert tables to DataFrames and export CSV
# -------------------------------------------------------------------
def save_combined_tables_per_model(r2_dict, spearman_dict, kendall_dict, mse_dict):
    """
    Salva un file CSV per modello contenente tutte e 4 le metriche di valutazione
    disposte verticalmente come nella figura
    """
    all_models = set(r2_dict.keys()) | set(spearman_dict.keys()) | set(kendall_dict.keys()) | set(mse_dict.keys())
    
    for model in sorted(all_models):
        # Crea DataFrames per ogni metrica
        r2_df = pd.DataFrame.from_dict(r2_dict.get(model, {}), orient="index").sort_index(axis=0).sort_index(axis=1)
        spearman_df = pd.DataFrame.from_dict(spearman_dict.get(model, {}), orient="index").sort_index(axis=0).sort_index(axis=1)
        kendall_df = pd.DataFrame.from_dict(kendall_dict.get(model, {}), orient="index").sort_index(axis=0).sort_index(axis=1)
        mse_df = pd.DataFrame.from_dict(mse_dict.get(model, {}), orient="index").sort_index(axis=0).sort_index(axis=1)
        
        # Aggiungi una colonna con il nome della metrica di valutazione
        r2_df.insert(0, '', 'R2')
        spearman_df.insert(0, '', 'Spearman')
        kendall_df.insert(0, '', 'Kendall')
        mse_df.insert(0, '', 'MSE')
        
        # Concatena verticalmente con una riga vuota tra le sezioni
        combined_df = pd.concat([
            r2_df,
            pd.DataFrame([[''] * len(r2_df.columns)], columns=r2_df.columns),
            kendall_df,
            pd.DataFrame([[''] * len(kendall_df.columns)], columns=kendall_df.columns),
            mse_df,
            pd.DataFrame([[''] * len(mse_df.columns)], columns=mse_df.columns),
            spearman_df
        ], ignore_index=False)
        
        # Salva il file
        filename = f"results_{model}.csv"
        combined_df.to_csv(OUTPUT_DIR / filename)
        print(f"✅ Saved {filename}")

save_combined_tables_per_model(r2_tables_by_model, spearman_tables_by_model, kendall_tables_by_model, MSE_tables_by_model)

print("🎉 Aggregation completed successfully!")