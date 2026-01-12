import pandas as pd
import numpy as np
from pathlib import Path
import re
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr, kendalltau

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
EXPERIMENT_DIRS = [
    Path("/home/jovyan/nfs/lsgroi/exp/exp_IMAGENET"),
    Path("/home/jovyan/nfs/lsgroi/exp/exp_IMAGENET_04"),
    Path("/home/jovyan/nfs/lsgroi/exp/exp_IMAGENET_07")
]

OUTPUT_DIR = Path("./test_results/IMAGENET_summary_Validation")
OUTPUT_DIR.mkdir(exist_ok=True)

# Metriche supportate
ALL_METRICS = ['ssim', 'fsim', 'ms_ssim', 'iw_ssim', 'sr_sim', 'vsi', 'dss']

# Pattern per estrarre modello, metrica e range dal nome della cartella
# Esempio: efficientnet_ssim_040100
pattern = r"^([^_]+)_([^_]+(?:_[^_]+)?)_(\d{6})$"

# Struttura: tables_by_model[model][experiment_name][metric] = value
r2_tables_by_model = {}
spearman_tables_by_model = {}
kendall_tables_by_model = {}
mse_tables_by_model = {}

# -------------------------------------------------------------------
# Parse all experiments
# -------------------------------------------------------------------
for base_exp_dir in EXPERIMENT_DIRS:
    if not base_exp_dir.exists():
        print(f"⚠️ Directory not found: {base_exp_dir}")
        continue
    
    experiment_name = base_exp_dir.name  # es: IMAGENET_our_our
    
    # Itera su tutte le sottocartelle (una per modello/metrica/range)
    for exp_folder in base_exp_dir.iterdir():
        if not exp_folder.is_dir():
            continue
        
        csv_path = exp_folder / "predictions.csv"
        if not csv_path.exists():
            print(f"[Skip] No predictions.csv in {exp_folder}")
            continue
        
        # Parse del nome della cartella
        m = re.search(pattern, exp_folder.name)
        if not m:
            print(f"⚠️ Skipping {exp_folder.name}: unexpected format")
            continue
        
        model = m.group(1).lower()
        metric = m.group(2).lower()
        score_range = m.group(3)  # es: 040100
        
        
        
        # Salta metriche non supportate
        if metric not in ALL_METRICS:
            print(f"[Skip] Unsupported metric: {metric} in {exp_folder.name}")
            continue
        
        # Determina le colonne da usare
        
        true_col, pred_col = f"true_{metric}", f"predicted_{metric}"
        
        try:
            df = pd.read_csv(csv_path)
            
            # Preferisci colonne denormalizzate se disponibili
            denorm_true_col = f"denorm_{true_col}"
            denorm_pred_col = f"denorm_{pred_col}"
            
            if denorm_true_col in df.columns and denorm_pred_col in df.columns:
                y_true = df[denorm_true_col].values
                y_pred = df[denorm_pred_col].values
            elif true_col in df.columns and pred_col in df.columns:
                y_true = df[true_col].values
                y_pred = df[pred_col].values
            else:
                print(f"[Skip] Missing required columns [{denorm_true_col}, {denorm_pred_col}, {true_col}, {pred_col}] in {csv_path}")
                continue
            
            # Calcola le metriche
            r2 = r2_score(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            spearman, _ = spearmanr(y_true, y_pred)
            kendall, _ = kendalltau(y_true, y_pred)
            
            # Salva i risultati nelle tabelle
            r2_tables_by_model.setdefault(model, {}).setdefault(experiment_name, {})[metric] = r2
            spearman_tables_by_model.setdefault(model, {}).setdefault(experiment_name, {})[metric] = spearman
            kendall_tables_by_model.setdefault(model, {}).setdefault(experiment_name, {})[metric] = kendall
            mse_tables_by_model.setdefault(model, {}).setdefault(experiment_name, {})[metric] = mse
            
            print(f"[Processed] {experiment_name}/{exp_folder.name}: R2={r2:.4f}, MSE={mse:.6f}")
            
        except Exception as e:
            print(f"[Error] {exp_folder.name}: {e}")

# -------------------------------------------------------------------
# Save results to CSV files (one per model)
# -------------------------------------------------------------------
def save_combined_tables_per_model(r2_dict, spearman_dict, kendall_dict, mse_dict):
    """
    Salva un file CSV per modello contenente tutte e 4 le metriche di valutazione
    disposte verticalmente
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

save_combined_tables_per_model(r2_tables_by_model, spearman_tables_by_model, kendall_tables_by_model, mse_tables_by_model)

print("🎉 Aggregation from predictions.csv completed successfully!")
