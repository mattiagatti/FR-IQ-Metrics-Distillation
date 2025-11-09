import csv
import math

csv_path = "scores.csv"

metrics_nan_count = {}
metrics_list = []

with open(csv_path, newline='') as f:
    reader = csv.DictReader(f)
    metrics_list = [col for col in reader.fieldnames if col != "filename"]

    # inizializza contatori
    for m in metrics_list:
        metrics_nan_count[m] = 0

    # leggi tutte le righe e conta i NaN
    for row in reader:
        for m in metrics_list:
            val = row[m]

            # condizioni per NaN:
            # - vuoto
            # - "nan" (case insensitive)
            # - non convertibile in float
            # - float NaN
            if val.strip() == "":
                metrics_nan_count[m] += 1
                continue

            try:
                x = float(val)
                if math.isnan(x):
                    metrics_nan_count[m] += 1
            except:
                metrics_nan_count[m] += 1

# stampa risultati
print("\n=== NaN count per metrica ===")
for m in metrics_list:
    print(f"{m:10s}: {metrics_nan_count[m]}")