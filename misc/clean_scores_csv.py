import argparse
import csv
import math
from pathlib import Path


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Read scores.csv, remove rows where at least one metric is NaN/Inf, "
            "and print removal statistics."
        )
    )
    parser.add_argument("csv_path", type=str, help="Path to scores.csv")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path. If omitted, the input file is overwritten.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["ssim", "fsim", "iw_ssim", "sr_sim"],
        help="Metric columns to validate.",
    )
    return parser.parse_args()


def _read_rows(csv_path: Path):
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError(f"CSV header not found in {csv_path}")
        rows = list(reader)
    return fieldnames, rows


def _is_nan_or_inf(value: str) -> tuple[bool, str | None]:
    if value is None:
        return False, None

    text = str(value).strip()
    if text == "":
        return False, None

    try:
        number = float(text)
    except ValueError:
        return False, None

    if math.isnan(number):
        return True, "nan"
    if math.isinf(number):
        return True, "inf"
    return False, None


def clean_scores(csv_path: Path, output_path: Path, metrics: list[str]):
    fieldnames, rows = _read_rows(csv_path)

    missing_metrics = [metric for metric in metrics if metric not in fieldnames]
    if missing_metrics:
        raise ValueError(
            f"Missing metric column(s) in CSV: {', '.join(missing_metrics)}"
        )

    total_rows = len(rows)
    cleaned_rows = []
    removed_rows = 0

    nan_by_metric = {metric: 0 for metric in metrics}
    inf_by_metric = {metric: 0 for metric in metrics}
    invalid_by_metric = {metric: 0 for metric in metrics}

    for row in rows:
        row_has_invalid = False
        for metric in metrics:
            is_invalid, invalid_type = _is_nan_or_inf(row.get(metric))
            if is_invalid:
                row_has_invalid = True
                invalid_by_metric[metric] += 1
                if invalid_type == "nan":
                    nan_by_metric[metric] += 1
                elif invalid_type == "inf":
                    inf_by_metric[metric] += 1

        if row_has_invalid:
            removed_rows += 1
        else:
            cleaned_rows.append(row)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned_rows)

    kept_rows = len(cleaned_rows)
    removed_pct = (removed_rows / total_rows * 100.0) if total_rows > 0 else 0.0

    print(f"Input file: {csv_path}")
    print(f"Output file: {output_path}")
    print(f"Total rows: {total_rows}")
    print(f"Kept rows: {kept_rows}")
    print(f"Removed rows: {removed_rows} ({removed_pct:.2f}% of total)")

    print("\nInvalid values by metric (on total rows):")
    for metric in metrics:
        invalid = invalid_by_metric[metric]
        nan_count = nan_by_metric[metric]
        inf_count = inf_by_metric[metric]
        invalid_pct = (invalid / total_rows * 100.0) if total_rows > 0 else 0.0
        print(
            f"- {metric}: {invalid} invalid ({invalid_pct:.2f}%), "
            f"NaN={nan_count}, Inf={inf_count}"
        )


if __name__ == "__main__":
    args = _parse_args()
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    output_path = Path(args.output) if args.output else csv_path
    clean_scores(csv_path=csv_path, output_path=output_path, metrics=args.metrics)
