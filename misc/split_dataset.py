import pandas as pd
import shutil
import random
from pathlib import Path
from tqdm import tqdm

def split_dataset(
    input_dir,
    output_dir,
    csv_name="scores.csv",
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42
):
    random.seed(seed)

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    csv_path = input_dir / csv_name

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Load scores.csv
    df = pd.read_csv(csv_path)

    # Shuffle
    file_list = df["filename"].tolist()
    random.shuffle(file_list)

    # Split indices
    n = len(file_list)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_files = file_list[:n_train]
    val_files = file_list[n_train:n_train + n_val]
    test_files = file_list[n_train + n_val:]

    print(f"Total images: {n}")
    print(f"Train: {len(train_files)}")
    print(f"Val:   {len(val_files)}")
    print(f"Test:  {len(test_files)}")

    # Make output directories
    train_dir = output_dir / "train"
    val_dir   = output_dir / "val"
    test_dir  = output_dir / "test"

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Copy files
    def copy_files(file_list, dest_dir):
        dest_scores = []
        for fname in tqdm(file_list, desc=f"Copying to {dest_dir.name}"):
            src = input_dir / fname
            dst = dest_dir / fname
            shutil.copy(src, dst)

            # Add row to new CSV
            row = df[df["filename"] == fname].iloc[0]
            dest_scores.append(row)

        # Save new CSV
        new_df = pd.DataFrame(dest_scores)
        new_df.to_csv(dest_dir / "scores.csv", index=False)

    copy_files(train_files, train_dir)
    copy_files(val_files, val_dir)
    copy_files(test_files, test_dir)

    print("\n✅ Dataset successfully split into train/val/test.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Split dataset into train/val/test sets")
    parser.add_argument("--input_dir", required=True, help="Directory containing images + scores.csv")
    parser.add_argument("--output_dir", required=True, help="Output directory for the split dataset")

    args = parser.parse_args()

    split_dataset(args.input_dir, args.output_dir)