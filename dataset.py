import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class SIMDataset(Dataset):
    def __init__(self, root_dir, transform=None, MOS=False):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.MOS = MOS

        # Load scores.csv
        csv_path = self.root_dir / "scores.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing scores.csv in {self.root_dir}")

        self.scores_df = pd.read_csv(csv_path)
        self.image_paths = [
            self.root_dir / row["filename"]
            for _, row in self.scores_df.iterrows()
        ]
        if self.MOS:
            if "mos" not in self.scores_df.columns:
                raise KeyError("Missing 'mos' column in scores.csv")
            self.mos_scores = torch.tensor(self.scores_df["mos"].values, dtype=torch.float32)
        self.metrics = ['ssim', 'fsim', 'ms_ssim', 'iw_ssim', 'sr_sim', 'vsi', 'dss', 'haarpsi', 'mdsi']
        for m in self.metrics:
            if m not in self.scores_df.columns:
                raise KeyError(f"Missing '{m}' column in scores.csv")
            setattr(self, f"{m}_scores", torch.tensor(self.scores_df[m].values, dtype=torch.float32))


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        path = self.image_paths[idx]
        img = Image.open(path).convert('RGB')

        if self.transform:
            img_tensor = self.transform(img)
        else:
            img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0

        # Build metrics dictionary
        scores_row = self.scores_df.iloc[idx]
        metrics_dict = {
            m: torch.tensor(scores_row[m], dtype=torch.float32)
            for m in self.metrics
        }
        if self.MOS:
            metrics_dict['mos'] = torch.tensor(scores_row['mos'], dtype=torch.float32)

        return img_tensor, metrics_dict