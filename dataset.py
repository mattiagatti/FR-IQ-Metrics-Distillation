import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import scipy.io as sio


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

class KonIQDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        

        # Load scores.csv
        csv_path = self.root_dir / "scores.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing scores.csv in {self.root_dir}")

        self.scores_df = pd.read_csv(csv_path)
        self.image_paths = [
            self.root_dir / row["image_name"]
            for _, row in self.scores_df.iterrows()
        ]


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
        mos = torch.tensor(scores_row['MOS'], dtype=torch.float32)

        #normalizaion to 0-1
        mos=(mos-1)/4
        metrics_dict={}
        metrics_dict['mos'] = mos
        return img_tensor, metrics_dict

class LiveItWDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        # Paths
        data_dir = self.root_dir / "Data"
        images_dir = self.root_dir / "Images"

        # ----- Load MAT files -----
        mos_path = data_dir / "AllMOS_release.mat"
        names_path = data_dir / "AllImages_release.mat"

        if not mos_path.exists():
            raise FileNotFoundError(f"Missing {mos_path}")
        if not names_path.exists():
            raise FileNotFoundError(f"Missing {names_path}")

        mos_mat = sio.loadmat(mos_path)
        names_mat = sio.loadmat(names_path)

        # Extract MOS values
        # They are usually stored as Nx1 arrays in MATLAB
        self.mos_list = mos_mat["AllMOS_release"].squeeze().astype(np.float32)
        # Extract image names (cell array → python list)
        raw_names = names_mat["AllImages_release"].squeeze()
        self.image_names = [str(name[0]) for name in raw_names]

        assert len(self.mos_list) == len(self.image_names), \
            "MOS and image name arrays have different lengths!"

        filtered_names = []
        filtered_mos = []

        for name, mos in zip(self.image_names, self.mos_list):
            if not name.lower().startswith("t"):
                filtered_names.append(name)
                filtered_mos.append(mos)

        self.image_names = filtered_names
        self.mos_list = np.array(filtered_mos, dtype=np.float32)

        # Build full paths to images
        self.image_paths = [images_dir / fname for fname in self.image_names]


    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, idx):
        # ----- Load image -----
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")

        if self.transform:
            img_tensor = self.transform(img)
        else:
            img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0

        # ----- Load MOS -----
        mos = torch.tensor(self.mos_list[idx], dtype=torch.float32)

        mos=mos/100.0  # Normalize to 0-1
        metrics_dict = {"mos": mos}

        return img_tensor, metrics_dict

class NRDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        # Load scores.csv
        csv_path = self.root_dir / "scores.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing scores.csv in {self.root_dir}")

        self.scores_df = pd.read_csv(csv_path)
        self.image_paths = [
            self.root_dir / row["image_name"]
            for _, row in self.scores_df.iterrows()
        ]
        
        self.mos_scores = torch.tensor(self.scores_df["MOS"].values, dtype=torch.float32)

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
        metrics_dict={}
        # Build metrics dictionary
        scores_row = self.scores_df.iloc[idx]
        metrics_dict['mos'] = torch.tensor(scores_row['MOS'], dtype=torch.float32)

        return img_tensor, metrics_dict


if __name__ == "__main__":

    dataset = LiveItWDataset(root_dir="./dataset/Live_NR" )
    print(f"Loaded LiveItWDataset with {len(dataset)} items")

    img_tensor, metrics = dataset[0]
    print(f"Sample 0: img_tensor.shape = {tuple(img_tensor.shape)}")
    print("Metrics:", {k: float(v) if isinstance(v, (int, float,)) or (hasattr(v, 'item') and callable(getattr(v, 'item'))) else v for k, v in metrics.items()})