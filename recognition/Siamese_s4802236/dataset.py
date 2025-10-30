# dataset.py

### IMPORTS ###
import os
import pandas as pd
import kagglehub as kh
from pathlib import Path
import shutil
import random, numpy as np
import torch, torch.nn as nn
from torchvision import transforms

### FUNCTIONS ###
def prepare_isic2020(clean_root: str | Path ="data", force: bool = False) -> tuple[Path, Path]:
    """ Prepares the ISIC 2020 224x224 dataset from Kaggle"""

    os.environ["KAGGLE_USERNAME"] = "ishaansood1"
    os.environ["KAGGLE_KEY"] = "27613e6479fb99452da36eb26b5c5cb3"

    src_path = kh.dataset_download("nischaydnk/isic-2020-jpg-224x224-resized")
    src_root = Path(src_path)

    root = Path(clean_root)


    raw_images_path = src_root / "train-image" / "image"
    images_path = root / "train-image" / "image"

    raw_metadata = src_root / "train-metadata.csv"
    metadata_path = root / "train-metadata.csv"

    
    if metadata_path.exists() and images_path.exists() and not force:
        return metadata_path, images_path
    
    df_raw = pd.read_csv(raw_metadata, index_col=0)
    print(df_raw.head())
    df = df_raw[["isic_id", "target"]].copy()
    df["isic_id"] = df["isic_id"].astype(str) + ".jpg"
    df.rename(columns={"isic_id": "image_name"}, inplace=True)
    images_path.mkdir(parents=True, exist_ok=True)
    missing = 0
    for name in df["image_name"]:
        src = raw_images_path / name
        dst = images_path / name
        if src.exists():
            if not dst.exists():  # Skip if already copied
                shutil.copy2(src, dst)
        else:
            missing += 1
    if missing:
        print(f" {missing} images referenced in metadata but not found. Please retry.")
    
    root.mkdir(parents=True, exist_ok=True)
    print(df.head())
    return metadata_path, images_path

def load_data(metadata_path: str, images_path: str | Path, subset: int | None=None) -> tuple[list]:
    pass




meta_path, img_dir = prepare_isic2020()
