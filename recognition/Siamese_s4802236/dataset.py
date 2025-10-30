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
from sklearn.model_selection import train_test_split
### FUNCTIONS ###
def prepare_isic2020(clean_root: str | Path ="data", force: bool = False) -> tuple[Path, Path]:
    """ Prepares the ISIC 2020 224x224 dataset from Kaggle"""

    os.environ["KAGGLE_USERNAME"] = "ishaansood1"
    os.environ["KAGGLE_KEY"] = "27613e6479fb99452da36eb26b5c5cb3"

    src_path = kh.dataset_download("nischaydnk/isic-2020-jpg-224x224-resized")
    src_root = Path(src_path)

    root = Path(clean_root)


    raw_images_path = src_root / "train-image" / "image"
    images_folder = root / "train-image" / "image"

    raw_metadata = src_root / "train-metadata.csv"
    metadata_path = root / "train-metadata.csv"

    
    if metadata_path.exists() and images_folder.exists() and not force:
        return metadata_path, images_folder
    
    df_raw = pd.read_csv(raw_metadata, index_col=0)
    print(df_raw.head())
    df = df_raw[["isic_id", "target"]].copy()
    df["isic_id"] = df["isic_id"].astype(str) + ".jpg"
    df.rename(columns={"isic_id": "image_name"}, inplace=True)
    images_folder.mkdir(parents=True, exist_ok=True)
    missing = 0
    for name in df["image_name"]:
        src = raw_images_path / name
        dst = images_folder / name
        if src.exists():
            if not dst.exists():  # Skip if already copied
                shutil.copy2(src, dst)
        else:
            missing += 1
    if missing:
        print(f" {missing} images referenced in metadata but not found. Please retry.")
    
    root.mkdir(parents=True, exist_ok=True)
    print(df.head())
    df.to_csv(metadata_path, index=False)
    return metadata_path, images_folder

def load_data(metadata_path: Path, images_folder: Path) -> tuple[list]:
    metadata = pd.read_csv(metadata_path)
    
    labels_dict = dict(zip(metadata["image_name"].astype(str), metadata["target"]))
    image_paths = [images_folder / n for n in metadata["image_name"] if (images_folder / n).exists()]
    labels = [labels_dict[p.name] for p in image_paths]
    return np.array(image_paths), np.array(labels)

def split_data(images: np.ndarray, labels: np.ndarray, seed: int = 42) -> tuple:
    """ Splits data into 0.8 train, 0.2 test sets """
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.2, random_state=seed, stratify=labels
    )
    return train_images, test_images, train_labels, test_labels

meta_path, images_folder = prepare_isic2020()

image_paths, labels = load_data(meta_path, images_folder)

