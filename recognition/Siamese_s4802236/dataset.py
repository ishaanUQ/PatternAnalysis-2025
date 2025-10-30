# dataset.py

### IMPORTS ###
import os, sys, subprocess, shutil, random
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import kagglehub as kh
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
### FUNCTIONS ###
def prepare_isic2020(clean_root: str | Path ="data", force: bool = False) -> tuple[Path, Path]:
    """Prepares the ISIC 2020 224x224 dataset from Kaggle (Colab-friendly)."""
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
    df = df_raw[["isic_id", "target"]].copy()
    df["isic_id"] = df["isic_id"].astype(str) + ".jpg"
    df.rename(columns={"isic_id": "image_name"}, inplace=True)

    images_folder.mkdir(parents=True, exist_ok=True)
    missing = 0
    for name in df["image_name"]:
        src = raw_images_path / name
        dst = images_folder / name
        if src.exists():
            if not dst.exists():
                shutil.copy2(src, dst)
        else:
            missing += 1
    if missing:
        print(f"{missing} images referenced in metadata but not found. Please retry.")

    root.mkdir(parents=True, exist_ok=True)
    df.to_csv(metadata_path, index=False)
    return metadata_path, images_folder

def load_data(metadata_path: Path, images_folder: Path):
    md = pd.read_csv(metadata_path)
    labels_dict = dict(zip(md["image_name"].astype(str), md["target"]))
    image_paths = [images_folder / n for n in md["image_name"] if (images_folder / n).exists()]
    labels = [labels_dict[p.name] for p in image_paths]
    return np.array(image_paths), np.array(labels)

def split_data(images: np.ndarray, labels: np.ndarray, seed: int = 42):
    """0.8 train, 0.1 val, 0.1 test (stratified)."""
    tr_x, te_x, tr_y, te_y = train_test_split(images, labels, test_size=0.2,
                                              random_state=seed, stratify=labels)
    va_x, te_x, va_y, te_y = train_test_split(te_x, te_y, test_size=0.5,
                                              random_state=seed, stratify=te_y)
    return (tr_x, tr_y), (va_x, va_y), (te_x, te_y)


def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def make_transforms(img_size: int = 224):
    norm = transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
    # compact, not copied: HFlip + mild affine + mild color jitter
    train_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
        transforms.ToTensor(),
        norm,
    ])
    eval_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        norm,
    ])
    return train_tfm, eval_tfm
