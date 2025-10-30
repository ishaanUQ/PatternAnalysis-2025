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
class SkinLesionDataset(Dataset):
    """(image, label, path)."""
    def __init__(self, image_paths, labels, tfm=None):
        self.paths = image_paths
        self.labels = labels.astype(int)
        self.tfm = tfm
    def __len__(self): return len(self.paths)
    def __getitem__(self, i: int):
        p = self.paths[i]
        img = Image.open(p).convert("RGB")
        if self.tfm: img = self.tfm(img)
        return img, int(self.labels[i]), str(p)

class TripletSet(Dataset):
    """
    Triplet dataset where the DataLoader index is the *anchor index*.
    We then sample:
    - a positive from the same class (different index),
    - a negative from the other class.
    This makes it compatible with WeightedRandomSampler over anchors.
    """
    def __init__(self, image_paths, labels, tfm=None):
        self.paths  = image_paths
        self.labels = labels.astype(int)
        self.tfm    = tfm

        # precompute indices by class for fast sampling
        self.by_class = {
            0: np.where(self.labels == 0)[0].tolist(),
            1: np.where(self.labels == 1)[0].tolist(),
        }

    def __len__(self):
        # one anchor per image -> sampler can reweight/oversample
        return len(self.paths)

    def _load(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.tfm(img) if self.tfm else transforms.ToTensor()(img)

    def __getitem__(self, anchor_idx: int):
        y_anchor = int(self.labels[anchor_idx])

        # choose a positive (same class, different index)
        pos_idx = anchor_idx
        same_pool = self.by_class[y_anchor]
        if len(same_pool) > 1:
            while pos_idx == anchor_idx:
                pos_idx = random.choice(same_pool)

        # choose a negative (other class)
        other_class = 1 - y_anchor
        neg_idx = random.choice(self.by_class[other_class])

        A = self._load(anchor_idx)
        P = self._load(pos_idx)
        N = self._load(neg_idx)
        return A, P, N, y_anchor

def class_weights(labels: np.ndarray) -> torch.Tensor:
    vals, counts = np.unique(labels, return_counts=True)
    freqs = counts / counts.sum()
    w = {int(c): float(1.0 / f) for c, f in zip(vals, freqs)}
    return torch.tensor([w[0], w[1]], dtype=torch.float32)

def make_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    vals, counts = np.unique(labels, return_counts=True)
    freq = {int(v): float(c) for v, c in zip(vals, counts)}
    weights = np.array([1.0 / freq[int(y)] for y in labels], dtype=np.float32)
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)

def make_weighted_anchor_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """
    For triplet training: per-anchor weights ∝ 1 / class_freq(anchor_class).
    This balances anchor selection across classes.
    """
    vals, counts = np.unique(labels, return_counts=True)
    freq = {int(v): float(c) for v, c in zip(vals, counts)}
    weights = np.array([1.0 / freq[int(y)] for y in labels], dtype=np.float32)
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)

def data_loaders(
    batch_size: int = 64,
    num_workers: int = 8,
    img_size: int = 224,
    seed: int = 42,
    use_weighted_sampler: bool = True,
    triplet_mode: bool = False,
) -> dict[str, object]:
    """
    Builds DataLoaders for either:
      - triplet_mode=True: TripletSet with optional weighted *anchor* sampling, or
      - triplet_mode=False: standard classification loaders with optional weighted sampling.

    Returns:
      {
        "train": DataLoader,
        "val":   DataLoader,
        "test":  DataLoader,
        "train_cls_weights": torch.Tensor,
        "splits": {"train": (paths,labels), "val": (...), "test": (...)}
      }
    """
    set_seed(seed)
    (tr_x, tr_y), (va_x, va_y), (te_x, te_y) = split_data(image_paths, labels, seed=seed)
    train_tfm, eval_tfm = make_transforms(img_size)

    if triplet_mode:
        # ---- Triplet pretraining branch ----
        # Dataset where the DataLoader-provided index is the *anchor* index.
        train_set = TripletSet(tr_x, tr_y, tfm=train_tfm)

        if use_weighted_sampler:
            # Weighted anchors ∝ 1 / class_freq(anchor_class)
            sampler = make_weighted_anchor_sampler(tr_y)
            train_loader = DataLoader(
                train_set,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
            )
        else:
            train_loader = DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
            )

        # Val/Test are plain classification batches for metrics
        val_set  = SkinLesionDataset(va_x, va_y, tfm=eval_tfm)
        test_set = SkinLesionDataset(te_x, te_y, tfm=eval_tfm)
        val_loader  = DataLoader(val_set,  batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=True)
        cls_w = class_weights(tr_y)

    else:
        # ---- Classification branch ----
        train_set = SkinLesionDataset(tr_x, tr_y, tfm=train_tfm)
        val_set   = SkinLesionDataset(va_x, va_y, tfm=eval_tfm)
        test_set  = SkinLesionDataset(te_x, te_y, tfm=eval_tfm)

        if use_weighted_sampler:
            sampler = make_weighted_sampler(tr_y)
            train_loader = DataLoader(
                train_set,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
            )
        else:
            train_loader = DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
            )

        val_loader  = DataLoader(val_set,  batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=True)
        cls_w = class_weights(tr_y)

    return {
        "train": train_loader,
        "val":   val_loader,
        "test":  test_loader,
        "train_cls_weights": cls_w,
        "splits": {
            "train": (tr_x, tr_y),
            "val":   (va_x, va_y),
            "test":  (te_x, te_y),
        },
    }


meta_path, images_folder = prepare_isic2020()

image_paths, labels = load_data(meta_path, images_folder)
print(f"Loaded {len(image_paths)} images.")