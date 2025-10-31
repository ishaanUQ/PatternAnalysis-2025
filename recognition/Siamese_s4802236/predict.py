"""
predict.py
-----------
Load a saved checkpoint and report metrics on train/val/test splits.

Run:
    python predict.py --ckpt checkpoints/siamese_classifier_*.pt --num_workers 0
"""

import argparse
from pathlib import Path
import torch

from dataset import data_loaders_from_disk
from modules import build_model, set_device, evaluate_classifier


def load_classifier(device, embed_dim=128, backbone="resnet18", ckpt_path: Path | None = None):
    """
    Instantiate the classifier architecture and load the given checkpoint.
    """

    clf = build_model(mode="classifier", embed_dim=embed_dim, pretrained=False, freeze_until="none", backbone=backbone).to(device)  # type: ignore[arg-type]
    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    blob = torch.load(ckpt_path, map_location=device)
    state = blob.get("state_dict", blob)
    clf.load_state_dict(state, strict=True)
    clf.eval()
    return clf


def main():
    """
    CLI: build loaders, load model weights, print metrics per split.
    """
    
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--root", type=str, default="data")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18","resnet50"]) 
    args = p.parse_args()

    device = set_device()

    loaders = data_loaders_from_disk(
        clean_root=args.root, force=False, batch_size=args.batch_size,
        num_workers=args.num_workers, img_size=args.img_size, seed=args.seed,
        triplet_mode=False, use_weighted_sampler=False,
    )

    clf = load_classifier(device, embed_dim=args.embed_dim, backbone=args.backbone, ckpt_path=Path(args.ckpt))

    for split in ["train", "val", "test"]:
        acc, auc = evaluate_classifier(clf, loaders[split], device)
        msg = f"[{split.upper()}] accuracy={acc:.4f}"
        if auc == auc:  # not NaN
            msg += f"  AUROC={auc:.4f}"
        print(msg)


if __name__ == "__main__":
    main()
