# ─────────────────────────────────────────────────────────────────────────────
# report.py
# ─────────────────────────────────────────────────────────────────────────────

import argparse, json
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import (
    average_precision_score, classification_report, confusion_matrix,
    precision_recall_curve, roc_auc_score, roc_curve,
)

from dataset import data_loaders_from_disk
from modules import build_model, set_device, collect_probs_targets


def plot_training_curves(H):
    epochs = H["epoch"]
    plt.figure(); plt.plot(epochs, H["train_loss"], label="Train loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training Loss"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.figure(); plt.plot(epochs, H["train_acc"], label="Train acc"); plt.plot(epochs, H["val_acc"], label="Val acc"); plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy (Train vs Val)"); plt.legend(); plt.grid(True, alpha=0.3)
    if (not np.all(np.isnan(H.get("train_auc", []))) if len(H.get("train_auc", []))>0 else False) or (not np.all(np.isnan(H.get("val_auc", []))) if len(H.get("val_auc", []))>0 else False):
        plt.figure(); plt.plot(epochs, H.get("train_auc", []), label="Train AUROC"); plt.plot(epochs, H.get("val_auc", []), label="Val AUROC"); plt.xlabel("Epoch"); plt.ylabel("AUROC"); plt.title("AUROC (Train vs Val)"); plt.legend(); plt.grid(True, alpha=0.3); plt.ylim(0.5, 1.0)
    plt.show()


def show_confusion(cm, labels=("normal","melanoma"), title="Confusion Matrix"):
    plt.figure(); im = plt.imshow(cm, interpolation="nearest"); plt.title(title); plt.colorbar(im, fraction=0.046, pad=0.04)
    ticks = np.arange(len(labels)); plt.xticks(ticks, labels, rotation=45); plt.yticks(ticks, labels)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color=("white" if cm[i, j] > thresh else "black"))
    plt.ylabel("True label"); plt.xlabel("Predicted label"); plt.tight_layout(); plt.show()


def plot_roc_pr(y_true, y_prob, split_name):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(); plt.plot(fpr, tpr, label=f"AUC={auc:.3f}"); plt.plot([0,1],[0,1],'--'); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"{split_name} ROC"); plt.legend(); plt.grid(True, alpha=0.3)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure(); plt.plot(rec, prec, label=f"AP={ap:.3f}"); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"{split_name} Precision-Recall"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.show()


def load_classifier(device, embed_dim=128, backbone="resnet18", ckpt_path: Path | None = None):
    clf = build_model(mode="classifier", embed_dim=embed_dim, pretrained=False, freeze_until="none", backbone=backbone).to(device)  # type: ignore[arg-type]
    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    blob = torch.load(ckpt_path, map_location=device)
    state = blob.get("state_dict", blob)
    clf.load_state_dict(state, strict=True)
    clf.eval()
    return clf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--history", type=str, default=None, help="Optional JSON history saved by train.py")
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

    if args.history and Path(args.history).exists():
        with open(args.history, "r") as f:
            H = json.load(f)
        plot_training_curves(H)

    clf = load_classifier(device, embed_dim=args.embed_dim, backbone=args.backbone, ckpt_path=Path(args.ckpt))

    # Train/Val/Test metrics + plots
    for split in ["train", "val", "test"]:
        y_prob, y_true, y_pred = collect_probs_targets(clf, loaders[split], device)
        acc = (y_pred == y_true).mean()
        auc = roc_auc_score(y_true, y_prob)
        print(f"[{split.upper()}] accuracy={acc:.4f}  AUROC={auc:.4f}")
        show_confusion(confusion_matrix(y_true, y_pred), title=f"{split.title()} Confusion Matrix")
        print(f"{split.title()} classification report:\n", classification_report(y_true, y_pred, target_names=["normal","melanoma"]))
        plot_roc_pr(y_true, y_prob, split_name=split.title())


if __name__ == "__main__":
    main()