"""
train.py
---------
End-to-end pipeline:
1) (Optional) triplet pretraining,
2) classifier fine-tuning,
3) test evaluation,
4) save checkpoint + training history,
5) (Optional) immediately run prediction metrics using the saved checkpoint.

Run:
    python train.py --num_workers _ --run_predict
"""

import argparse, time, json
from datetime import datetime
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from typing import Optional, Tuple, Literal
import torch
import torch.nn as nn
from dataset import data_loaders_from_disk
from modules import (
    TripletMarginLossWrapper, build_model, make_optimizer, make_scheduler,
    set_device, evaluate_classifier,
)

# ── Training stages ──────────────────────────────────────────────────────────

def train_triplet(epochs, device, loaders, embed_dim=128, freeze_until="layer2", lr=3e-4, weight_decay=1e-4, backbone="resnet18"):
    """
    Stage 1: metric-learning pretraining with triplet loss.
    """

    print(f"[Stage 1] Triplet pretraining for {epochs} epoch(s)")
    model = build_model(mode="triplet", embed_dim=embed_dim, pretrained=True, freeze_until=freeze_until, backbone=backbone).to(device)
    criterion = TripletMarginLossWrapper(margin=0.3)
    optimizer = make_optimizer(model, lr=lr, weight_decay=weight_decay)
    scheduler = make_scheduler(optimizer, warmup_epochs=1, total_epochs=max(epochs, 2))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    for ep in range(epochs):
        model.train(); t0 = time.time(); running = 0.0
        for A, P, N, _ in loaders["train"]:
            A = A.to(device, non_blocking=True); P = P.to(device, non_blocking=True); N = N.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                zA, zP, zN = model(A, P, N)
                loss = criterion(zA, zP, zN)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            running += loss.item()
        scheduler.step()
        print(f"  Epoch {ep+1:02d}/{epochs} | triplet loss: {running/len(loaders['train']):.4f} | {time.time()-t0:.1f}s")
    return model


def train_classifier(
    epochs,
    device,
    loaders,
    triplet_model=None,
    embed_dim=128,
    freeze_until="layer2",
    lr=3e-4,
    weight_decay=1e-4,
    patience=5,
    backbone="resnet18",
):
    """
    Stage 2: classifier fine-tuning with class-balanced loss and early stopping.
    """

    print(f"[Stage 2] Classifier fine-tune for {epochs} epoch(s)")
    clf = build_model(mode="classifier", embed_dim=embed_dim, pretrained=True,
                      freeze_until=freeze_until, num_classes=2, backbone=backbone).to(device)
    
    # Reuse encoder weights from triplet stage, if available
    if triplet_model is not None:
        clf.encoder.load_state_dict(triplet_model.encoder.state_dict(), strict=False)
        print("  Loaded encoder weights from triplet pretrain.")

    criterion = nn.CrossEntropyLoss(weight=loaders["train_cls_weights"].to(device))
    optimizer = make_optimizer(clf, lr=lr, weight_decay=weight_decay)
    scheduler = make_scheduler(optimizer, warmup_epochs=1, total_epochs=max(epochs, 2))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type=="cuda"))

    best_val = -1.0
    es_left = patience
    history = {"epoch": [], "train_loss": [], "train_acc": [], "train_auc": [], "val_acc": [], "val_auc": []}
    best_state = None

    from sklearn.metrics import roc_auc_score

    for ep in range(epochs):
        clf.train(); t0 = time.time(); running = 0.0
        tr_correct = 0; tr_total = 0
        tr_probs = []; tr_targs = []
        for images, targets, _ in loaders["train"]:
            images = images.to(device, non_blocking=True); targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                logits, _ = clf(images)
                loss = criterion(logits, targets)
            with torch.no_grad():
                probs = logits.softmax(dim=1)[:, 1]
                preds = logits.argmax(dim=1)
                tr_correct += (preds == targets).sum().item()
                tr_total += targets.numel()
                tr_probs.append(probs.detach().cpu()); tr_targs.append(targets.detach().cpu())
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            running += loss.item()
        scheduler.step()

        train_loss = running / max(1, len(loaders["train"]))
        train_acc = tr_correct / max(1, tr_total)
        try:
            train_auc = roc_auc_score(torch.cat(tr_targs).numpy(), torch.cat(tr_probs).numpy())
        except ValueError:
            train_auc = float("nan")

        val_acc, val_auc = evaluate_classifier(clf, loaders["val"], device)

        history["epoch"].append(ep+1)
        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["train_auc"].append(float(train_auc))
        history["val_acc"].append(float(val_acc))
        history["val_auc"].append(float(val_auc))

        print(f"  Epoch {ep+1:02d}/{epochs} | loss {train_loss:.4f} | "
              f"train_acc {train_acc:.4f} | train_auc {train_auc:.4f} | "
              f"val_acc {val_acc:.4f} | val_auc {val_auc:.4f} | {time.time()-t0:.1f}s")

        # Early stopping on validation accuracy
        if val_acc > best_val:
            best_val = val_acc
            es_left = patience
            best_state = {k: v.detach().cpu().clone() for k, v in clf.state_dict().items()}
            print(f"  ↳ new best (val_acc={best_val:.4f})")
        else:
            es_left -= 1
            if es_left <= 0:
                print("  Early stopping.")
                break

    if best_state is not None:
        clf.load_state_dict(best_state, strict=True)

    return clf, history, best_state


# ── Predict hook (called after training) ─────────────────────────────────────

def run_predict_from_train(ckpt_path: Path, root: str, img_size: int, num_workers: int, batch_size: int, seed: int, embed_dim: int, backbone: str):
    """
    In-process version of predict.py so train.py can evaluate freshly saved weights
    across train/val/test without spawning a new process.
    """

    from modules import build_model, set_device, evaluate_classifier
    from dataset import data_loaders_from_disk

    device = set_device()
    loaders = data_loaders_from_disk(
        clean_root=root, force=False, batch_size=batch_size, num_workers=num_workers,
        img_size=img_size, seed=seed, triplet_mode=False, use_weighted_sampler=False,
    )

    clf = build_model(mode="classifier", embed_dim=embed_dim, pretrained=False, freeze_until="none", backbone=backbone).to(device)
    blob = torch.load(ckpt_path, map_location=device)
    state = blob.get("state_dict", blob)
    clf.load_state_dict(state, strict=True)
    clf.eval()

    for split in ["train", "val", "test"]:
        acc, auc = evaluate_classifier(clf, loaders[split], device)
        msg = f"[PREDICT from train • {split.upper()}] accuracy={acc:.4f}"
        if auc == auc:
            msg += f"  AUROC={auc:.4f}"
        print(msg)

# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    """
    Parse CLI flags, run the full pipeline, save outputs, and optionally run prediction.
    """

    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--triplet_epochs", type=int, default=3)
    p.add_argument("--clf_epochs", type=int, default=10)
    p.add_argument("--freeze_until", type=str, default="layer2", choices=["none","layer1","layer2","layer3"])
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18","resnet50"])
    p.add_argument("--root", type=str, default="data")
    p.add_argument("--force_prep", action="store_true")
    p.add_argument("--out_dir", type=str, default="checkpoints")
    p.add_argument("--run_predict", action="store_true", help="After saving the checkpoint, also run full predict (train/val/test) using it")
    args = p.parse_args()

    device = set_device()
    print(f"Using device: {device}")

    print("Preparing data loaders…")
    loaders_triplet = data_loaders_from_disk(
        clean_root=args.root, force=args.force_prep, batch_size=args.batch_size, num_workers=args.num_workers,
        img_size=args.img_size, seed=args.seed, triplet_mode=True,
    )
    loaders_clf = data_loaders_from_disk(
        clean_root=args.root, force=args.force_prep, batch_size=args.batch_size, num_workers=args.num_workers,
        img_size=args.img_size, seed=args.seed, triplet_mode=False, use_weighted_sampler=True,
    )

    triplet_model = None
    if args.triplet_epochs > 0:
        triplet_model = train_triplet(
            epochs=args.triplet_epochs, device=device, loaders=loaders_triplet, embed_dim=args.embed_dim,
            freeze_until=args.freeze_until, lr=args.lr, weight_decay=args.weight_decay, backbone=args.backbone,
        )

    clf, history, best_state = train_classifier(
        epochs=args.clf_epochs, device=device, loaders=loaders_clf, triplet_model=triplet_model,
        embed_dim=args.embed_dim, freeze_until=args.freeze_until, lr=args.lr, weight_decay=args.weight_decay,
        patience=5, backbone=args.backbone,
    )

    # Final test metrics on the classification loaders
    te_acc, te_auc = evaluate_classifier(clf, loaders_clf["test"], device)
    print(f"[RESULT] Test accuracy: {te_acc:.4f} | Test AUROC: {te_auc:.4f}")

    # Save checkpoint + history
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path = out_dir / f"siamese_classifier_{stamp}.pt"
    torch.save({
        "state_dict": clf.state_dict(),
        "meta": {"test_acc": float(te_acc), "test_auc": float(te_auc), "args": vars(args)},
    }, ckpt_path)
    with open(out_dir / f"train_history_{stamp}.json", "w") as f:
        json.dump(history, f)
    print(f"Saved checkpoint → {ckpt_path}")

    # Optional: immediately evaluate the saved checkpoint
    if args.run_predict:
        print("[RUN PREDICT] Evaluating saved checkpoint across train/val/test…")
        run_predict_from_train(
            ckpt_path=ckpt_path, root=args.root, img_size=args.img_size, num_workers=args.num_workers,
            batch_size=args.batch_size, seed=args.seed, embed_dim=args.embed_dim, backbone=args.backbone,
        )


if __name__ == "__main__":
    main()
