# train.py

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

from dataset import data_loaders
from models import (
    build_model,
    TripletMarginLossWrapper,
    make_optimizer,
    make_scheduler,
)

# Optional: AUROC
try:
    from sklearn.metrics import roc_auc_score
    HAVE_SK = True
except ImportError:
    HAVE_SK = False


def set_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate_classifier(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    all_probs = []
    all_targets = []

    for images, targets, _ in loader:
        images = images.to(device)
        targets = targets.to(device)
        logits, _ = model(images)
        probs = logits.softmax(dim=1)[:, 1]
        preds = logits.argmax(dim=1)

        correct += (preds == targets).sum().item()
        total += targets.numel()

        all_probs.append(probs.detach().cpu())
        all_targets.append(targets.detach().cpu())

    acc = correct / max(1, total)
    if HAVE_SK:
        import numpy as np
        probs = torch.cat(all_probs).numpy()
        targs = torch.cat(all_targets).numpy()
        try:
            auc = roc_auc_score(targs, probs)
        except ValueError:
            auc = float("nan")
    else:
        auc = None
    return acc, auc


def train_triplet(
    epochs,
    device,
    loaders,
    embed_dim=128,
    freeze_until="layer2",
    lr=3e-4,
    weight_decay=1e-4,
):
    print(f"\n[Stage 1] Triplet pretraining for {epochs} epoch(s)")
    model = build_model(
        mode="triplet",
        embed_dim=embed_dim,
        pretrained=True,
        freeze_until=freeze_until,
    ).to(device)

    criterion = TripletMarginLossWrapper(margin=0.3)
    optimizer = make_optimizer(model, lr=lr, weight_decay=weight_decay)
    scheduler = make_scheduler(optimizer, warmup_epochs=1, total_epochs=max(epochs, 2))
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type=="cuda"))

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        for A, P, N, _ in loaders["train"]:
            A = A.to(device, non_blocking=True)
            P = P.to(device, non_blocking=True)
            N = N.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda")):
                zA, zP, zN = model(A, P, N)
                loss = criterion(zA, zP, zN)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()

        scheduler.step()
        dt = time.time() - t0
        print(f"  Epoch {epoch+1:02d}/{epochs} | triplet loss: {running/len(loaders['train']):.4f} | {dt:.1f}s")

    return model  # SiameseTripletNet


def train_classifier(
    epochs,
    device,
    loaders,
    triplet_model=None,
    embed_dim=128,
    freeze_until="layer2",
    lr=3e-4,
    weight_decay=1e-4,
    save_path="checkpoints/siamese_classifier.pt",
    patience=5,
):
    print(f"\n[Stage 2] Classifier fine-tune for {epochs} epoch(s)")
    clf = build_model(
        mode="classifier",
        embed_dim=embed_dim,
        pretrained=True,            # start from ImageNet
        freeze_until=freeze_until,
        num_classes=2,
    ).to(device)

    # If we have a pretrained encoder from triplet, copy its weights in
    if triplet_model is not None:
        clf.encoder.load_state_dict(triplet_model.encoder.state_dict(), strict=False)
        print("  Loaded encoder weights from triplet pretrain.")

    # Weighted CE from loaders (dataset.py returns these)
    criterion = nn.CrossEntropyLoss(
        weight=loaders["train_cls_weights"].to(device)
    )

    optimizer = make_optimizer(clf, lr=lr, weight_decay=weight_decay)
    scheduler = make_scheduler(optimizer, warmup_epochs=1, total_epochs=max(epochs, 2))
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type=="cuda"))

    best_val = -1.0
    es_left = patience
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        clf.train()
        t0 = time.time()
        running = 0.0

        for images, targets, _ in loaders["train"]:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda")):
                logits, _ = clf(images)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()

        scheduler.step()
        train_loss = running / max(1, len(loaders["train"]))

        # Validate
        val_acc, val_auc = evaluate_classifier(clf, loaders["val"], device)
        dt = time.time() - t0
        if val_auc is not None:
            print(f"  Epoch {epoch+1:02d}/{epochs} | loss {train_loss:.4f} | val_acc {val_acc:.4f} | val_auc {val_auc:.4f} | {dt:.1f}s")
        else:
            print(f"  Epoch {epoch+1:02d}/{epochs} | loss {train_loss:.4f} | val_acc {val_acc:.4f} | {dt:.1f}s")

        # Early stopping + checkpointing
        if val_acc > best_val:
            best_val = val_acc
            es_left = patience
            torch.save(
                {
                    "state_dict": clf.state_dict(),
                    "embed_dim": embed_dim,
                    "freeze_until": freeze_until,
                },
                save_path,
            )
            print(f"  ↳ saved new best to {save_path} (val_acc={best_val:.4f})")
        else:
            es_left -= 1
            if es_left <= 0:
                print("  Early stopping.")
                break

    # Load best and evaluate on test
    if save_path.exists():
        ckpt = torch.load(save_path, map_location=device)
        clf.load_state_dict(ckpt["state_dict"], strict=True)

    test_acc, test_auc = evaluate_classifier(clf, loaders["test"], device)
    if test_auc is not None:
        print(f"\n[RESULT] Test accuracy: {test_acc:.4f} | Test AUROC: {test_auc:.4f}")
    else:
        print(f"\n[RESULT] Test accuracy: {test_acc:.4f}")

    return clf


def main():
    parser = argparse.ArgumentParser()
    # Data & training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    # Stages
    parser.add_argument("--triplet_epochs", type=int, default=4)
    parser.add_argument("--clf_epochs", type=int, default=10)
    parser.add_argument("--freeze_until", type=str, default="layer2", choices=["none", "layer1", "layer2", "layer3"])
    parser.add_argument("--embed_dim", type=int, default=128)

    # Optim
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # Save
    parser.add_argument("--save_path", type=str, default="checkpoints/siamese_classifier.pt")

    args = parser.parse_args()

    # Stage 1 uses triplet loader; Stage 2 uses normal loader
    print("Preparing data loaders…")
    loaders_triplet = data_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        seed=args.seed,
        triplet_mode=True,
    )
    loaders_clf = data_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        seed=args.seed,
        triplet_mode=False,
        use_weighted_sampler=True,
    )

    device = set_device()
    print(f"Using device: {device}")

    # Stage 1: triplet pretraining (optional if epochs=0)
    triplet_model = None
    if args.triplet_epochs > 0:
        triplet_model = train_triplet(
            epochs=args.triplet_epochs,
            device=device,
            loaders=loaders_triplet,
            embed_dim=args.embed_dim,
            freeze_until=args.freeze_until,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    # Stage 2: classifier fine-tune
    _ = train_classifier(
        epochs=args.clf_epochs,
        device=device,
        loaders=loaders_clf,
        triplet_model=triplet_model,
        embed_dim=args.embed_dim,
        freeze_until=args.freeze_until,
        lr=args.lr,
        weight_decay=args.weight_decay,
        save_path=args.save_path,
        patience=5,
    )


if __name__ == "__main__":
    main()
