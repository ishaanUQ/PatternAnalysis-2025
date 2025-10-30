import argparse
import time
import torch
from sklearn.metrics import roc_auc_score
from dataset import data_loaders
from modules import TripletMarginLossWrapper, build_model, make_optimizer, make_scheduler
import torch.nn as nn
def set_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--img_size", type=int, default=224)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--triplet_epochs", type=int, default=3)
parser.add_argument("--clf_epochs", type=int, default=10)
parser.add_argument("--freeze_until", type=str, default="layer2", choices=["none","layer1","layer2","layer3"])
parser.add_argument("--embed_dim", type=int, default=128)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--weight_decay", type=float, default=1e-4)
args = parser.parse_args(args=[])

device = set_device()
print(f"Using device: {device}")

print("Preparing data loaders…")
loaders_triplet = data_loaders(batch_size=args.batch_size, num_workers=args.num_workers,
                               img_size=args.img_size, seed=args.seed, triplet_mode=True)
loaders_clf = data_loaders(batch_size=args.batch_size, num_workers=args.num_workers,
                           img_size=args.img_size, seed=args.seed, triplet_mode=False, use_weighted_sampler=True)


@torch.no_grad()
def evaluate_classifier(model, loader, device):
    model.eval()
    correct = 0; total = 0
    all_probs = []; all_targets = []
    for images, targets, _ in loader:
        images = images.to(device); targets = targets.to(device)
        logits, _ = model(images)
        probs = logits.softmax(dim=1)[:, 1]
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.numel()
        all_probs.append(probs.detach().cpu())
        all_targets.append(targets.detach().cpu())
    acc = correct / max(1, total)
    probs = torch.cat(all_probs).numpy()
    targs = torch.cat(all_targets).numpy()
    try: auc = roc_auc_score(targs, probs)
    except ValueError: auc = float("nan")
    return acc, auc

def train_triplet(epochs, device, loaders, embed_dim=128, freeze_until="layer2", lr=3e-4, weight_decay=1e-4):
    print(f"\n[Stage 1] Triplet pretraining for {epochs} epoch(s)")
    model = build_model(mode="triplet", embed_dim=embed_dim, pretrained=True, freeze_until=freeze_until).to(device)
    criterion = TripletMarginLossWrapper(margin=0.3)
    optimizer = make_optimizer(model, lr=lr, weight_decay=weight_decay)
    scheduler = make_scheduler(optimizer, warmup_epochs=1, total_epochs=max(epochs, 2))
    scaler = torch.GradScaler("cuda", enabled=(device.type=="cuda"))
    for ep in range(epochs):
        model.train(); t0 = time.time(); running = 0.0
        for A, P, N, _ in loaders["train"]:
            A = A.to(device, non_blocking=True); P = P.to(device, non_blocking=True); N = N.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
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
):
    print(f"\n[Stage 2] Classifier fine-tune for {epochs} epoch(s)")
    clf = build_model(mode="classifier", embed_dim=embed_dim, pretrained=True,
                      freeze_until=freeze_until, num_classes=2).to(device)
    if triplet_model is not None:
        clf.encoder.load_state_dict(triplet_model.encoder.state_dict(), strict=False)
        print("  Loaded encoder weights from triplet pretrain.")

    criterion = nn.CrossEntropyLoss(weight=loaders["train_cls_weights"].to(device))
    optimizer = make_optimizer(clf, lr=lr, weight_decay=weight_decay)
    scheduler = make_scheduler(optimizer, warmup_epochs=1, total_epochs=max(epochs, 2))
    scaler = torch.GradScaler("cuda", enabled=(device.type=="cuda"))

    best_val = -1.0
    es_left = patience
    history = {"epoch": [], "train_loss": [], "train_acc": [], "train_auc": [], "val_acc": [], "val_auc": []}
    best_state = None

    for ep in range(epochs):
        clf.train(); t0 = time.time(); running = 0.0
        tr_correct = 0; tr_total = 0
        tr_probs = []; tr_targs = []
        for images, targets, _ in loaders["train"]:
            images = images.to(device, non_blocking=True); targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                logits, _ = clf(images)
                loss = criterion(logits, targets)

            # online train metrics
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
        try: train_auc = roc_auc_score(torch.cat(tr_targs).numpy(), torch.cat(tr_probs).numpy())
        except ValueError: train_auc = float("nan")

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

    # load best weights into clf
    if best_state is not None:
        clf.load_state_dict(best_state, strict=True)

    return clf, history, best_state


triplet_model = None
if args.triplet_epochs > 0:
    triplet_model = train_triplet(epochs=args.triplet_epochs, device=device, loaders=loaders_triplet,
                                embed_dim=args.embed_dim, freeze_until=args.freeze_until,
                                lr=args.lr, weight_decay=args.weight_decay)

clf, TRAIN_HISTORY, BEST_CKPT = train_classifier(
    epochs=args.clf_epochs, device=device, loaders=loaders_clf, triplet_model=triplet_model,
    embed_dim=args.embed_dim, freeze_until=args.freeze_until, lr=args.lr, weight_decay=args.weight_decay,
    patience=5,
)

print("Training complete. Globals set: TRAIN_HISTORY (dict), BEST_CKPT (state_dict).")