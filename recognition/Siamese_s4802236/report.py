from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve
import torch
from dataset import data_loaders
from modules import build_model, set_device
from train import BEST_CKPT, TRAIN_HISTORY



@torch.no_grad()
def collect_probs_targets(model, loader, device):
    model.eval()
    probs = []; targs = []; preds = []
    for images, targets, _ in loader:
        images = images.to(device); targets = targets.to(device)
        logits, _ = model(images)
        p = logits.softmax(dim=1)[:, 1]
        yhat = logits.argmax(dim=1)
        probs.append(p.detach().cpu().numpy())
        targs.append(targets.detach().cpu().numpy())
        preds.append(yhat.detach().cpu().numpy())
    return np.concatenate(probs), np.concatenate(targs), np.concatenate(preds)

def plot_training_curves(H):
    epochs = H["epoch"]
    # Loss
    plt.figure()
    plt.plot(epochs, H["train_loss"], label="Train loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training Loss")
    plt.legend(); plt.grid(True, alpha=0.3); plt.show()
    # Accuracy
    plt.figure()
    plt.plot(epochs, H["train_acc"], label="Train acc")
    plt.plot(epochs, H["val_acc"], label="Val acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy (Train vs Val)")
    plt.legend(); plt.grid(True, alpha=0.3); plt.show()
    # AUROC
    if not np.all(np.isnan(H["train_auc"])) or not np.all(np.isnan(H["val_auc"])):
        plt.figure()
        plt.plot(epochs, H["train_auc"], label="Train AUROC")
        plt.plot(epochs, H["val_auc"], label="Val AUROC")
        plt.xlabel("Epoch"); plt.ylabel("AUROC"); plt.title("AUROC (Train vs Val)")
        plt.legend(); plt.grid(True, alpha=0.3); plt.ylim(0.5, 1.0); plt.show()

def show_confusion(cm, labels=("normal","melanoma"), title="Confusion Matrix"):
    plt.figure()
    im = plt.imshow(cm, interpolation="nearest")
    plt.title(title); plt.colorbar(im, fraction=0.046, pad=0.04)
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45); plt.yticks(ticks, labels)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True label"); plt.xlabel("Predicted label")
    plt.tight_layout(); plt.show()

def plot_roc_pr(y_true, y_prob, split_name):
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"{split_name} ROC")
    plt.legend(); plt.grid(True, alpha=0.3); plt.show()
    # PR
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure()
    plt.plot(rec, prec, label=f"AP={ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"{split_name} Precision-Recall")
    plt.legend(); plt.grid(True, alpha=0.3); plt.show()

device = set_device()

packs = data_loaders(triplet_mode=False, seed=42)
train_loader, val_loader, test_loader = packs["train"], packs["val"], packs["test"]

clf = build_model(mode="classifier", embed_dim=128, pretrained=False, freeze_until="none").to(device)
clf.load_state_dict(BEST_CKPT, strict=True)
clf.eval()

# 1) Training curves
plot_training_curves(TRAIN_HISTORY)

# 2) TRAIN metrics
tr_prob, tr_y, tr_pred = collect_probs_targets(clf, train_loader, device)
tr_acc = (tr_pred == tr_y).mean()
tr_auc = roc_auc_score(tr_y, tr_prob)
print(f"[TRAIN] accuracy={tr_acc:.4f}  AUROC={tr_auc:.4f}")
show_confusion(confusion_matrix(tr_y, tr_pred), title="Train Confusion Matrix")
print("Train classification report:\n", classification_report(tr_y, tr_pred, target_names=["normal","melanoma"]))
plot_roc_pr(tr_y, tr_prob, split_name="Train")

# 3) VAL metrics
va_prob, va_y, va_pred = collect_probs_targets(clf, val_loader, device)
va_acc = (va_pred == va_y).mean()
va_auc = roc_auc_score(va_y, va_prob)
print(f"[VAL]   accuracy={va_acc:.4f}  AUROC={va_auc:.4f}")
show_confusion(confusion_matrix(va_y, va_pred), title="Validation Confusion Matrix")
print("Validation classification report:\n", classification_report(va_y, va_pred, target_names=["normal","melanoma"]))
plot_roc_pr(va_y, va_prob, split_name="Val")

# 4) TEST metrics — headline for your report
te_prob, te_y, te_pred = collect_probs_targets(clf, test_loader, device)
te_acc = (te_pred == te_y).mean()
te_auc = roc_auc_score(te_y, te_prob)
print(f"[TEST]  accuracy={te_acc:.4f}  AUROC={te_auc:.4f}")
show_confusion(confusion_matrix(te_y, te_pred), title="Test Confusion Matrix")
print("Test classification report:\n", classification_report(te_y, te_pred, target_names=["normal","melanoma"]))
plot_roc_pr(te_y, te_prob, split_name="Test")