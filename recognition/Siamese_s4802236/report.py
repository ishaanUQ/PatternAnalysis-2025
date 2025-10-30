from matplotlib import pyplot as plt
import numpy as np
import torch
from modules import set_device


device = set_device()

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