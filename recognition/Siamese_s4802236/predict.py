from modules import set_device
import torch
from torchvision import transforms
from pathlib import Path
from dataset import data_loaders
from modules import build_model
from train import BEST_CKPT, evaluate_classifier

device = set_device()

def make_eval_transform(img_size=224):
    norm = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    return transforms.Compose([transforms.Resize((img_size,img_size)), transforms.ToTensor(), norm])

@torch.no_grad()
def _load_clf(device, embed_dim=128, weights_path: Path | None = None):
    clf = build_model(mode="classifier", embed_dim=embed_dim, pretrained=False, freeze_until="none").to(device)
    if weights_path is None:
        # try in-memory checkpoint first
        if 'BEST_CKPT' in globals() and BEST_CKPT is not None:
            clf.load_state_dict(BEST_CKPT, strict=True)
        else:
            raise RuntimeError("No BEST_CKPT in memory and no weights_path provided.")
    else:
        ckpt = torch.load(weights_path, map_location=device)
        clf.load_state_dict(ckpt["state_dict"], strict=True)
    clf.eval()
    return clf

@torch.no_grad()
def run_eval(weights="checkpoints/siamese_classifier.pt", use_in_memory_if_available=True, seed=42):
    device = set_device()
    packs = data_loaders(triplet_mode=False, seed=seed)
    weights_path = None
    if not (use_in_memory_if_available and 'BEST_CKPT' in globals() and BEST_CKPT is not None):
        weights_path = Path(weights)
        if not weights_path.exists():
            raise FileNotFoundError(f"Could not find weights at {weights_path}")
    clf = _load_clf(device, weights_path=weights_path)
    acc, auc = evaluate_classifier(clf, packs["test"], device)
    if auc == auc:
        print(f"\n[RESULT] Test accuracy: {acc:.4f} | Test AUROC: {auc:.4f}")
    else:
        print(f"\n[RESULT] Test accuracy: {acc:.4f}")