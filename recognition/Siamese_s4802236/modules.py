"""
modules.py
-----------
Models, losses, optimiser/scheduler setup, device helpers, and evaluation utilities.

Shared by train/predict/report to avoid circular imports.
"""

from torchvision import models
import torch.nn as nn
from typing import Tuple, Literal, Optional
import torch.nn.functional as F
import torch
import math
from sklearn.metrics import roc_auc_score, roc_curve

# ── Backbone + encoder ───────────────────────────────────────────────────────

def _load_backbone(name="resnet50", pretrained=True):
    """
    Create a ResNet backbone returning feature vectors (fc replaced by Identity).

    Returns
    -------
    (backbone_module, out_dim)
    """

    if name=="resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None); dim=512
    elif name=="resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None); dim=2048
    else:
        raise ValueError("backbone must be 'resnet18' or 'resnet50'")
    m.fc = nn.Identity()
    return m, dim


class ProjectionHead(nn.Module):
    """Two-layer MLP that outputs L2-normalised embeddings."""

    def __init__(self, in_dim: int = 512, embed_dim: int = 128, p_drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p_drop),
            nn.Linear(embed_dim, embed_dim),
        )
    def forward(self, x):
        # Normalise so dot products become cosine similarities when needed
        return F.normalize(self.net(x), p=2, dim=-1)

class Encoder(nn.Module):
    """
    CNN encoder + projection head.

    freeze_until: optionally freeze early ResNet blocks (transfer learning).
    """

    def __init__(self, embed_dim=128, pretrained=True,
                 freeze_until: Optional[Literal["none","layer1","layer2","layer3"]] = None,
                 backbone: Literal["resnet18","resnet50"] = "resnet18"):
        super().__init__()
        self.backbone, out_dim = _load_backbone(backbone, pretrained=pretrained)
        self.proj = ProjectionHead(in_dim=out_dim, embed_dim=embed_dim)
        if freeze_until and freeze_until != "none":
            self._freeze_resnet_layers(until=freeze_until)

    @torch.no_grad()
    def _freeze_resnet_layers(self, until: Literal["layer1","layer2","layer3"]):
        """Freeze up to (and including) the given stage."""

        def freeze(m):
            for p in m.parameters(): p.requires_grad = False
        freeze(self.backbone.conv1); freeze(self.backbone.bn1)
        freeze(self.backbone.layer1)
        if until in ("layer2", "layer3"): freeze(self.backbone.layer2)
        if until == "layer3": freeze(self.backbone.layer3)

    def forward(self, x):
        feats = self.backbone(x)
        return self.proj(feats)
    
# ── Siamese models ───────────────────────────────────────────────────────────

class SiameseTripletNet(nn.Module):
    """
    Siamese network used for triplet loss pretraining.

    forward(A, P, N) returns (ZA, ZP, ZN) embeddings.
    """

    def __init__(self, encoder: Encoder):
        super().__init__(); self.encoder = encoder
    def forward(self, A, P, N):
        B = A.shape[0]
        Z = self.encoder(torch.cat([A, P, N], dim=0))
        return torch.split(Z, B, dim=0)

class SiameseClassifier(nn.Module):
    """
    Classifier head on top of the encoder’s embedding space.
    """

    def __init__(self, encoder: Encoder, num_classes: int = 2):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(self.encoder.proj.net[-1].out_features, num_classes)
        nn.init.kaiming_normal_(self.head.weight, nonlinearity="linear")
        if self.head.bias is not None: nn.init.zeros_(self.head.bias)
    def forward(self, x):
        z = self.encoder(x)
        return self.head(z), z

# ── Loss / Optim / Sched ─────────────────────────────────────────────────────
class TripletMarginLossWrapper(nn.Module):
    """Thin wrapper so the loss can be constructed via config."""

    def __init__(self, margin: float = 0.3, p: float = 2.0, swap: bool = False):
        super().__init__(); self.loss = nn.TripletMarginLoss(margin=margin, p=p, swap=swap)
    def forward(self, zA, zP, zN): return self.loss(zA, zP, zN)

def build_model(mode: Literal["triplet","classifier"] = "triplet", embed_dim=128,
                pretrained=True, freeze_until="layer2", num_classes=2, backbone="resnet18"):
    
    """Factory for either the triplet or classifier model."""

    enc = Encoder(embed_dim=embed_dim, pretrained=pretrained, freeze_until=freeze_until, backbone=backbone)
    if mode=="triplet":    return SiameseTripletNet(enc)
    if mode=="classifier": return SiameseClassifier(enc, num_classes=num_classes)
    raise ValueError("Unknown mode")


def make_optimizer(model: nn.Module, lr=3e-4, weight_decay=1e-4, head_lr_mult=2.0):

    """
    AdamW with parameter groups:
    * backbone at base lr
    * projection and (optional) classifier head at head_lr_mult × base lr
    """

    enc = model.encoder if hasattr(model, "encoder") else model
    params = []
    bb = [p for p in enc.backbone.parameters() if p.requires_grad]
    if bb: params.append({"params": bb, "lr": lr})
    pj = [p for p in enc.proj.parameters() if p.requires_grad]
    if pj: params.append({"params": pj, "lr": lr * head_lr_mult})
    if hasattr(model, "head"):
        hd = [p for p in model.head.parameters() if p.requires_grad]
        if hd: params.append({"params": hd, "lr": lr * head_lr_mult})
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def make_scheduler(optimizer, warmup_epochs=1, total_epochs=15):
    """
    Cosine decay with a linear warmup (epoch-wise LambdaLR).
    """

    def lr_lambda(cur_epoch):
        if cur_epoch < warmup_epochs:
            return float(cur_epoch + 1) / float(max(1, warmup_epochs))
        t = (cur_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * t))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

# ── Device + evaluation helpers ──────────────────────────────────────────────

def set_device():
    """Prefer CUDA, then Apple MPS, otherwise CPU."""

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate_classifier(model: nn.Module, loader, device) -> tuple[float, float]:
    """
    Compute accuracy and AUROC over a loader.
    """

    model.eval()
    correct = 0; total = 0
    all_probs = []; all_targets = []
    with torch.no_grad():
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
    try:
        auc = roc_auc_score(targs, probs)
    except ValueError:
        auc = float("nan")
    return acc, auc


@torch.no_grad()
def collect_probs_targets(model: nn.Module, loader, device):
    """
    Return concatenated (probabilities, targets, predictions) for a loader.
    Useful for ROC/PR curves and confusion matrices.
    """
    
    model.eval()
    import numpy as np
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