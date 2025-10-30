from torchvision import models
import torch.nn as nn
from typing import Tuple, Literal, Optional
import torch.nn.functional as F
import torch
import math
def _load_backbone(name="resnet50", pretrained=True):
    if name=="resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None); dim=512
    elif name=="resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None); dim=2048
    else:
        raise ValueError("backbone must be 'resnet18' or 'resnet50'")
    m.fc = nn.Identity()
    return m, dim

class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int = 512, embed_dim: int = 128, p_drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p_drop),
            nn.Linear(embed_dim, embed_dim),
        )
    def forward(self, x): return F.normalize(self.net(x), p=2, dim=-1)

class Encoder(nn.Module):
    def __init__(self, embed_dim=128, pretrained=True,
                freeze_until: Optional[Literal["none","layer1","layer2","layer3"]] = None,
                backbone: Literal["resnet18","resnet50"]="resnet18"):
        super().__init__()
        self.backbone, out_dim = _load_backbone(backbone, pretrained=pretrained)
        self.proj = ProjectionHead(in_dim=out_dim, embed_dim=embed_dim)
        if freeze_until and freeze_until != "none":
            self._freeze_resnet_layers(until=freeze_until)

    @torch.no_grad()
    def _freeze_resnet_layers(self, until: Literal["layer1","layer2","layer3"]):
        def freeze(m):
            for p in m.parameters(): p.requires_grad = False
        freeze(self.backbone.conv1); freeze(self.backbone.bn1)
        freeze(self.backbone.layer1)
        if until in ("layer2","layer3"): freeze(self.backbone.layer2)
        if until == "layer3": freeze(self.backbone.layer3)

    def forward(self, x):
        feats = self.backbone(x)
        return self.proj(feats)
class SiameseTripletNet(nn.Module):
    def __init__(self, encoder: Encoder):
        super().__init__(); self.encoder = encoder
    def forward(self, A, P, N):
        B = A.shape[0]
        Z = self.encoder(torch.cat([A,P,N], dim=0))
        return torch.split(Z, B, dim=0)

class SiameseClassifier(nn.Module):
    def __init__(self, encoder: Encoder, num_classes: int = 2):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(self.encoder.proj.net[-1].out_features, num_classes)
        nn.init.kaiming_normal_(self.head.weight, nonlinearity="linear")
        if self.head.bias is not None: nn.init.zeros_(self.head.bias)
    def forward(self, x):
        z = self.encoder(x)
        return self.head(z), z
class TripletMarginLossWrapper(nn.Module):
    def __init__(self, margin: float = 0.3, p: float = 2.0, swap: bool = False):
        super().__init__(); self.loss = nn.TripletMarginLoss(margin=margin, p=p, swap=swap)
    def forward(self, zA, zP, zN): return self.loss(zA, zP, zN)

def build_model(mode: Literal["triplet","classifier"]="triplet", embed_dim=128,
                pretrained=True, freeze_until="layer2", num_classes=2, backbone="resnet18"):
    enc = Encoder(embed_dim=embed_dim, pretrained=pretrained, freeze_until=freeze_until, backbone=backbone)
    if mode=="triplet":    return SiameseTripletNet(enc)
    if mode=="classifier": return SiameseClassifier(enc, num_classes=num_classes)
    raise ValueError("Unknown mode")

def make_optimizer(model: nn.Module, lr=3e-4, weight_decay=1e-4, head_lr_mult=2.0):
    enc = model.encoder if hasattr(model, "encoder") else model
    params = []
    bb = [p for p in enc.backbone.parameters() if p.requires_grad]
    if bb: params.append({"params": bb, "lr": lr})
    pj = [p for p in enc.proj.parameters() if p.requires_grad]
    if pj: params.append({"params": pj, "lr": lr*head_lr_mult})
    if hasattr(model, "head"):
        hd = [p for p in model.head.parameters() if p.requires_grad]
        if hd: params.append({"params": hd, "lr": lr*head_lr_mult})
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

def make_scheduler(optimizer, warmup_epochs=1, total_epochs=15):
    def lr_lambda(cur_epoch):
        if cur_epoch < warmup_epochs:
            return float(cur_epoch + 1) / float(max(1, warmup_epochs))
        t = (cur_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * t))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)