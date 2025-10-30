import argparse
import torch
from sklearn.metrics import roc_auc_score
from dataset import data_loaders
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

