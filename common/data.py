"""Shared data code. Everyone must use this so the hold-out split and preprocessing are identical."""
import os
import random
from collections import defaultdict

from PIL import Image
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

ROOT = os.environ.get("CS604_ROOT", "/content/cs604-course-data")
NUM_CLASSES = 10000

# Team-wide constants: do NOT change without telling everyone (results stop being comparable).
HOLDOUT_PER_CLASS = 2      # labelled train images per class kept out of training for local scoring (20k total)
SPLIT_SEED = 0

MEAN = STD = (0.5, 0.5, 0.5)   # timm vit_base_patch16_224.augreg2_in21k_ft_in1k
BICUBIC = T.InterpolationMode.BICUBIC


def read_train(root=ROOT):
    """train.txt -> [(image_path, class_id)]"""
    with open(f"{root}/train.txt") as f:
        return [(p, int(c)) for p, c in (line.split() for line in f)]


def read_paths(split, root=ROOT):
    """val.txt / test.txt -> [image_path]  (labels are withheld for these splits)"""
    with open(f"{root}/{split}.txt") as f:
        return [line.split()[0] for line in f if line.strip()]


def split_holdout(rows, per_class=HOLDOUT_PER_CLASS, seed=SPLIT_SEED):
    """Deterministic per-class hold-out. Independent of file order. Returns (train_rows, holdout_rows)."""
    by_class = defaultdict(list)
    for r in rows:
        by_class[r[1]].append(r)
    train, hold = [], []
    for c in sorted(by_class):
        items = sorted(by_class[c])
        random.Random(f"{seed}-{c}").shuffle(items)
        hold += items[:per_class]
        train += items[per_class:]
    return train, hold


def train_tf(scale_min=0.35):
    return T.Compose([
        T.RandomResizedCrop(224, scale=(scale_min, 1.0), interpolation=BICUBIC),
        T.RandomHorizontalFlip(),
        T.ToTensor(), T.Normalize(MEAN, STD),
    ])


def eval_tf():
    return T.Compose([
        T.Resize(248, interpolation=BICUBIC), T.CenterCrop(224),
        T.ToTensor(), T.Normalize(MEAN, STD),
    ])


class ImageDS(Dataset):
    """rows: [(path, label)] or [path] (label -1)."""
    def __init__(self, rows, tf, root=ROOT):
        self.rows, self.tf, self.root = rows, tf, root

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        p, y = (r, -1) if isinstance(r, str) else r
        return self.tf(Image.open(f"{self.root}/{p}").convert("RGB")), y


def make_loader(rows, tf, batch_size, shuffle=False, workers=None, root=ROOT, drop_last=False):
    workers = os.cpu_count() if workers is None else workers
    return DataLoader(ImageDS(rows, tf, root), batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), drop_last=drop_last,
                      prefetch_factor=4 if workers > 0 else None)
