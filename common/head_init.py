"""Classifier-head initialisation from class means (nearest-class-mean classifier).

logit_c = s * (f . mu_c - ||mu_c||^2 / 2)   ==  s * (-0.5 * ||f - mu_c||^2 + const)
so the initial head predicts the nearest class mean in feature space. This is the 'zero-shot with
initialised head' baseline of the brief (reference: 43.7% top-1) and a much better start than a random head.

The raw logits (s = 1) are very large, which makes cross-entropy over-confident at the start of training.
s="auto" picks the scale that minimises leave-one-out cross-entropy on the images used for the means.
The scale does not change which class wins (zero-shot accuracy is unchanged), only how confident the start is.
"""
import torch
import torch.nn.functional as F

from common.data import eval_tf, make_loader

SCALE_GRID = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)


@torch.no_grad()
def calibrate_scale(feats, labels, mu, counts, grid=SCALE_GRID, n=20000, chunk=1000):
    """Leave-one-out: each sample's own class mean is recomputed without that sample."""
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(labels), generator=g)[:n].to(feats.device)
    f, y = feats[idx], labels[idx]
    sums = mu * counts[:, None]
    mu_loo = (sums[y] - f) / (counts[y] - 1).clamp(min=1)[:, None]
    mu2 = (mu ** 2).sum(1)
    losses = {s: 0.0 for s in grid}
    for a in range(0, len(y), chunk):
        fb, yb, ml = f[a:a + chunk], y[a:a + chunk], mu_loo[a:a + chunk]
        logits = fb @ mu.T - 0.5 * mu2
        logits[torch.arange(len(yb)), yb] = (fb * ml).sum(1) - 0.5 * (ml ** 2).sum(1)
        for s in grid:
            losses[s] += F.cross_entropy(s * logits, yb, reduction="sum").item()
    best = min(losses, key=losses.get)
    return best, {s: l / len(y) for s, l in losses.items()}


@torch.no_grad()
def ncm_head_init(model, train_rows, device, per_class=10, batch_size=256, workers=None, amp=None, scale="auto"):
    by_class = {}
    for p, c in train_rows:
        if len(by_class.setdefault(c, [])) < per_class:
            by_class[c].append((p, c))
    rows = [r for c in sorted(by_class) for r in by_class[c]]

    model.eval().to(device)
    dim, n_cls = model.head.in_features, model.head.out_features
    feats, labels = [], []
    for x, y in make_loader(rows, eval_tf(), batch_size, workers=workers):
        with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
            f = model.forward_head(model.forward_features(x.to(device, non_blocking=True)), pre_logits=True)
        feats.append(f.float()); labels.append(y.to(device))
    feats, labels = torch.cat(feats), torch.cat(labels)

    sums = torch.zeros(n_cls, dim, device=device).index_add_(0, labels, feats)
    counts = torch.zeros(n_cls, device=device).index_add_(0, labels, torch.ones_like(labels, dtype=torch.float))
    mu = sums / counts.clamp(min=1)[:, None]

    if scale == "auto":
        if counts.min() < 2:                       # leave-one-out needs at least 2 images per class
            scale = 1.0
        else:
            scale, losses = calibrate_scale(feats, labels, mu, counts)
            print("head-init scale (leave-one-out CE):", {s: round(l, 3) for s, l in losses.items()}, "-> chose", scale)
    model.head.weight.copy_(scale * mu)
    model.head.bias.copy_(-0.5 * scale * (mu ** 2).sum(1))
    return len(rows), scale
