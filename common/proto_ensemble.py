"""Head + class-mean ensemble: score a finished run with its trained head AND a nearest-class-mean classifier on the
adapted features, combined at test time. Nothing is trained; no trainable parameters are added.

    python -m common.proto_ensemble runs/<name>                      # at the training size
    python -m common.proto_ensemble runs/<name> --size 288           # at a larger test size
    python -m common.proto_ensemble runs/<name> --per-class 10       # quick check: class means from 10 images per class

Why: the trained head overfits first (H9, H10). Class means of the adapted features are a second classifier with no
trainable parameters, so they cannot overfit the 45 images per class (similar in spirit to Tip-Adapter's cache model).
Combined score = log_softmax(head logits) + w * log_softmax(cosine(feature, class mean) / T). w = 0 is the plain model.

Honest scoring: w and T are chosen on one half of the hold-out and reported on the other half (and the other way
round); the "cross-fitted" number is the average of the two. The full-hold-out grid is printed too, but picking the
best cell from it is optimistic. Cost at test time: one extra 768 x 10,000 matrix product (the class means are stored
like a second head, 7.7M values, not trained).
"""
import argparse
import sys
import time

import torch
import torch.nn.functional as F
import yaml

from common import data
from common.build import build_model
from common.train import pick_amp

W_GRID = (0.0, 0.25, 0.5, 1.0, 2.0)
T_GRID = (0.02, 0.05, 0.1)


@torch.no_grad()
def feats_and_logits(model, x, amp, device):
    with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
        f = model.forward_head(model.forward_features(x), pre_logits=True)
        logits = model.head(f)
    return f.float(), logits.float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--size", type=int, help="test image size, multiple of 16 (default: the training size)")
    ap.add_argument("--per-class", type=int, default=0, help="training images per class for the means (0 = all)")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(f"{a.run_dir}/config.yaml"))
    if not cfg["holdout_per_class"]:
        sys.exit("this run trained on every labelled image (holdout_per_class=0), so there is no hold-out to score it on")
    train_size = cfg.get("train_size", 224)
    size = a.size or train_size
    if size % 16:
        sys.exit("size must be a multiple of 16 (the patch size)")
    cfg["dynamic_img_size"] = size != train_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = pick_amp(cfg["precision"], device)
    torch.manual_seed(cfg.get("seed", 0))
    model = build_model(cfg).to(device).eval()
    _, unexpected = model.load_state_dict(torch.load(f"{a.run_dir}/final.pt", map_location="cpu"), strict=False)
    assert not unexpected, unexpected

    rows = data.read_train()
    if cfg.get("debug_classes"):
        rows = [r for r in rows if r[1] < cfg["debug_classes"]]
    train_rows, hold = data.split_holdout(rows, cfg["holdout_per_class"])
    if a.per_class:                          # keep the first N training images of each class
        kept, seen = [], {}
        for r in train_rows:
            seen[r[1]] = seen.get(r[1], 0) + 1
            if seen[r[1]] <= a.per_class:
                kept.append(r)
        train_rows = kept
    n_cls = model.head.out_features
    bs, workers = cfg["eval_batch_size"], cfg["num_workers"]
    tf = data.eval_tf(size)
    print(f"[{cfg['name']}] {size} px | class means from {len(train_rows)} training images | hold-out {len(hold)}")

    # 1. class means of the L2-normalised adapted features (training images only; the hold-out is never used here)
    t0 = time.time()
    sums = None
    counts = torch.zeros(n_cls, device=device)
    for x, y in data.make_loader(train_rows, tf, bs, workers=workers):
        f, _ = feats_and_logits(model, x.to(device, non_blocking=True), amp, device)
        f = F.normalize(f, dim=1)
        y = y.to(device)
        sums = torch.zeros(n_cls, f.shape[1], device=device) if sums is None else sums
        sums.index_add_(0, y, f)
        counts.index_add_(0, y, torch.ones_like(y, dtype=torch.float))
    means = F.normalize(sums / counts.clamp(min=1)[:, None], dim=1)
    print(f"class means computed in {(time.time() - t0) / 60:.1f} min")

    # 2. hold-out: correct counts for every (w, T) cell, separately for the two halves (even / odd index)
    correct = torch.zeros(2, len(W_GRID), len(T_GRID))
    proto_only = torch.zeros(2)
    n_half = torch.zeros(2)
    i0 = 0
    for x, y in data.make_loader(hold, tf, bs, workers=workers):
        f, logits = feats_and_logits(model, x.to(device, non_blocking=True), amp, device)
        y = y.to(device)
        half = (torch.arange(i0, i0 + len(y), device=device) % 2)
        i0 += len(y)
        cos = F.normalize(f, dim=1) @ means.T
        lh = F.log_softmax(logits, dim=1)
        proto_hit = (cos.argmax(1) == y).float()
        for h in (0, 1):
            m = half == h
            n_half[h] += m.sum().item()
            proto_only[h] += proto_hit[m].sum().item()
        for ti, T in enumerate(T_GRID):
            lp = F.log_softmax(cos / T, dim=1)
            for wi, w in enumerate(W_GRID):
                hit = ((lh + w * lp).argmax(1) == y).float()
                for h in (0, 1):
                    correct[h, wi, ti] += hit[half == h].sum().item()

    acc = 100 * correct / n_half[:, None, None]                 # per half
    full = 100 * correct.sum(0) / n_half.sum()                  # whole hold-out
    print(f"\nhead only (w = 0): {full[0, 0]:.2f}%   class means only: {100 * proto_only.sum() / n_half.sum():.2f}%")
    print("whole hold-out, top-1 % (rows w, columns T) - picking the best cell here is optimistic:")
    print("w \\ T  " + "  ".join(f"{T:>6}" for T in T_GRID))
    for wi, w in enumerate(W_GRID):
        print(f"{w:<6} " + "  ".join(f"{full[wi, ti]:6.2f}" for ti in range(len(T_GRID))))

    # 3. cross-fitting: choose (w, T) on one half, score it on the other half
    picks = []
    for h in (0, 1):
        wi, ti = divmod(int(acc[h].argmax()), len(T_GRID))
        picks.append((W_GRID[wi], T_GRID[ti], acc[1 - h, wi, ti].item(), acc[1 - h, 0, 0].item()))
    est = sum(p[2] for p in picks) / 2
    base = sum(p[3] for p in picks) / 2
    for h, (w, T, other, other_base) in enumerate(picks):
        print(f"chosen on half {h}: w={w}, T={T} -> on half {1 - h}: {other:.2f}% (head only {other_base:.2f}%)")
    print(f"\ncross-fitted estimate: {est:.2f}% against head only {base:.2f}% ({est - base:+.2f}); "
          f"noise bar about 0.5 points")


if __name__ == "__main__":
    main()
