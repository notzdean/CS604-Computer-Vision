"""Calibration check for H22: expected calibration error (ECE) of uniform against look-alike label smoothing. Nothing is trained.

Run on Colab (GPU, data downloaded by notebook 00, Drive mounted), from the repo root:

    python -m analysis.ece --runs "/content/drive/MyDrive/cs604_runs"

Scores each run on the fixed hold-out (2 images per class, the same images for every run; common/data.split_holdout)
at its training size, the same way common/evaluate.py builds the model, and prints top-1 (check it against
results.csv: 74.28 and 73.67), mean NLL, mean confidence and ECE over 15 equal-width bins of the top-1 confidence.
"""
import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from common import data
from common.build import build_model
from common.train import pick_amp

RUNS = ["imp_trivialaug", "imp_ls_neighbours"]      # uniform smoothing 0.1, smoothing towards 20 look-alike classes


@torch.no_grad()
def score(run_dir):
    cfg = yaml.safe_load(open(f"{run_dir}/config.yaml"))
    if not cfg["holdout_per_class"]:
        sys.exit(f"{run_dir} has no hold-out")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = pick_amp(cfg["precision"], device)
    torch.manual_seed(cfg.get("seed", 0))
    model = build_model(cfg).to(device)
    _, unexpected = model.load_state_dict(torch.load(f"{run_dir}/final.pt", map_location="cpu"), strict=False)
    assert not unexpected, unexpected
    model.eval()
    _, hold = data.split_holdout(data.read_train(), cfg["holdout_per_class"])
    conf, correct, nll = [], [], []
    size = cfg.get("train_size", 224)
    for x, y in data.make_loader(hold, data.eval_tf(size), cfg["eval_batch_size"], workers=cfg["num_workers"]):
        x = x.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
            logits = model(x)
        logits = logits.float().cpu()
        p = logits.softmax(1)
        c, k = p.max(1)
        conf.append(c.numpy()); correct.append((k == y).numpy())
        nll.append(F.cross_entropy(logits, y, reduction="none").numpy())
    return np.concatenate(conf), np.concatenate(correct).astype(float), np.concatenate(nll)


def ece(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return 100 * e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="folder holding the run folders")
    a = ap.parse_args()
    print(f"{'run':22s} {'top-1':>7s} {'NLL':>7s} {'mean conf':>10s} {'ECE':>7s}")
    for r in RUNS:
        conf, correct, nll = score(f"{a.runs}/{r}")
        print(f"{r:22s} {100 * correct.mean():6.2f}% {nll.mean():7.3f} {100 * conf.mean():9.2f}% {ece(conf, correct):6.2f}%")
    print("Lower ECE = better calibrated. Mean confidence above top-1 = over-confident, below = under-confident.")


if __name__ == "__main__":
    main()
