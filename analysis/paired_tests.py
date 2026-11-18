"""Paired tests for the report's deciding comparisons: are the differences bigger than hold-out sampling noise? Nothing is trained.

Run on Colab (GPU, data downloaded by notebook 00, Drive mounted), from the repo root:

    python -m analysis.paired_tests --runs "/content/drive/MyDrive/cs604_runs"

Every run is scored on the same 20,000 hold-out images in the same order (common/data.split_holdout), so two runs can be
compared image by image. For each pair (A - B) it prints the top-1 difference, a 95% interval from 10,000 paired
bootstrap resamples of the images, and the exact two-sided McNemar p on the images that exactly one run gets right.
Per-image results are cached as <run>/correct_<size>.npy, so a re-run only scores what is new.
These tests measure noise from which images were held out; training randomness is measured by the second seed (H24).
"""
import argparse
import os

import numpy as np

from analysis.ece import score

# (hypothesis, run A, run B): A - B is reported, each run at its own training size
PAIRS = [
    ("H2", "abl_combined", "abl_adaptformer_only"),
    ("H3 (288 px)", "imp_trivialaug_288", "imp_adaptformer_only_288"),
    ("H10", "tune_head_lr_half", "tune_ls_lr1e4"),
    ("H21", "imp_head_delay", "imp_trivialaug"),
    ("H22", "imp_ls_neighbours", "imp_trivialaug"),
    ("H16 (288 px)", "imp_trivialaug_288", "full_ft_llrd_288"),
    ("H24 (seed)", "imp_trivialaug_288_s1", "imp_trivialaug_288"),
]


def correct(runs, run):
    cache = f"{runs}/{run}/correct_holdout.npy"
    if os.path.exists(cache):
        return np.load(cache)
    _, c, _ = score(f"{runs}/{run}")
    np.save(cache, c.astype(bool))
    print(f"  scored {run}: top-1 {100 * c.mean():.2f}% (check against results.csv)")
    return c.astype(bool)


def bootstrap_ci(a, b, reps=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = a.astype(np.float32) - b.astype(np.float32)
    n, stats = len(d), np.empty(reps)
    for i in range(0, reps, 500):                    # chunks keep memory small
        idx = rng.integers(0, n, size=(min(500, reps - i), n))
        stats[i:i + len(idx)] = d[idx].mean(1)
    return 100 * np.percentile(stats, 2.5), 100 * np.percentile(stats, 97.5)


def mcnemar_p(a, b):
    from scipy.stats import binomtest
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return p, only_a, only_b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="folder holding the run folders")
    a = ap.parse_args()
    rows = []
    for h, ra, rb in PAIRS:
        ca, cb = correct(a.runs, ra), correct(a.runs, rb)
        assert len(ca) == len(cb), "hold-out sizes differ"
        diff = 100 * (ca.mean() - cb.mean())
        lo, hi = bootstrap_ci(ca, cb)
        p, only_a, only_b = mcnemar_p(ca, cb)
        rows.append((h, ra, rb, diff, lo, hi, p, only_a, only_b))
    print(f"\n{'test':13s} {'A - B':>7s} {'95% interval':>17s} {'McNemar p':>10s}  only A / only B right")
    for h, ra, rb, diff, lo, hi, p, only_a, only_b in rows:
        ps = "<0.001" if p < 0.001 else f"{p:.3f}"
        print(f"{h:13s} {diff:+7.2f} [{lo:+6.2f}, {hi:+6.2f}] {ps:>10s}  {only_a} / {only_b}   ({ra} - {rb})")
    print("\nAn interval that excludes 0 (or p < 0.05) means the difference is larger than hold-out sampling noise.")


if __name__ == "__main__":
    main()
