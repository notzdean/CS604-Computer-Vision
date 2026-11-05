"""Batch-1 inference latency of finished runs, measured several times per size. Nothing is trained or scored.

    python -m common.latency runs/<a> runs/<b> --sizes 288 320 352
    python -m common.latency runs/<a> --sizes 320 --repeats 10

Why: a single reading (as in common.evaluate and the results row) can be thrown off by anything else using the GPU at
that moment; single readings of 16 to 40 ms appeared next to about 7 to 12 ms for neighbouring sizes. Here every
(run, size) is timed `--repeats` times, each time with metrics.latency_ms (median of 100 passes after 20 warm-up
passes), and the sizes and runs are interleaved, so a short slowdown shows up as one outlier in the min-to-max range
instead of shifting a whole size. Report the median and say how it was measured.

Works on any finished run, including final runs without a hold-out (only config.yaml and final.pt are read).
Check first that nothing else uses the GPU: !pgrep -af common.train ; nvidia-smi
"""
import argparse
import statistics
import sys

import torch
import yaml

from common import metrics
from common.build import build_model
from common.train import pick_amp


def load_run(run_dir, device):
    cfg = yaml.safe_load(open(f"{run_dir}/config.yaml"))
    cfg["dynamic_img_size"] = True           # as in common.evaluate when scoring several sizes
    torch.manual_seed(cfg.get("seed", 0))
    model = build_model(cfg).to(device).eval()
    _, unexpected = model.load_state_dict(torch.load(f"{run_dir}/final.pt", map_location="cpu"), strict=False)
    assert not unexpected, unexpected
    return cfg["name"], model, pick_amp(cfg["precision"], device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--sizes", type=int, nargs="+", default=[224], help="test image sizes, multiples of 16")
    ap.add_argument("--repeats", type=int, default=5, help="timings per (run, size), each the median of 100 passes")
    a = ap.parse_args()
    if any(s % 16 for s in a.sizes):
        sys.exit("sizes must be multiples of 16 (the patch size)")
    if not torch.cuda.is_available():
        sys.exit("needs a GPU: CPU latency is not comparable with the numbers in the results")
    device = torch.device("cuda")

    runs = [load_run(d, device) for d in a.run_dirs]
    times = {(name, s): [] for name, _, _ in runs for s in a.sizes}
    for _ in range(a.repeats):
        for name, model, amp in runs:
            for s in a.sizes:
                times[(name, s)].append(metrics.latency_ms(model, device, amp, n=100, warm=20, size=s))

    print(f"{torch.cuda.get_device_name(0)} | batch 1 | median of {a.repeats} repeats (min to max), ms")
    for name, _, _ in runs:
        print(name)
        for s in a.sizes:
            t = times[(name, s)]
            print(f"  {s} px: {statistics.median(t):6.1f}   ({min(t):.1f} to {max(t):.1f})")


if __name__ == "__main__":
    main()
