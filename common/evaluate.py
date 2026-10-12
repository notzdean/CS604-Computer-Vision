"""Score a finished run on the hold-out again, without and with mirror averaging. Nothing is trained.

    python -m common.evaluate runs/<name>

Loads <run>/config.yaml + <run>/final.pt and prints hold-out top-1 / top-5 for a single pass and for mirror
averaging (each image and its mirror image, probabilities averaged). Use it to decide whether
`python -m common.submit ... --tta flip` is worth the doubled inference latency.
"""
import sys

import torch
import yaml

from common import data
from common.build import build_model
from common.train import evaluate, pick_amp


def main(run_dir):
    cfg = yaml.safe_load(open(f"{run_dir}/config.yaml"))
    if not cfg["holdout_per_class"]:
        sys.exit("this run trained on every labelled image (holdout_per_class=0), so there is no hold-out to score it on")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = pick_amp(cfg["precision"], device)
    torch.manual_seed(cfg.get("seed", 0))      # same init as in training for anything that is not pre-trained or saved
    model = build_model(cfg).to(device)
    _, unexpected = model.load_state_dict(torch.load(f"{run_dir}/final.pt", map_location="cpu"), strict=False)
    assert not unexpected, unexpected

    rows = data.read_train()
    if cfg.get("debug_classes"):
        rows = [r for r in rows if r[1] < cfg["debug_classes"]]
    _, hold = data.split_holdout(rows, cfg["holdout_per_class"])
    plain = evaluate(model, hold, cfg, device, amp)
    mirror = evaluate(model, hold, cfg, device, amp, flip=True)
    print(f"[{cfg['name']}] hold-out images: {plain['n']}")
    print(f"single pass      : top-1 {plain['top1']:.2f}%  top-5 {plain['top5']:.2f}%  loss {plain['loss']:.3f}")
    print(f"mirror averaging : top-1 {mirror['top1']:.2f}%  top-5 {mirror['top5']:.2f}%  loss {mirror['loss']:.3f}")
    print(f"gain from mirror averaging: {mirror['top1'] - plain['top1']:+.2f} points top-1 (costs 2x inference latency)")


if __name__ == "__main__":
    main(sys.argv[1])
