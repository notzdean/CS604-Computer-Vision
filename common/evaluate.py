"""Score a finished run on the hold-out again, without and with mirror averaging. Nothing is trained.

    python -m common.evaluate runs/<name>                        # at 224 px, as in training
    python -m common.evaluate runs/<name> --sizes 224 256 288    # also at larger test sizes (FixRes check)

Loads <run>/config.yaml + <run>/final.pt and prints hold-out top-1 / top-5 for a single pass and for mirror
averaging (each image and its mirror image, probabilities averaged). Use it to decide whether
`python -m common.submit ... --tta flip` is worth the doubled inference latency.
--sizes: the model was trained on random crops, which make objects look larger than the 224 px centre crop at test
time, so testing at a slightly larger size often scores higher (Touvron et al., FixRes, 2019). Position embeddings
are interpolated; latency grows with the number of patches, so it is printed per size.
"""
import argparse
import sys

import torch
import yaml

from common import data, metrics
from common.build import build_model
from common.train import evaluate, pick_amp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--sizes", type=int, nargs="+", default=[224], help="test image sizes (multiples of 16)")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(f"{a.run_dir}/config.yaml"))
    if not cfg["holdout_per_class"]:
        sys.exit("this run trained on every labelled image (holdout_per_class=0), so there is no hold-out to score it on")
    if any(s % 16 for s in a.sizes):
        sys.exit("sizes must be multiples of 16 (the patch size)")
    cfg["dynamic_img_size"] = any(s != 224 for s in a.sizes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = pick_amp(cfg["precision"], device)
    torch.manual_seed(cfg.get("seed", 0))      # same init as in training for anything that is not pre-trained or saved
    model = build_model(cfg).to(device)
    _, unexpected = model.load_state_dict(torch.load(f"{a.run_dir}/final.pt", map_location="cpu"), strict=False)
    assert not unexpected, unexpected

    rows = data.read_train()
    if cfg.get("debug_classes"):
        rows = [r for r in rows if r[1] < cfg["debug_classes"]]
    _, hold = data.split_holdout(rows, cfg["holdout_per_class"])
    print(f"[{cfg['name']}] hold-out images: {len(hold)}")
    base = None
    for s in a.sizes:
        plain = evaluate(model, hold, cfg, device, amp, size=s)
        mirror = evaluate(model, hold, cfg, device, amp, flip=True, size=s)
        lat = metrics.latency_ms(model, device, amp, size=s)
        base = plain["top1"] if base is None else base
        print(f"{s} px | single pass: top-1 {plain['top1']:.2f}% top-5 {plain['top5']:.2f}% loss {plain['loss']:.3f} "
              f"| mirror averaging: top-1 {mirror['top1']:.2f}% ({mirror['top1'] - plain['top1']:+.2f}) "
              f"| latency {lat:.1f} ms (x2 with mirror) | vs first size: {plain['top1'] - base:+.2f}")


if __name__ == "__main__":
    main()
