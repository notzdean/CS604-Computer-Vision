"""Score an ensemble of finished runs on the hold-out. Nothing is trained.

    python -m common.ensemble runs/<a> runs/<b> --size 288

Loads <run>/config.yaml + <run>/final.pt of each run, predicts the hold-out at one test size, and prints top-1 / top-5
of each model alone and of the average of their class probabilities. Inference latency to report is the sum of the
models' latencies, because every image goes through each model. Runs with no hold-out (holdout_per_class=0) are refused.
"""
import argparse
import sys

import torch
import yaml

from common import data, metrics
from common.build import build_model
from common.train import pick_amp


@torch.no_grad()
def hold_probs(run_dir, size, device):
    """-> (probabilities on CPU, labels on CPU, latency in ms) of one run on the hold-out."""
    cfg = yaml.safe_load(open(f"{run_dir}/config.yaml"))
    if not cfg["holdout_per_class"]:
        sys.exit(f"{run_dir} trained on every labelled image (holdout_per_class=0), so there is no hold-out to score it on")
    cfg["dynamic_img_size"] = size != cfg.get("train_size", 224)
    amp = pick_amp(cfg["precision"], device)
    torch.manual_seed(cfg.get("seed", 0))      # same init as in training for anything that is not pre-trained or saved
    model = build_model(cfg).to(device)
    _, unexpected = model.load_state_dict(torch.load(f"{run_dir}/final.pt", map_location="cpu"), strict=False)
    assert not unexpected, unexpected
    model.eval()
    rows = data.read_train()
    _, hold = data.split_holdout(rows, cfg["holdout_per_class"])
    probs, labels = [], []
    for x, y in data.make_loader(hold, data.eval_tf(size), cfg["eval_batch_size"], workers=cfg.get("num_workers")):
        with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
            logits = model(x.to(device, non_blocking=True))
        probs.append(logits.float().softmax(1).cpu()); labels.append(y)
    return torch.cat(probs), torch.cat(labels), metrics.latency_ms(model, device, amp, size=size)


def score(probs, labels, chunk=1000):
    c1 = c5 = 0
    for a in range(0, len(labels), chunk):
        top5 = probs[a:a + chunk].topk(5, dim=1).indices
        c1 += (top5[:, 0] == labels[a:a + chunk]).sum().item()
        c5 += (top5 == labels[a:a + chunk, None]).any(1).sum().item()
    return 100 * c1 / len(labels), 100 * c5 / len(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--size", type=int, default=224, help="test image size, a multiple of 16 (default 224)")
    a = ap.parse_args()
    if a.size % 16:
        sys.exit("--size must be a multiple of 16 (the patch size)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total, labels, latency, best = None, None, 0.0, 0.0
    for run_dir in a.run_dirs:
        probs, y, lat = hold_probs(run_dir, a.size, device)
        assert labels is None or torch.equal(labels, y), "the runs do not share the same hold-out"
        top1, top5 = score(probs, y)
        print(f"{run_dir} | {a.size} px | top-1 {top1:.2f}% top-5 {top5:.2f}% | latency {lat:.1f} ms")
        total, labels, latency, best = (probs if total is None else total + probs), y, latency + lat, max(best, top1)
    top1, top5 = score(total, labels)
    print(f"ensemble of {len(a.run_dirs)} | {a.size} px | top-1 {top1:.2f}% top-5 {top5:.2f}% ({top1 - best:+.2f} over the "
          f"best single model) | latency {latency:.1f} ms")


if __name__ == "__main__":
    main()
