"""Turn a finished run into a CodaBench submission.

    python -m common.submit runs/<name> --team 05 [--split val|test]

Loads <run>/config.yaml + <run>/final.pt, predicts every image of the split, writes
<run>/<split>_team_XX.txt ("<image_path> <class_id>" per line), validates it, and zips it.
"""
import argparse
import os
import zipfile

import numpy as np
import torch
import yaml

from common import data
from common.build import build_model
from common.train import pick_amp


@torch.no_grad()
def predict(model, paths, device, amp, batch_size=256, workers=None):
    model.eval()
    top5_idx, top5_p = [], []
    for x, _ in data.make_loader(paths, data.eval_tf(), batch_size, workers=workers):
        with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
            logits = model(x.to(device, non_blocking=True))
        p = logits.float().softmax(1)
        v, i = p.topk(5, dim=1)
        top5_idx.append(i.cpu()); top5_p.append(v.cpu())
    return torch.cat(top5_idx).numpy(), torch.cat(top5_p).numpy()


def check(lines, paths, num_classes):
    assert len(lines) == len(paths), f"{len(lines)} lines, expected {len(paths)}"
    got = [l.split()[0] for l in lines]
    assert got == paths, "paths differ from the split file (order or content)"
    assert len(set(got)) == len(got), "duplicate paths"
    assert all(0 <= int(l.split()[1]) < num_classes for l in lines), "class id out of range"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--team", required=True, help="team number, e.g. 05 -> val_team_05.txt")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    a = ap.parse_args()

    cfg = yaml.safe_load(open(f"{a.run_dir}/config.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = pick_amp(cfg["precision"], device)
    model = build_model(cfg).to(device)          # pre-trained backbone + method structure; head is loaded below
    _, unexpected = model.load_state_dict(torch.load(f"{a.run_dir}/final.pt", map_location="cpu"), strict=False)
    assert not unexpected, unexpected

    paths = data.read_paths(a.split)
    idx, prob = predict(model, paths, device, amp, workers=cfg.get("num_workers"))
    lines = [f"{p} {c}" for p, c in zip(paths, idx[:, 0].tolist())]
    check(lines, paths, cfg["num_classes"])

    name = f"{a.split}_team_{a.team}.txt"
    txt = os.path.join(a.run_dir, name)
    with open(txt, "w") as f:
        f.write("\n".join(lines) + "\n")
    with zipfile.ZipFile(txt[:-4] + ".zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(txt, arcname=name)
    np.savez_compressed(os.path.join(a.run_dir, f"{a.split}_top5.npz"), idx=idx.astype(np.int16), prob=prob.astype(np.float16))
    print(f"wrote and validated {txt} ({len(lines)} lines) and {txt[:-4]}.zip - upload the zip to CodaBench")


if __name__ == "__main__":
    main()
