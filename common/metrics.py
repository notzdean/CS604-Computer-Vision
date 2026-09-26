"""Numbers the report needs: trainable params, latency, GPU info, and the shared results.csv."""
import csv
import os
import statistics
import time

import torch

BUDGET_GB_HOURS = 96          # 2 h x 48 GB


def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def gpu_info():
    """(name, nominal size in GB). Nominal size decides the compute budget: 96 GB-hours / size."""
    if not torch.cuda.is_available():
        return "cpu", 0
    props = torch.cuda.get_device_properties(0)
    gib = props.total_memory / 2 ** 30
    nominal = min([16, 24, 40, 48, 80], key=lambda v: abs(v - gib))
    return props.name, nominal


@torch.inference_mode()
def latency_ms(model, device, amp=None, n=50, warm=10):
    model.eval()
    x = torch.randn(1, 3, 224, 224, device=device)
    ts = []
    for i in range(n + warm):
        if device.type == "cuda": torch.cuda.synchronize()
        t = time.time()
        with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
            model(x)
        if device.type == "cuda": torch.cuda.synchronize()
        if i >= warm: ts.append((time.time() - t) * 1000)
    return statistics.median(ts)


RESULT_COLS = ["name", "who", "method", "holdout_top1", "holdout_top5", "zero_shot_top1", "trainable_params",
               "trainable_pct", "latency_ms_bs1", "train_hours", "gpu", "gpu_gb", "equiv_hours_48gb",
               "peak_vram_gib", "epochs", "steps", "lr", "batch_size", "leaderboard_val_top1", "notes"]


def append_result(path, row):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLS, extrasaction="ignore")
        if new: w.writeheader()
        w.writerow(row)
