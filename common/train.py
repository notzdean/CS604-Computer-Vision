"""Shared training loop for every method.

    python -m common.train configs/lora.yaml [key=value ...]      (run from the repo root)

- Trains on `train` minus a fixed labelled hold-out; scores on the hold-out (val/test labels are withheld).
  `holdout_per_class=0` trains on every labelled image (final model only): no local score, hold-out columns stay empty.
- Precision: bf16 on Ampere+ GPUs, fp16 + loss scaling otherwise (T4), fp32 on CPU.
- Resumable: state is written to <out_dir>/latest.pt; re-running the same command continues from it.
- Budget: `budget_hours: auto` -> 96 GB-hours / GPU size. If `epochs` is null the number of steps is chosen
  after measuring the real speed for a few steps, so the run fits the budget.
- Appends one row to results/results.csv when finished.
- Opt-in regularisers (all off by default, so older configs train exactly as before): `drop_path_rate` (stochastic
  depth), `layer_decay` (layer-wise lr decay), `ema_decay` (weight EMA: evaluations and final.pt use the EMA weights),
  `auto_augment` / `re_prob` (TrivialAugment or RandAugment / random erasing), `mixup_alpha` / `cutmix_alpha`.
"""
import json
import math
import os
import re
import sys
import time
from contextlib import contextmanager

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

from common import data, metrics
from common.build import build_model
from common.head_init import ncm_head_init

DEFAULTS = dict(
    name=None, who=None, notes="",
    method="full_ft", method_args={},
    num_classes=data.NUM_CLASSES, debug_classes=None,        # debug_classes=N: only classes < N (fast tests)
    epochs=None, budget_hours="auto", max_train_steps=None,   # epochs=null -> as many as fit in the budget
    batch_size=128, lr=1e-4, head_lr_mult=10.0, weight_decay=0.05, warmup_steps=200, min_lr_frac=0.01,
    grad_clip=1.0, label_smoothing=0.0, scale_min=0.35,
    drop_path_rate=0.0, layer_decay=None, ema_decay=None,                  # opt-in regularisers (see docstring)
    auto_augment=None, re_prob=0.0, mixup_alpha=0.0, cutmix_alpha=0.0, mixup_prob=1.0,
    head_init=True, head_init_per_class=10, head_init_scale="auto", eval_at_start=True,
    precision="auto", num_workers=None, eval_batch_size=256, seed=0, holdout_per_class=data.HOLDOUT_PER_CLASS,
    eval_every_epochs=1, ckpt_every_steps=2000, log_every=50,
    out_dir=None, results_csv="results/results.csv",
)
PROBE_START, PROBE_END = 100, 300    # steps used to measure speed when epochs is null (the first ~50 steps are slow)


def load_cfg(argv):
    cfg = dict(DEFAULTS)
    if argv and "=" not in argv[0]:
        with open(argv[0]) as f:
            cfg.update(yaml.safe_load(f) or {})
        argv = argv[1:]
    for kv in argv:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    cfg["name"] = cfg["name"] or cfg["method"]
    cfg["out_dir"] = cfg["out_dir"] or f"runs/{cfg['name']}"
    cfg["who"] = cfg["who"] or os.environ.get("USER", "")
    return cfg


def pick_amp(precision, device):
    """returns the autocast dtype, or None for plain fp32"""
    if device.type != "cuda":
        return None
    if precision == "auto":
        precision = "bf16" if torch.cuda.get_device_capability(0)[0] >= 8 else "fp16"
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[precision]


@torch.no_grad()
def evaluate(model, rows, cfg, device, amp, flip=False, size=224):
    """flip=True: mirror averaging, the class probabilities of each image and of its mirror image are averaged.
    size != 224 needs a model built with dynamic_img_size (see common/evaluate.py)."""
    model.eval()
    n = c1 = c5 = 0
    loss_sum = 0.0
    for x, y in data.make_loader(rows, data.eval_tf(size), cfg["eval_batch_size"], workers=cfg["num_workers"]):
        x, y = x.to(device, non_blocking=True), y.to(device)
        with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
            logits = model(x)
            logits_m = model(x.flip(3)) if flip else None
        logits = logits.float()
        if flip:
            scores = (logits.softmax(1) + logits_m.float().softmax(1)) / 2
            loss_sum += F.nll_loss(scores.clamp_min(1e-12).log(), y, reduction="sum").item()
        else:
            scores = logits
            loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        top5 = scores.topk(min(5, scores.shape[1]), dim=1).indices
        c1 += (top5[:, 0] == y).sum().item()
        c5 += (top5 == y[:, None]).any(1).sum().item()
        n += len(y)
    return dict(top1=100 * c1 / n, top5=100 * c5 / n, loss=loss_sum / n, n=n)


def layer_id(name, n_layers):
    """Depth of a parameter for layer-wise lr decay: 0 = embeddings, i + 1 = block i, n_layers + 1 = final norm / head."""
    m = re.search(r"blocks\.(\d+)\.", name)
    if m:
        return int(m.group(1)) + 1
    if any(k in name for k in ("patch_embed", "cls_token", "pos_embed", "reg_token")):
        return 0
    return n_layers + 1


def make_groups(model, lr, head_mult, wd, layer_decay=None):
    """4 param groups: (head / backbone) x (decay for matrices / no decay for biases & norms).
    layer_decay (e.g. 0.75): each group is also split by depth and gets lr * layer_decay ** (layers above it),
    so early blocks change less than late ones (BEiT / MAE fine-tuning). None = one lr for every depth."""
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    n_layers = 1 + max((int(m.group(1)) for n, _ in named if (m := re.search(r"blocks\.(\d+)\.", n))), default=-1)
    buckets = {}
    for name, p in named:
        lid = layer_id(name, n_layers) if layer_decay else 0
        key = ("head" in name.split("."), p.ndim > 1, lid)
        buckets.setdefault(key, []).append(p)
    groups = []
    for (is_head, decay, lid), ps in buckets.items():
        base = lr * (head_mult if is_head else 1.0) * (layer_decay ** (n_layers + 1 - lid) if layer_decay else 1.0)
        groups.append(dict(params=ps, weight_decay=wd if decay else 0.0, base_lr=base, lr=base))
    return groups


@contextmanager
def swapped_in(params, values):
    """Temporarily copy `values` (the EMA weights) into `params`; no-op when values is None."""
    if values is None:
        yield
        return
    backup = [p.detach().clone() for p in params]
    with torch.no_grad():
        for p, v in zip(params, values):
            p.copy_(v)
    try:
        yield
    finally:
        with torch.no_grad():
            for p, b in zip(params, backup):
                p.copy_(b)


def trainable_state(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_curve(out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rows = [json.loads(l) for l in open(f"{out}/log.jsonl")]
        tr = [r for r in rows if r["kind"] == "train"]
        ev = [r for r in rows if r["kind"] == "eval"]
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
        ax[0].plot([r["step"] for r in tr], [r["loss"] for r in tr]); ax[0].set(xlabel="step", ylabel="train loss")
        ax[1].plot([r["step"] for r in ev], [r["top1"] for r in ev], "o-"); ax[1].set(xlabel="step", ylabel="hold-out top-1 (%)")
        fig.tight_layout(); fig.savefig(f"{out}/curves.png", dpi=120); plt.close(fig)
    except Exception as e:                                   # plotting must never kill a finished run
        print("could not plot curves:", e)


def main(argv):
    cfg = load_cfg(argv)
    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = pick_amp(cfg["precision"], device)
    gpu_name, gpu_gb = metrics.gpu_info()
    out = cfg["out_dir"]
    os.makedirs(out, exist_ok=True)
    if cfg["num_workers"] is None:
        cfg["num_workers"] = os.cpu_count()
    workers, bs = cfg["num_workers"], cfg["batch_size"]

    rows = data.read_train()
    if cfg["debug_classes"]:
        rows = [r for r in rows if r[1] < cfg["debug_classes"]]
        cfg["num_classes"] = cfg["debug_classes"]
    train_rows, hold_rows = data.split_holdout(rows, cfg["holdout_per_class"])
    spe = len(train_rows) // bs
    print(f"[{cfg['name']}] {gpu_name} ({gpu_gb} GB) | precision {amp} | train {len(train_rows)} hold-out {len(hold_rows)} "
          f"| {spe} steps/epoch")
    if not hold_rows:       # holdout_per_class=0: final-model mode, every labelled image is used for training
        print("no hold-out: this run cannot be scored locally, only on the leaderboard")

    latest = f"{out}/latest.pt"
    ck = torch.load(latest, map_location="cpu") if os.path.exists(latest) else None
    t0, elapsed_prev = time.time(), (ck["elapsed_s"] if ck else 0.0)
    elapsed = lambda: elapsed_prev + time.time() - t0
    with open(f"{out}/config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    init = None
    if cfg["head_init"] and not ck:
        init = lambda m: ncm_head_init(m, train_rows, device, cfg["head_init_per_class"], workers=workers, amp=amp,
                                       scale=cfg["head_init_scale"])
    model = build_model(cfg, init_head=init).to(device)
    n_train, n_head, n_base = metrics.count_params(model)
    print(f"trainable adapter params: {n_train:,} = {100 * n_train / n_base:.2f}% of backbone ({n_base:,}); "
          f"classifier head (not counted): {n_head:,} trainable")

    opt = torch.optim.AdamW(make_groups(model, cfg["lr"], cfg["head_lr_mult"], cfg["weight_decay"], cfg["layer_decay"]),
                            betas=(0.9, 0.999), fused=device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp == torch.float16)

    step, total_steps, zero_shot = 0, None, None
    if cfg["epochs"]:
        total_steps = cfg["epochs"] * spe
    if cfg["max_train_steps"]:
        total_steps = min(total_steps or 10 ** 12, cfg["max_train_steps"])
    if ck:
        _, unexpected = model.load_state_dict(ck["trainable"], strict=False)
        assert not unexpected, unexpected
        opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
        step, total_steps, zero_shot = ck["step"], ck["total_steps"], ck.get("zero_shot")
        print(f"resumed from step {step} (total {total_steps})")
    trainable = [p for p in model.parameters() if p.requires_grad]
    ema = None
    if cfg["ema_decay"]:                     # EMA of the trainable weights only (the frozen backbone never changes)
        ema = [t.to(device) for t in ck["ema"]] if ck and ck.get("ema") else [p.detach().clone() for p in trainable]
    if not ck and cfg["head_init"] and cfg["eval_at_start"] and hold_rows:
        zero_shot = evaluate(model, hold_rows, cfg, device, amp)
        print(f"zero-shot (head init only) hold-out top-1: {zero_shot['top1']:.2f}%")
    budget_h = 96 / gpu_gb if cfg["budget_hours"] == "auto" and gpu_gb else (
        None if cfg["budget_hours"] == "auto" else cfg["budget_hours"])
    if total_steps is None and budget_h is None:
        sys.exit("set `epochs`, `max_train_steps`, or run on a GPU with budget_hours: auto")

    warm, mn = cfg["warmup_steps"], cfg["min_lr_frac"]
    def lr_scale(s):
        if s < warm: return (s + 1) / warm
        if total_steps is None: return 1.0
        prog = min(1.0, (s - warm) / max(1, total_steps - warm))
        return mn + (1 - mn) * 0.5 * (1 + math.cos(math.pi * prog))

    def save_ckpt(path):
        atomic_save(dict(step=step, total_steps=total_steps, elapsed_s=elapsed(), trainable=trainable_state(model),
                         opt=opt.state_dict(), scaler=scaler.state_dict(), zero_shot=zero_shot,
                         ema=[t.cpu() for t in ema] if ema else None), path)

    log = open(f"{out}/log.jsonl", "a")
    def emit(rec):
        log.write(json.dumps(rec) + "\n"); log.flush()

    ds = data.ImageDS(train_rows, data.train_tf(cfg["scale_min"], cfg["auto_augment"], cfg["re_prob"]))
    mixer = None
    if cfg["mixup_alpha"] or cfg["cutmix_alpha"]:   # soft targets; label smoothing is applied inside the mixer
        from timm.data import Mixup
        mixer = Mixup(mixup_alpha=cfg["mixup_alpha"], cutmix_alpha=cfg["cutmix_alpha"], prob=cfg["mixup_prob"],
                      label_smoothing=cfg["label_smoothing"], num_classes=cfg["num_classes"])
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats()
    win_t, win_n, run_loss, probe_t = time.time(), 0, torch.zeros((), device=device), None
    done = total_steps is not None and step >= total_steps
    while not done:
        epoch, off = divmod(step, spe)
        perm = torch.randperm(len(train_rows), generator=torch.Generator().manual_seed(cfg["seed"] * 1000 + epoch))
        dl = DataLoader(Subset(ds, perm.tolist()[off * bs: spe * bs]), batch_size=bs, num_workers=workers,
                        pin_memory=device.type == "cuda", drop_last=True, prefetch_factor=4 if workers > 0 else None)
        model.train()
        for x, y in dl:
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * lr_scale(step)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if mixer:
                x, y = mixer(x, y)
            with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
                logits = model(x)
            if mixer:
                loss = torch.sum(-y * F.log_softmax(logits.float(), dim=1), dim=1).mean()
            else:
                loss = F.cross_entropy(logits.float(), y, label_smoothing=cfg["label_smoothing"])
            scaler.scale(loss).backward()
            if cfg["grad_clip"]:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable, cfg["grad_clip"])
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            step += 1; win_n += bs; run_loss += loss.detach()
            if ema is not None:
                d = min(cfg["ema_decay"], (1 + step) / (10 + step))     # short warm-up so early weights fade fast
                with torch.no_grad():
                    torch._foreach_lerp_(ema, [p.detach() for p in trainable], 1 - d)

            if step == PROBE_START:
                if device.type == "cuda": torch.cuda.synchronize()
                probe_t = time.time()
            if total_steps is None and step == PROBE_END:   # choose the number of steps that fits the budget
                if device.type == "cuda": torch.cuda.synchronize()
                ips = (PROBE_END - PROBE_START) * bs / (time.time() - probe_t)
                left = budget_h * 3600 * 0.95 - elapsed()
                total_steps = step + max(0, int(left * ips / bs * 0.93))     # 7% margin for evals/checkpoints
                print(f"measured {ips:.0f} img/s -> {total_steps} steps = {total_steps / spe:.1f} epochs "
                      f"in the {budget_h:.2f} h budget")
            if step % cfg["log_every"] == 0:
                l = (run_loss / cfg["log_every"]).item(); run_loss.zero_()
                ips = win_n / (time.time() - win_t); win_t, win_n = time.time(), 0
                emit(dict(kind="train", step=step, epoch=step / spe, loss=l, lr=opt.param_groups[0]["lr"], ips=ips,
                          elapsed_s=elapsed()))
                print(f"step {step}/{total_steps} ep {step / spe:.2f} loss {l:.3f} {ips:.0f} img/s {elapsed() / 60:.1f} min")
                if budget_h and elapsed() > budget_h * 3600 * 1.08:
                    print("over budget - stopping early"); total_steps = step
            if step % cfg["ckpt_every_steps"] == 0:
                save_ckpt(latest)
            if total_steps is not None and step >= total_steps:
                done = True
                break
        if not done and (epoch + 1) % cfg["eval_every_epochs"] == 0:
            if hold_rows:
                with swapped_in(trainable, ema):
                    ev = evaluate(model, hold_rows, cfg, device, amp)
                emit(dict(kind="eval", step=step, epoch=step / spe, **ev))
                print(f"== epoch {epoch + 1}: hold-out top-1 {ev['top1']:.2f}%  top-5 {ev['top5']:.2f}%  loss {ev['loss']:.3f}")
            save_ckpt(latest)

    train_s = elapsed()
    if ema is not None:                      # the EMA weights are the final model (scored, timed and saved)
        with torch.no_grad():
            for p, e in zip(trainable, ema):
                p.copy_(e)
    ev = evaluate(model, hold_rows, cfg, device, amp) if hold_rows else None
    if ev:
        emit(dict(kind="eval", step=step, epoch=step / spe, **ev))
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else 0.0
    lat = metrics.latency_ms(model, device, amp)
    atomic_save(trainable_state(model), f"{out}/final.pt")
    save_ckpt(latest)
    row = dict(name=cfg["name"], who=cfg["who"], method=cfg["method"], holdout_top1=round(ev["top1"], 2) if ev else "",
               holdout_top5=round(ev["top5"], 2) if ev else "", zero_shot_top1=round(zero_shot["top1"], 2) if zero_shot else "",
               trainable_params=n_train, trainable_pct=round(100 * n_train / n_base, 3), head_params=n_head, latency_ms_bs1=round(lat, 1),
               train_hours=round(train_s / 3600, 3), gpu=gpu_name, gpu_gb=gpu_gb,
               equiv_hours_48gb=round(train_s / 3600 * gpu_gb / 48, 3) if gpu_gb else "", peak_vram_gib=round(peak, 1),
               epochs=round(step / spe, 2), steps=step, lr=cfg["lr"], batch_size=bs, notes=cfg["notes"])
    metrics.append_result(cfg["results_csv"], row)
    with open(f"{out}/summary.json", "w") as f:
        json.dump(row, f, indent=1)
    save_curve(out)
    print("DONE", json.dumps(row))


if __name__ == "__main__":
    main(sys.argv[1:])
