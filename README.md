# CS604 project: adapting ViT-B/16 to a 10,000-class dataset

Team project: full fine-tuning and parameter-efficient adaptation (LoRA, Adapter, AdaptFormer, BitFit, ...) of a pre-trained
ViT-B/16, under a 2 h x 48 GB budget per run. Everything runs on Google Colab GPUs from VS Code (or the Colab web UI).

## Layout

```
common/      SHARED. Change only by pull request that all 3 of us agree on.
  data.py      dataset reading, fixed hold-out split, transforms
  build.py     pre-trained ViT-B/16 + new head + method
  head_init.py nearest-class-mean classifier head (the "zero-shot" baseline)
  train.py     training loop (bf16/fp16, resume, budget, logging, results row)
  metrics.py   trainable params, latency, GPU info, results.csv writer
  submit.py    predict val/test, write + validate + zip the CodaBench file
methods/     OWN. One file per adaptation method: apply(model, args) -> model. Register in methods/__init__.py.
configs/     One yaml per run (method, lr, batch size, ...). Baselines + smoke test.
notebooks/   00 data setup/check, 01 throughput test, 02 train + submit (thin runners).
results/results.csv   One row per run, everybody appends (failed runs too).
```

## Quick start (Colab)

1. In VS Code install the **Google Colab** extension. Open `notebooks/02_train_and_submit.ipynb`, Select Kernel -> Colab -> a GPU
   (**A100** for real runs; T4 is ~25 img/s and unusable).
2. Set `REPO_URL`, `CONFIG`, `WHO`, `TEAM` in the first cell and run top to bottom. Run `CONFIG = "smoke"` first.
3. Add the printed result row to `results/results.csv` locally and commit it.

Or from any Colab terminal/cell, from the repo root: `python -m common.train configs/lora_baseline.yaml out_dir=runs/lora who=me`.

## Rules (so that our numbers are comparable)

- Same data split, preprocessing and metrics for everyone: use `common/`, do not fork it. Need a change? Open a PR and tell the team.
- New method = new file in `methods/` + a config in `configs/`. Work on your own branch (`yourname/method`).
- Every run adds one row to `results/results.csv`. One change at a time so each run is a clean ablation.
- Only one person submits to CodaBench for Phase 2 (only 2 submissions in total). Phase 1: max 5 per day.
- Never commit data, checkpoints, zips, or the course PDF (it contains the CodaBench secret link). `.gitignore` covers these.
- Notebooks are committed **without outputs**: `pip install nbstripout && nbstripout --install` once per clone.

## Status

The code in `common/` was written before any GPU run: only a tiny CPU smoke test of the LoRA path was done, before the last change
(automatic head-init scale). Treat the first GPU run of `configs/smoke.yaml` as the real test and fix what breaks.
The hyper-parameters in `configs/*_baseline.yaml` are untuned starting points.
