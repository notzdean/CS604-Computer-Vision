# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

CS604 group project (3 people): fine-grained **10,000-class** image classification by adapting a pre-trained **ViT-B/16** (`timm` `vit_base_patch16_224.augreg2_in21k_ft_in1k`). Two required tracks, each within **2 h x 48 GB VRAM** (about 96 GB-hours; tolerance +-10%; e.g. 4 h on a 24 GB GPU or ~2.4 h on an A100 40 GB):

1. Full fine-tuning.
2. Efficient adaptation: freeze the backbone, train a small set (<5%) of parameters: LoRA, Adapter, AdaptFormer, Prefix tuning, BitFit, selected-layer tuning.

**Parameter-count rule (decided with the team, inferred from the brief):** the classifier head (768 x 10,000, ~7.7M params, 8.2% of the model) is **not counted**. Trainable % = adapter params / backbone params (85.8M). This matches the brief's LoRA 0.34% = rank 8 on `qkv`. The head's trainable count is logged separately as `head_params`. Report both in the write-up and say how the head is counted.

The brief hints at bf16, gradient accumulation, classifier-head initialisation, PEFT-adapter initialisation and loss design. Reference results from the brief (not bounds; 90%+ is not expected): zero-shot with initialised head 43.7%, full FT 70.8%, LoRA 68.3% (0.34% params). The original brief PDF is deliberately not in the repo (it contains the CodaBench link with a secret key).

## Data (not in the repo; lives on the Colab VM)

- Hugging Face dataset `doem1997/cs604-course-data`: `train.tar.gz` (~40 GB), `val.tar.gz` (~4.4 GB), `train.txt`, `val.txt`; `test.*` appears in Phase 2 (31 Oct).
- Train: 450,000 images, exactly 45 per class, `train/class_XXXX/<uuid>.jpg`, `train.txt` = `path class_id`. Images ~500x400 px, ~90 KB.
- Val: 50,000 images (5/class) and test: 100,000 (10/class) have **labels withheld** (`val.txt` is paths only). Do not label or train on them (README forbids it). There is no local val accuracy: `common/data.py` holds out a fixed labelled slice of train (2 per class = 20k images, seed 0) and every run is scored on that. The CodaBench val leaderboard is the only check on real val.
- Setup: notebook `00` downloads by streaming both archives into `tar` (~10-20 min) to `/content/cs604-course-data` (Colab local disk). The VM disk is wiped on disconnect, so this repeats per new session; the cell skips itself if the `.ready` marker exists. Google Drive (15 GB) is too small for the data: use it only for run outputs.
- Code finds the data via env var `CS604_ROOT` (default `/content/cs604-course-data`).

## Repo layout and how the pieces fit

- `common/` (shared, change only by team agreement; `metrics.count_params` implements the head-excluded counting rule): `data.py` (split, transforms, loaders) -> `build.py` (timm model, optional head init, then the method wrapper) -> `train.py` (loop) -> `metrics.py` (results row) and `submit.py` (CodaBench file). `head_init.py` sets the new head from class means (nearest-class-mean; `head_init_scale: auto` calibrates confidence with leave-one-out CE).
- `methods/`: one file per method, `apply(model, args) -> model`, registered in `methods/__init__.py`. Contract: `model.head` stays trainable. Head init runs *before* the method wraps the model (PEFT copies the head at wrap time). Existing: `full_ft`, `lora` (peft, on the fused `qkv`), `bitfit`, `adaptformer` (parallel bottleneck adapter on every block's FFN, zero-init up-projection). Adapter/Prefix are not written yet. `adaptformer_lora` has a config (`configs/adaptformer_lora_baseline.yaml`) but the method is **not registered**, so that config fails until it is written.

Measured adapter share (backbone-relative, head excluded): LoRA r=8 qkv 0.34%, LoRA r=16 qkv 0.69%, AdaptFormer b=64 1.39%, BitFit 0.12%, full FT 100%.
- `configs/*.yaml`: one run each; all keys and defaults are in `DEFAULTS` at the top of `common/train.py`. Command-line `key=value` overrides win. Current configs: `smoke`, `lora_baseline` (rank 8, alpha 16), `full_ft_baseline`, `bitfit_baseline`, `adaptformer_baseline`, `adaptformer_lora_baseline` (not runnable yet).
- `notebooks/`: `00_setup_and_data_check`, `01_throughput`, `02_train_and_submit` (thin runner; the only one you normally need).
- `results/results.csv`: one row per run, columns in `common/metrics.py:RESULT_COLS`.

Behaviours worth knowing before editing `train.py`:
- `epochs: null` + `budget_hours: auto` -> after 35 steps the loop measures its own img/s and sets the number of steps so the run fits 96 GB-hours / GPU size (7% margin); it stops early at 108% of budget. Set `epochs` or `max_train_steps` for fixed-length runs.
- Precision auto: bf16 on compute capability >= 8 (A100/L4/H100), fp16 + GradScaler on T4, fp32 on CPU.
- Resumable: `<out_dir>/latest.pt` (trainable weights + optimizer + step); re-running the same command continues. The frozen backbone is never saved, only trainable tensors (`final.pt`), so `submit.py` rebuilds the model from `config.yaml` + pre-trained weights + `final.pt`.
- LR schedule is per-step linear warmup then cosine, with 4 param groups (head/backbone x decay/no-decay), head LR = `lr * head_lr_mult`.

## Commands (run from the repo root)

```
pip install -r requirements.txt                       # timm, peft, pyyaml (Colab has torch)
python -m common.train configs/smoke.yaml out_dir=runs/smoke          # ~minutes, 100 classes, run this first
python -m common.train configs/lora_baseline.yaml out_dir=<drive>/lora who=<you> results_csv=<drive>/results.csv
python -m common.submit <run_dir> --team XX --split val               # writes + validates val_team_XX.txt and .zip
```
There is no test suite or linter. The smoke config is the pipeline test. Do not run these on the local Windows machine for real work: it has no GPU; all training happens on Colab.

## Colab and VS Code notes

- Compute is a Colab GPU used from VS Code with the official **Google Colab** extension: open the `.ipynb` locally, Select Kernel -> Colab -> New Colab Server (or reuse an existing server so the downloaded data is visible). Only notebooks run on Colab (no remote terminal); use `!` shell cells. Files written on the VM stay on the VM.
- Measured on a Colab **T4**: 25 img/s training (bf16 is emulated there) = ~5 h per epoch. **Unusable; use an A100** (or L4 as fallback) for real runs. The data loader (8 CPU cores) delivered ~880 img/s, so a fast GPU may become loader-bound. A100/L4 speeds have not been measured yet: the training loop logs its own img/s.
- Notebook 02, first cell: `CONFIG` (which yaml), `RUN` (output folder name; empty = same as CONFIG), `WHO`, `TEAM`, `EXTRA` (overrides). Use a **new** `RUN` whenever you repeat a config (pilot, changed lr), otherwise the run resumes the old checkpoint. Smoke test: leave `RUN` empty.
- Outputs always go to Google Drive: `My Drive -> cs604_runs -> <RUN or CONFIG>` (checkpoints, log, curves, results row, submission zip). The notebook fails with an error if Drive is not mounted, instead of silently using the VM disk.
- Disconnects kill the run; re-running the notebook resumes from the last checkpoint on Drive, after the data is re-downloaded. Terminate idle sessions (Runtime -> Manage sessions) so credits are not wasted.
- Colab credits are per person, so the 3 of us can run experiments in parallel.

## Conventions for anyone (human or agent) working here

- **Status (4 Oct 2026): untested on a GPU.** Only a tiny CPU run of the LoRA path has happened. The smoke config on an A100 is the first real test; expect small bugs. The submission cell has never run. Track progress in `results/results.csv`.
- Repo is public: https://github.com/notzdean/CS604-Computer-Vision. Pull before starting. Notebooks are committed with placeholder `WHO = "yourname"` and `TEAM = "XX"`, so set your own values locally and do not commit them.
- Keep runs comparable: use `common/` as is (split, transforms, metrics). Changing `common/` or the constants in `common/data.py` (`HOLDOUT_PER_CLASS`, `SPLIT_SEED`) invalidates everybody's results, so do it only by agreement.
- A new method is a new file in `methods/` + a config in `configs/`; change one thing per run so it is a clean ablation. The report is graded on design justification and ablations, so note *why* in the config `notes:` field.
- Every finished run appends one row to `results/results.csv` (failed runs too). Report metrics per run: hold-out top-1, trainable params (count and %), latency, train hours / peak VRAM, loss curve (`curves.png`).
- Submission output: notebook 02 section 6 writes `val_team_XX.txt` + `.zip` into the run folder on Drive. The zip is what you upload (CodaBench), not an Excel file. Only submit runs that finished training (`final.pt` exists); smoke output is not a real submission.
- Submissions: Phase 1 val leaderboard 26 Sep - 30 Oct 2026, max 5/day. Phase 2 test due 6 Nov, **2 submissions total**, one person owns them. File `val_team_XX.txt` / `test_team_XX.txt`, `<image_path> <class_id>` per line, every image exactly once, **zipped**. Use `common/submit.py` (it validates).
- Never commit data, checkpoints, zips, tokens or the CodaBench secret link. Notebooks are committed without outputs (`nbstripout --install`, configured in `.gitattributes`).
- Deadlines: presentation 7 Nov 12:00 (15+5 min), final report 18 Nov 23:59 in the CVPR 2026 LaTeX template.
