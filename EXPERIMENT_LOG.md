# Experiment log

Running record of every experiment for the ablation study and the report. One section per run, plus a summary table.
Source of truth for numbers is `results/results.csv` (one row per run); this file adds the interpretation.
Brief marks: motivation of design (most important), design, **ablation of each part**, top-1, trainable params, latency, GPU-hours, loss curves.

**Counting rule (confirmed by the professor, 11 Oct 2026):** trainable % = adapter params / backbone params (85.8M). The classifier head (7.69M) is excluded and shown separately. The <5% limit applies to the efficient-adaptation track only.
**Hold-out:** 2 labelled train images per class (20,000 images). These are *not* the val set. The CodaBench val leaderboard is the real check.

## Summary

| Run | Method | Config | Hold-out top-1 | Trainable % (adapter) | Head params | GPU-h (A100 80GB) | Leaderboard val top-1 (not published) |
|---|---|---|---|---|---|---|---|
| zero-shot (head init only) | none | `lora_baseline` before training | 35.70 | 0 | 7.69M (init only) | - | - |
| lora_r8_full | LoRA r=8, qkv | `lora_baseline` | 68.78 | 0.344 | 7.69M (trained) | 0.76 | not recorded here |
| lora_adaptformer | LoRA r=12 qkv + AdaptFormer b=208 (FFN) | `lora_adaptformer` | 71.29 | 4.998 | 7.69M (trained) | 0.93 | not recorded here |

Brief reference points: zero-shot 43.7%, full FT 70.8%, LoRA 68.3% (0.34%).

## Runs

### lora_r8_full (4 Oct 2026, Nicole)

- **Setup:** `configs/lora_baseline.yaml`, LoRA r=8, alpha=16, qkv only, head trainable, AdamW lr 5e-4, head lr x1, wd 0.01, batch 128, bf16, nearest-class-mean head init (scale auto -> 0.02).
- **Result:** hold-out top-1 68.78%, top-5 85.78%. Zero-shot 35.70%.
- **Efficiency:** 294,912 adapter params = 0.344% of backbone. Latency 9.6 ms (batch 1). Peak VRAM 6.7 GiB. Train 0.76 h (~1.27 h in 48 GB-equivalent, inside the 2 h budget). 7.02 epochs (23,582 steps).
- **Loss:** 3.48 at step 50, 1.87 at step 2,050, falling steadily. No early spike, so the head-init scale choice looks right.
- **Interpretation:**
  - Training added ~33 points on top of the head init.
  - LoRA with 0.34% of the backbone matches the brief's 68.3% reference, but the hold-out and the reference are not identical sets, so compare with care.
  - The run stopped at 7 epochs in 0.76 h, well under the 1.2 h budget estimated after the 35-step probe. The probe underestimated throughput (815 img/s measured, ~1,245 img/s sustained), so the budget logic was conservative.
- **Hold-out by epoch (from `results/lora_r8_full/log.jsonl`):**

  | Epoch | Hold-out top-1 | Top-5 | Hold-out loss |
  |---|---|---|---|
  | 1 | 62.01 | 84.06 | 1.645 |
  | 2 | 65.23 | 85.38 | 1.571 |
  | 3 | 66.26 | 85.42 | 1.616 |
  | 4 | 66.78 | 85.36 | 1.705 |
  | 5 | 67.92 | 85.57 | 1.712 |
  | 6 | 68.62 | 85.83 | 1.700 |
  | 7 | 68.75 | 85.80 | 1.691 |
  | 7.02 (final) | 68.78 | 85.78 | 1.692 |

- **Training loss (mean per epoch):** 2.09, 1.07, 0.65, 0.39, 0.23, 0.15, 0.12. It falls about 20x over training.
- **Interpretation:**
  - Top-1 keeps rising slowly after epoch 2, but the hold-out loss is flat or rising from epoch 2 (1.57 to 1.69) while training loss drops to 0.12. That's a sign of overfitting in confidence: the model gets more sure of its answers, and not all of them are right.
  - The accuracy gains in epochs 5 to 7 are small (+0.9 points in total). More epochs will not help much, so the limit is the model's capacity or the regularisation, not the training time.
  - Regularisation ablations (label smoothing, weight decay, stronger augmentation) are the next thing to test, not longer training.
- **Open questions:** the zero-shot 35.7% is below the brief's 43.7%. Likely causes: head init uses 10 images per class and a scale chosen by leave-one-out CE. Test a higher `per_class` before trusting the head init.
- **Artefacts (Drive):** `My Drive/cs604_runs/lora_r8_full/` (`curves.png`, `log.jsonl`, `summary.json`, `val_team_01.zip`).

### lora_adaptformer (4 Oct 2026, Nicole)

- **Setup:** `configs/lora_adaptformer.yaml`. LoRA r=12 (alpha 24) on each block's `qkv`, plus an AdaptFormer bottleneck of 208 on each block's FFN (zero-init up-projection). AdamW, lr 3e-4 (lower than LoRA alone, for the larger adapter), head trainable, head initialised the same way as before, bf16.
- **Result:** hold-out top-1 71.29%, top-5 88.02%, hold-out loss 1.35. Zero-shot 35.70% (same head init as the LoRA run).
- **Efficiency:** 4,287,936 adapter params = 4.998% of backbone (the 5% limit, with almost no margin). Head (7.69M) not counted. Latency 11.3 ms (batch 1), against 9.6 ms for LoRA alone. Peak VRAM 7.3 GiB. Train 0.93 h, 7.78 epochs (26,123 steps). Throughput about 1,105 img/s, about 11% slower than LoRA alone.
- **Interpretation:**
  - +2.5 points top-1 over LoRA alone (68.78%), at about 0.17 GPU-hours extra.
  - Hold-out loss keeps falling to the end (1.49 at epoch 1, 1.35 at the end). LoRA alone rose from epoch 2, so the extra capacity isn't overfitting in the same way. The gain looks like more capacity used well, not only longer training.
  - Hold-out accuracy is still rising at the end (70.2 at epoch 5, 71.3 at epoch 7.8), so it hasn't fully plateaued.
- **Caveats:** the hold-out (20,000 images) has about 0.3 points of sampling error per run, so the 2.5-point gap is not sampling noise, and the leaderboard ranks the two runs the same way. But this run differs from the LoRA baseline in three things at once (rank 8 to 12, the added adapter, lr 5e-4 to 3e-4) and ran 0.76 epochs longer, so it does not show which part earned the gain. The `abl_*` trio below answers that. Single seed. The 4.998% share leaves almost no margin, so state the counting rule in the report.
- **Log:** `results/lora_adaptformer/log.jsonl`.

## Next runs (planned 11 Oct 2026)

### Run queue

Take a run marked "open" and tell the team before you start, so no run is done twice. How to run one is in `README.md`. Every run uses `RUN = ""`; leave `EXTRA` empty unless a value is given here. Each takes about 1.2 h on an A100 80 GB.

| # | `CONFIG` | `EXTRA` | What it is | Owner | Status |
|---|---|---|---|---|---|
| 1 | `full_ft_baseline` | `ckpt_every_steps=1000000` | required baseline: the whole model is trained | Nicole | running since 11 Oct |
| 2 | `abl_combined` | | our design for 10 epochs: the comparison point for runs 3 to 8 | Nicole | next |
| 3 | `tune_label_smoothing` | | run 2 + label smoothing 0.1 | open | |
| 4 | `tune_lr_high` | | run 2 + learning rate 5e-4 | open | |
| 5 | `tune_lr_low` | | run 2 + learning rate 2e-4 | open | |
| 6 | `tune_head_init` | | run 2 + classifier start from 43 images per class (about 1.25 h) | open | |
| 7 | `abl_adaptformer_only` | | AdaptFormer alone at 5%, for the report | Lilian | claimed 12 Oct |
| 8 | `abl_lora_only` | | LoRA alone at 5%, for the report | open | |

When a run finishes, copy `log.jsonl` and `summary.json` from `My Drive/cs604_runs/<CONFIG>/` into `results/<CONFIG>/` and push those two files. Do not upload to CodaBench; one person handles submissions.

### 1. Full fine-tuning (required track, no result yet)

`CONFIG = "full_ft_baseline"`, `EXTRA = "ckpt_every_steps=1000000"`: each checkpoint is the whole model (about 1.1 GB), so this turns off the saves every 2,000 steps and keeps the one at the end of each epoch. It is the brief's Task 1 and the 100% baseline that the 5% design is compared with (brief reference 70.8%). The 5% limit does not apply to it. First 200 steps on 11 Oct: about 1,087 img/s (the same speed as the LoRA + AdaptFormer run), loss falling from 3.20 to 2.70.

### 2. Equal-budget ablation: which part of the design earns the gain?

All three use about 5% of the backbone, lr 3e-4, 10 epochs (about 1.18 h on an A100 80 GB, inside the 1.2 h budget) and identical settings otherwise. Only the adapter design differs.

| Run | Config | Design | Adapter % | Hold-out top-1 |
|---|---|---|---|---|
| A | `abl_lora_only` | LoRA r=116 on qkv | 4.984 | to run |
| B | `abl_adaptformer_only` | AdaptFormer b=232 on FFN | 4.998 | to run |
| C | `abl_combined` | LoRA r=12 on qkv + AdaptFormer b=208 on FFN | 4.998 | to run |

How to read it: if C beats both A and B by about 1 point or more, adapting attention and FFN together is better than spending the same budget on one of them. If A or B matches C, the simpler design is the better story. Rank 116 has never been run, so watch its first loss lines.

### 3. One-change tuning on `abl_combined` (one config per change)

Each config is `abl_combined` with exactly one setting changed, so it is compared with the `abl_combined` run (C above). Leave `RUN` empty so each config gets its own output folder.

| Config | The one change (baseline value) | Why | Hold-out top-1 |
|---|---|---|---|
| `abl_combined` | none: the baseline, 10 epochs (the earlier run stopped at 7.8) | uses the whole time budget | to run |
| `tune_label_smoothing` | label smoothing 0.1 (0.0) | the LoRA run got over-confident (train loss 0.12, hold-out loss rising); the brief hints at loss design | to run |
| `tune_lr_low` | lr 2e-4 (3e-4) | 3e-4 was never tuned | to run |
| `tune_lr_high` | lr 5e-4 (3e-4) | 3e-4 was never tuned | to run |
| `tune_head_init` | head init from 43 images per class (10) | zero-shot 35.7% vs the brief's 43.7% | to run |

How to decide: one change worth a few tenths of a point cannot be told apart from noise on its own (about 0.3 points of sampling error per run). So after these runs, make one config with every change that went up, run it once, and keep it only if it beats `abl_combined` clearly (about 1 point). That combined run is the one to submit.

`tune_head_init` ends near 1.25 h on an A100 80 GB (estimate: the extra feature pass adds 4 to 5 minutes), which is over the 1.2 h budget but inside the +-10% tolerance. Cheap check first: `EXTRA = "max_train_steps=50"` and read the zero-shot line.

Not config-only, so they come later: training the final model on all 45 images per class (no hold-out left to score it), mirror averaging at prediction time (doubles latency), and moving the 5% between attention and FFN (depends on runs A to C).

## Planned ablations (one change per run)

Each row is one run. Add the result to `results/results.csv` and a section above.

| Question (design part) | Variants | Why it matters |
|---|---|---|
| Head init | per_class 10 vs 45; scale auto vs 1.0 vs 0.1; random head | brief hints "initialise the classifier head"; zero-shot is below reference |
| LoRA rank | r = 4, 8, 16 | trade-off between trainable % and accuracy |
| LoRA target | qkv vs qkv + proj vs q,v only | which layers carry the adaptation |
| Adapter type | LoRA vs AdaptFormer (b=64) vs BitFit | the brief's efficient-adaptation options |
| AdaptFormer bottleneck | 16, 64, 128 | size vs accuracy |
| Head trainable | trainable vs frozen (NCM) | what training the 10,000-class head adds |
| Loss | cross-entropy vs label smoothing 0.1 | brief hints "consider what loss to use" |
| Full fine-tuning | `full_ft_baseline` | reference 70.8%, for comparison |
| Precision | bf16 vs fp16 | brief hints bf16; checks no accuracy loss |

## Notes for the report

- Trainable % is measured against the backbone and the head is excluded. The professor confirmed on 11 Oct 2026 that the classifier head does not count towards the 5% limit. It also matches the brief's 0.34% LoRA figure (rank 8 on qkv). State the rule in the report anyway.
- Open question for the professor: an 80 GB A100 is counted here as 80 GB (budget 1.2 h). The LoRA run peaked at 6.7 GiB, so ask whether such a run may use the full 2 h.
- Hold-out numbers are not leaderboard numbers. Report both and say which is which.
- Leaderboard scores are not published by teams, and this repo is public, so they are not recorded here. Keep them in the report only.
