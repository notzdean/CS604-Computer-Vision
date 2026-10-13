# Experiment log

Running record of every experiment for the ablation study and the report. One section per run, plus a summary table.
Source of truth for numbers is `results/results.csv` (one row per run); this file adds the interpretation.
Brief marks: motivation of design (most important), design, **ablation of each part**, top-1, trainable params, latency, GPU-hours, loss curves.

**Counting rule (confirmed by the professor, 11 Oct 2026):** trainable % = adapter params / backbone params (85.8M). The classifier head (7.69M) is excluded and shown separately. The <5% limit applies to the efficient-adaptation track only.
**Hold-out:** 2 labelled train images per class (20,000 images). These are *not* the val set. The CodaBench val leaderboard is the real check.

## Summary

| Run | Method | Config | Hold-out top-1 | Trainable % (adapter) | Head params | GPU-h (A100 80GB) | Leaderboard val top-1 (not published) |
|---|---|---|---|---|---|---|---|
| zero-shot (head init only, 10 images per class) | none | `lora_baseline` before training | 35.70 | 0 | 7.69M (init only) | - | - |
| zero-shot (head init only, 43 images per class) | none | `tune_head_init` before training | 43.30 | 0 | 7.69M (init only) | - | - |
| lora_r8_full | LoRA r=8, qkv | `lora_baseline` | 68.78 | 0.344 | 7.69M (trained) | 0.76 | not recorded here |
| lora_adaptformer | LoRA r=12 qkv + AdaptFormer b=208 (FFN) | `lora_adaptformer` | 71.29 | 4.998 | 7.69M (trained) | 0.93 | not recorded here |
| full_ft_baseline | Full fine-tuning (required baseline) | `full_ft_baseline` | 72.53 | 100 | 7.69M (trained) | 1.13 | not recorded here |
| abl_combined | LoRA r=12 qkv + AdaptFormer b=208 (FFN), 10 epochs | `abl_combined` | 71.05 | 4.998 | 7.69M (trained) | 1.18 | not recorded here |
| abl_adaptformer_only | AdaptFormer b=232 (FFN) only, 10 epochs | `abl_adaptformer_only` | 71.09 | 4.998 | 7.69M (trained) | 1.04 | not recorded here |

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
- **Open questions:** the zero-shot 35.7% is below the brief's 43.7%. Likely causes: head init uses 10 images per class and a scale chosen by leave-one-out CE. Test a higher `per_class` before trusting the head init. Answered on 12 Oct: with all 43 training images per class the zero-shot is 43.30% (top-5 67.30%), in line with the brief's 43.7%.
- **Artefacts (Drive):** `My Drive/cs604_runs/lora_r8_full/` (`curves.png`, `log.jsonl`, `summary.json`, `val_team_01.zip`).

### lora_adaptformer (4 Oct 2026, Nicole)

- **Setup:** `configs/lora_adaptformer.yaml`. LoRA r=12 (alpha 24) on each block's `qkv`, plus an AdaptFormer bottleneck of 208 on each block's FFN (zero-init up-projection). AdamW, lr 3e-4 (lower than LoRA alone, for the larger adapter), head trainable, head initialised the same way as before, bf16.
- **Result:** hold-out top-1 71.29%, top-5 88.02%, hold-out loss 1.35. Zero-shot 35.70% (same head init as the LoRA run).
- **Efficiency:** 4,287,936 adapter params = 4.998% of backbone (the 5% limit, with almost no margin). Head (7.69M) not counted. Latency 11.3 ms (batch 1), against 9.6 ms for LoRA alone. Peak VRAM 7.3 GiB. Train 0.93 h, 7.78 epochs (26,123 steps). Throughput about 1,105 img/s, about 11% slower than LoRA alone.
- **Interpretation:**
  - +2.5 points top-1 over LoRA alone (68.78%), at about 0.17 GPU-hours extra.
  - Hold-out loss keeps falling to the end (1.49 at epoch 1, 1.35 at the end). LoRA alone rose from epoch 2, so the extra capacity isn't overfitting in the same way. The gain looks like more capacity used well, not only longer training.
  - Hold-out accuracy is still rising at the end (70.2 at epoch 5, 71.3 at epoch 7.8). Later finding: the 10-epoch run of the same design (`abl_combined`, 71.05%) did not go higher, so this was the learning rate decaying, not a run cut short.
- **Caveats:** the hold-out (20,000 images) has about 0.3 points of sampling error per run, so the 2.5-point gap is not sampling noise, and the leaderboard ranks the two runs the same way. But this run differs from the LoRA baseline in three things at once (rank 8 to 12, the added adapter, lr 5e-4 to 3e-4) and ran 0.76 epochs longer, so it does not show which part earned the gain. The `abl_*` trio below answers that. Single seed. The 4.998% share leaves almost no margin, so state the counting rule in the report.
- **Log:** `results/lora_adaptformer/log.jsonl`.

### full_ft_baseline (11 Oct 2026, Nicole)

- **Setup:** `configs/full_ft_baseline.yaml`. Every backbone weight is trained (100%). AdamW, lr 3e-5 for the backbone and 3e-4 for the head (`head_lr_mult` 10), weight decay 0.05, batch 128, bf16, same head init as the other runs. Untuned: one run, one learning rate.
- **Result:** hold-out top-1 72.53%, top-5 88.44%, hold-out loss 1.31. Zero-shot 35.70%.
- **Efficiency:** 85,798,656 trainable backbone params (100%); head (7.69M) not counted. Latency 7.0 ms (batch 1). Peak VRAM 9.6 GiB. Train 1.134 h on an A100 80 GB = 1.89 h in 48 GB-equivalent, inside the 2 h budget. 9.33 epochs (31,352 steps), about 1,080 img/s.
- **Hold-out by epoch (from `results/full_ft_baseline/log.jsonl`):**

  | Epoch | Hold-out top-1 | Top-5 | Hold-out loss | LoRA + AdaptFormer top-1 (7.78-epoch run) |
  |---|---|---|---|---|
  | 1 | 63.16 | 84.83 | 1.567 | 64.83 |
  | 2 | 65.58 | 86.11 | 1.511 | 67.22 |
  | 3 | 66.92 | 86.14 | 1.521 | 68.31 |
  | 4 | 67.81 | 86.50 | 1.524 | 69.59 |
  | 5 | 69.80 | 86.83 | 1.468 | 70.23 |
  | 6 | 70.43 | 87.41 | 1.418 | 71.12 |
  | 7 | 71.39 | 87.92 | 1.367 | 71.20 |
  | 8 | 72.16 | 88.23 | 1.326 | 71.29 (final, epoch 7.78) |
  | 9 | 72.50 | 88.42 | 1.310 | |
  | 9.33 (final) | 72.53 | 88.44 | 1.308 | |

- **Training loss (mean per epoch):** 1.95, 1.04, 0.64, 0.39, 0.24, 0.14, 0.09, 0.06, 0.05.
- **Interpretation:**
  - Best run so far: 1.24 points above LoRA + AdaptFormer (71.29%), and above the brief's 70.8% reference for full fine-tuning. The gap is about 2.7 times its sampling error (0.45 points), so it is not noise.
  - Not yet a like-for-like comparison: this run trained 9.33 epochs, the LoRA + AdaptFormer run 7.78. The 5% design was ahead at every epoch up to epoch 6, but its learning-rate schedule was shorter, so that is not like for like either. `abl_combined` (10 epochs) is the fair comparison (done: 71.05%, so full fine-tuning stays 1.48 points ahead).
  - The hold-out loss falls from 1.57 to 1.31 (flat in epochs 2 to 4) while the training loss reaches 0.05, so full fine-tuning does not overfit worse than the adapters at this length. The brief expects adapters to win "when the dataset is small"; 430,000 training images is probably not small enough for that. Say so in the report whichever way run 2 goes.
  - Fastest model at inference: 7.0 ms, against 9.6 ms for LoRA and 11.3 ms for LoRA + AdaptFormer, most likely because it adds no layers. The LoRA part can be merged into the frozen weights after training, which removes its overhead; the AdaptFormer branch cannot be merged. The latencies come from different sessions, so treat gaps of 1 to 2 ms with care.
  - The speed-probe fix works: the run used 1.134 h of the 1.2 h budget (the two earlier auto-length runs used 0.76 and 0.93 h).
- **Log:** `results/full_ft_baseline/log.jsonl`, `summary.json`.

### abl_combined (11 Oct 2026, Nicole)

- **Setup:** `configs/abl_combined.yaml`: the `lora_adaptformer` design (LoRA r=12 on qkv + AdaptFormer b=208 on FFN, 4.998%) with the length fixed at 10 epochs. Everything else as in the 7.78-epoch run (lr 3e-4, weight decay 0.01, no label smoothing).
- **Result:** hold-out top-1 71.05%, top-5 87.52%, hold-out loss 1.416. Zero-shot 35.70%.
- **Efficiency:** 4,287,936 adapter params = 4.998% of backbone. Latency 11.4 ms (batch 1). Peak VRAM 7.3 GiB. Train 1.183 h on an A100 80 GB = 1.97 h in 48 GB-equivalent, inside the 2 h budget with little room. 10 epochs (33,590 steps), about 1,100 img/s.
- **Hold-out by epoch (from `results/abl_combined/log.jsonl`):**

  | Epoch | Hold-out top-1 | Top-5 | Hold-out loss | Full fine-tuning top-1 |
  |---|---|---|---|---|
  | 1 | 64.56 | 85.69 | 1.499 | 63.16 |
  | 2 | 66.97 | 86.63 | 1.441 | 65.58 |
  | 3 | 67.53 | 86.53 | 1.495 | 66.92 |
  | 4 | 68.45 | 86.77 | 1.509 | 67.81 |
  | 5 | 68.77 | 86.67 | 1.515 | 69.80 |
  | 6 | 69.59 | 87.00 | 1.488 | 70.43 |
  | 7 | 70.48 | 87.17 | 1.457 | 71.39 |
  | 8 | 70.77 | 87.30 | 1.435 | 72.16 |
  | 9 | 71.07 | 87.59 | 1.418 | 72.50 |
  | 10 (final) | 71.05 | 87.52 | 1.416 | 72.53 (final, epoch 9.33) |

- **Training loss (mean per epoch):** 1.87, 0.96, 0.59, 0.39, 0.26, 0.17, 0.12, 0.09, 0.08, 0.07.
- **Interpretation:**
  - Training longer did not help: 71.05% after 10 epochs against 71.29% after 7.78 epochs. The 0.24-point difference is inside the sampling error (0.45). The 7.78-epoch run flattened because its learning rate had decayed, not because it ran out of time, and a longer schedule ends in the same place.
  - Like for like, full fine-tuning wins: 72.53% (9.33 epochs) against 71.05% (10 epochs). The gap of 1.48 points is about 3 times its sampling error. Put the other way: the 5% design reaches 98% of the full fine-tuning accuracy with 5% of the backbone trained.
  - Over-confidence shows up with the longer schedule: the hold-out loss rises from 1.44 (epoch 2) to 1.52 (epoch 5) while the learning rate is still high, and ends at 1.416, worse than the shorter run (1.351) and full fine-tuning (1.308), although top-1 matches the shorter run. That is the case label smoothing is meant for, so `tune_label_smoothing` is the next run to look at.
  - The latency repeats: 11.4 ms here and 11.3 ms in the earlier run of the same design, against 7.0 ms for full fine-tuning. The adapter overhead at batch 1 is real, about 4 ms.
  - The two runs of this design (71.29 and 71.05) show the run-to-run spread: treat single-run differences under about 1 point as ties.
- **Comparison point for runs 3 to 8: 71.05%.**
- **Log:** `results/abl_combined/log.jsonl`, `summary.json`.

### abl_adaptformer_only (12 Oct 2026, lilian)

- **Setup:** `configs/abl_adaptformer_only.yaml`: AdaptFormer bottleneck 232 on every FFN and no LoRA, 4.998% of backbone. Same learning rate (3e-4), 10 epochs and other settings as `abl_combined`.
- **Result:** hold-out top-1 71.09%, top-5 87.47%, hold-out loss 1.426. Zero-shot 35.70%.
- **Efficiency:** 4,288,224 adapter params = 4.998% of backbone. Latency 9.2 ms (batch 1). Peak VRAM 6.5 GiB. Train 1.043 h on an A100 80 GB = 1.74 h in 48 GB-equivalent. 10 epochs (33,590 steps), about 1,270 img/s.
- **Hold-out by epoch (from `results/abl_adaptformer_only/log.jsonl`):**

  | Epoch | Hold-out top-1 | Top-5 | Hold-out loss | `abl_combined` top-1 |
  |---|---|---|---|---|
  | 1 | 64.37 | 85.38 | 1.513 | 64.56 |
  | 2 | 66.47 | 86.59 | 1.457 | 66.97 |
  | 3 | 67.44 | 86.19 | 1.495 | 67.53 |
  | 4 | 68.08 | 86.17 | 1.529 | 68.45 |
  | 5 | 68.69 | 86.45 | 1.520 | 68.77 |
  | 6 | 69.45 | 86.92 | 1.492 | 69.59 |
  | 7 | 70.17 | 86.99 | 1.467 | 70.48 |
  | 8 | 70.62 | 87.33 | 1.443 | 70.77 |
  | 9 | 70.98 | 87.44 | 1.430 | 71.07 |
  | 10 (final) | 71.09 | 87.47 | 1.426 | 71.05 |

- **Training loss (mean per epoch):** 1.89, 0.96, 0.59, 0.38, 0.25, 0.17, 0.12, 0.09, 0.08, 0.07.
- **Interpretation:**
  - It ties the combined design: 71.09% against 71.05%, and the two curves stay within 0.5 points of each other at every epoch. At an equal parameter budget, adding LoRA on the attention to the FFN adapter gains nothing measurable.
  - It is the cheaper design: 9.2 ms against 11.4 ms latency, 1.04 h against 1.18 h of training (about 1,270 against 1,100 img/s), 6.5 against 7.3 GiB. These come from another Colab session, so treat them as indicative.
  - Same over-confidence pattern as `abl_combined`: the hold-out loss rises from 1.46 (epoch 2) to 1.53 (epoch 4) and ends at 1.43.
  - Still 1.44 points below full fine-tuning (72.53%).
  - Still open: does the FFN placement matter, or only the number of adapter parameters? `abl_lora_only` (same budget, attention only) answers that.
- **Log:** `results/abl_adaptformer_only/log.jsonl`, `summary.json`.

## Next runs (planned 11 Oct 2026)

### Run queue

Take a run marked "open" and tell the team before you start, so no run is done twice. How to run one is in `README.md`. Every run uses `RUN = ""`; leave `EXTRA` empty unless a value is given here. Each takes about 1.2 h on an A100 80 GB.

| # | `CONFIG` | `EXTRA` | What it is | Owner | Status |
|---|---|---|---|---|---|
| 1 | `full_ft_baseline` | `ckpt_every_steps=1000000` | required baseline: the whole model is trained | Nicole | done 11 Oct: 72.53% |
| 2 | `abl_combined` | | our design for 10 epochs: the comparison point for runs 3 to 8 | Nicole | done 11 Oct: 71.05% |
| 3 | `tune_label_smoothing` | | run 2 + label smoothing 0.1 | open | |
| 4 | `tune_lr_high` | | run 2 + learning rate 5e-4 | open | |
| 5 | `tune_lr_low` | | run 2 + learning rate 2e-4 | open | |
| 6 | `tune_head_init` | | run 2 + classifier start from 43 images per class (about 1.25 h) | lilian | running 12 Oct: snapshot at epoch 5 of 10 received, final result pending |
| 7 | `abl_adaptformer_only` | | AdaptFormer alone at 5%, for the report | lilian | done 12 Oct: 71.09% |
| 8 | `abl_lora_only` | | LoRA alone at 5%, for the report | open | |
| 9 | `abl_adaptformer_last6` | | AdaptFormer in the last 6 blocks only, twice the bottleneck; compare with run 7 | open | |
| 10 | `tune_ls_lr2e4` | | run 2 + label smoothing 0.1 + learning rate 2e-4 (the two changes that each helped) | Nicole | next |
| 11 | `tune_ls_lr1e4` | | run 10 with learning rate 1e-4 (does lower still help?) | open | |
| 12 | `full_ft_ls` | | full fine-tuning + label smoothing 0.1, so the baseline gets the same trick | open | |

When a run finishes, copy `log.jsonl` and `summary.json` from `My Drive/cs604_runs/<CONFIG>/` into `results/<CONFIG>/` and push those two files. Do not upload to CodaBench; one person handles submissions.

### 1. Full fine-tuning (required track): done, 72.53%

`CONFIG = "full_ft_baseline"`, `EXTRA = "ckpt_every_steps=1000000"`: each checkpoint is the whole model (about 1.1 GB), so this turns off the saves every 2,000 steps and keeps the one at the end of each epoch. It is the brief's Task 1 and the 100% baseline that the 5% design is compared with (brief reference 70.8%). The 5% limit does not apply to it. Result and interpretation: see the `full_ft_baseline` run section above.

### 2. Equal-budget ablation: which part of the design earns the gain?

All three use about 5% of the backbone, lr 3e-4, 10 epochs (about 1.18 h on an A100 80 GB, inside the 1.2 h budget) and identical settings otherwise. Only the adapter design differs.

| Run | Config | Design | Adapter % | Hold-out top-1 |
|---|---|---|---|---|
| A | `abl_lora_only` | LoRA r=116 on qkv | 4.984 | to run |
| B | `abl_adaptformer_only` | AdaptFormer b=232 on FFN | 4.998 | 71.09 |
| C | `abl_combined` | LoRA r=12 on qkv + AdaptFormer b=208 on FFN | 4.998 | 71.05 |

How to read it: if C beats both A and B by about 1 point or more, adapting attention and FFN together is better than spending the same budget on one of them. If A or B matches C, the simpler design is the better story. Rank 116 has never been run, so watch its first loss lines.

Reading so far (12 Oct): B and C tie (71.09 against 71.05), so adding LoRA on the attention to the FFN adapter gains nothing at an equal budget, and B is the simpler and faster design. Run A decides whether the FFN placement matters or only the number of adapter parameters.

### 3. One-change tuning on `abl_combined` (one config per change)

Each config is `abl_combined` with exactly one setting changed, so it is compared with the `abl_combined` run (C above). Leave `RUN` empty so each config gets its own output folder.

| Config | The one change (baseline value) | Why | Hold-out top-1 |
|---|---|---|---|
| `abl_combined` | none: the baseline, 10 epochs (the earlier run stopped at 7.8) | uses the whole time budget | 71.05 (the 7.8-epoch run: 71.29, so no gain from training longer) |
| `tune_label_smoothing` | label smoothing 0.1 (0.0) | the LoRA run got over-confident (train loss 0.12, hold-out loss rising); the brief hints at loss design | to run |
| `tune_lr_low` | lr 2e-4 (3e-4) | 3e-4 was never tuned | to run |
| `tune_lr_high` | lr 5e-4 (3e-4) | 3e-4 was never tuned | to run |
| `tune_head_init` | head init from 43 images per class (10) | zero-shot 35.7% vs the brief's 43.7% | in progress (snapshot, 12 Oct): zero-shot 43.30%; ahead of `abl_combined` at epochs 1 and 2 (65.72 / 67.22 against 64.56 / 66.97), behind at epochs 3 and 4 (67.08 / 67.54 against 67.53 / 68.45), and its hold-out loss rises faster (1.60 against 1.51 at epoch 4). The head-init pass added about 5 minutes. |

Status on 13 Oct: runs 3, 4, 5 and 8 are finished and were scored on the leaderboard (scores are not written here). Label smoothing and the lower learning rate (2e-4) each gave a clear gain over `abl_combined`, the higher learning rate (5e-4) was clearly worse, and LoRA only was about level with AdaptFormer only and the combination. Their hold-out numbers are added when the run files arrive. Runs 10 to 12 follow from this.

How to decide: one change worth a few tenths of a point cannot be told apart from noise on its own (about 0.3 points of sampling error per run). So after these runs, make one config with every change that went up, run it once, and keep it only if it beats `abl_combined` clearly (about 1 point). That combined run is the one to submit.

`tune_head_init` ends near 1.25 h on an A100 80 GB (estimate: the extra feature pass adds 4 to 5 minutes), which is over the 1.2 h budget but inside the +-10% tolerance. Cheap check first: `EXTRA = "max_train_steps=50"` and read the zero-shot line.

### 4. Later-layer adapters and final-model extras (written 12 Oct, none run on a GPU yet)

**Later layers (queue run 9).** `abl_adaptformer_last6`: AdaptFormer only in the last 6 of the 12 blocks with twice the bottleneck (464), 4.993% of the backbone. Everything else equals `abl_adaptformer_only` (all 12 blocks, 71.09%), so the pair shows whether the 5% is better spent where the fine-grained decision is made.

**Final-model extras.** Use them on the design that is finally chosen, not as separate experiments:

| Extra | How | What to know |
|---|---|---|
| Train on all 45 images per class | add `holdout_per_class=0` to `EXTRA`, with a new `RUN` name | about 5% more training data; the run has no hold-out, so it can only be scored on the leaderboard and its hold-out columns stay empty |
| Mirror averaging at prediction time | add `--tta flip` to the `common.submit` command | each image and its mirror image are predicted and the probabilities averaged; two forward passes, so report twice the single-pass latency |
| Check mirror averaging first | `python -m common.evaluate <run folder>` | prints hold-out top-1 with and without mirror averaging for a finished run, in about a minute and without training |

Moving the 5% between attention and FFN is covered by runs A to C above.

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
- Full fine-tuning (72.53%) beats the 5% design like for like (71.05% at 10 epochs; 71.29% at 7.78). The story for the report is how close 5% of the parameters gets (98% of the full fine-tuning accuracy) and what it costs or saves (latency 11.4 ms against 7.0 ms; 4.3M trained weights to store per task instead of 85.8M), not that it wins.
- Ablation finding so far: AdaptFormer alone (71.09%) equals LoRA + AdaptFormer (71.05%) at the same 5% budget and is faster (9.2 against 11.4 ms). Unless `abl_lora_only` or a tuning run changes the picture, the simpler AdaptFormer-only design is the one to present as the final 5% model, with the combination reported as an ablation that did not help.
- Open question for the professor: which model's predictions go to the leaderboard and to the 2 Phase 2 submissions: the full fine-tuning model, the 5% model, or one of each?
- Open question for the professor: an 80 GB A100 is counted here as 80 GB (budget 1.2 h). The LoRA run peaked at 6.7 GiB, so ask whether such a run may use the full 2 h.
- Zero-shot baseline for the report (0% trained, head set from class means): 43.30% with all 43 training images per class, 35.70% with 10 per class. The brief reports 43.7%.
- Base model: the professor ruled on 12 Oct 2026 that a different base checkpoint is not allowed (the IN21K-only weights were considered and dropped). Whether a larger input size than 224 px is allowed has not been asked yet.
- Hold-out numbers are not leaderboard numbers. Report both and say which is which.
- Leaderboard scores are not published by teams, and this repo is public, so they are not recorded here. Keep them in the report only.
