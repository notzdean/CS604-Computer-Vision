# Experiment log

Running record of every experiment for the ablation study and the report. One section per run, plus a summary table.
Source of truth for numbers is `results/results.csv` (one row per run); this file adds the interpretation.
Brief marks: motivation of design (most important), design, **ablation of each part**, top-1, trainable params, latency, GPU-hours, loss curves.

**Counting rule (confirmed by the professor, 11 Oct 2026):** trainable % = adapter params / backbone params (85.8M). The classifier head (7.69M) is excluded and shown separately. The <5% limit applies to the efficient-adaptation track only.
**Hold-out:** 2 labelled train images per class (20,000 images). These are *not* the val set. The CodaBench val leaderboard is the real check.

## Summary

| Run | Method | Config | Hold-out top-1 | Trainable % (adapter) | Head params | GPU-h (A100 80 GB unless noted) | Leaderboard val top-1 (not published) |
|---|---|---|---|---|---|---|---|
| zero-shot (head init only, 10 images per class) | none | `lora_baseline` before training | 35.70 | 0 | 7.69M (init only) | - | - |
| zero-shot (head init only, 43 images per class) | none | `tune_head_init` before training | 43.30 | 0 | 7.69M (init only) | - | - |
| lora_r8_full | LoRA r=8, qkv | `lora_baseline` | 68.78 | 0.344 | 7.69M (trained) | 0.76 | not recorded here |
| lora_adaptformer | LoRA r=12 qkv + AdaptFormer b=208 (FFN) | `lora_adaptformer` | 71.29 | 4.998 | 7.69M (trained) | 0.93 | not recorded here |
| full_ft_baseline | Full fine-tuning (required baseline) | `full_ft_baseline` | 72.53 | 100 | 7.69M (trained) | 1.13 | not recorded here |
| abl_combined | LoRA r=12 qkv + AdaptFormer b=208 (FFN), 10 epochs | `abl_combined` | 71.05 | 4.998 | 7.69M (trained) | 1.18 | not recorded here |
| abl_adaptformer_only | AdaptFormer b=232 (FFN) only, 10 epochs | `abl_adaptformer_only` | 71.09 | 4.998 | 7.69M (trained) | 1.04 | not recorded here |
| abl_lora_only | LoRA r=116 (qkv) only, 10 epochs | `abl_lora_only` | 70.76 | 4.984 | 7.69M (trained) | 1.17 (A100 40 GB) | not recorded here |
| abl_adaptformer_last6 | AdaptFormer b=464 (FFN), last 6 blocks only, 10 epochs | `abl_adaptformer_last6` | 70.17 | 4.993 | 7.69M (trained) | 1.01 | not recorded here |
| tune_label_smoothing | `abl_combined` + label smoothing 0.1 | `tune_label_smoothing` | 72.17 | 4.998 | 7.69M (trained) | 1.28 (A100 40 GB) | not recorded here |
| tune_lr_low | `abl_combined` + lr 2e-4 | `tune_lr_low` | 72.37 | 4.998 | 7.69M (trained) | 1.28 (A100 40 GB) | not recorded here |
| tune_lr_high | `abl_combined` + lr 5e-4 | `tune_lr_high` | 69.28 | 4.998 | 7.69M (trained) | 1.29 (A100 40 GB) | not recorded here |
| tune_head_init | `abl_combined` + head init from 43 images per class | `tune_head_init` | 69.98 | 4.998 | 7.69M (trained) | 1.26 | not recorded here |
| tune_ls_lr2e4 | `abl_combined` + label smoothing 0.1 + lr 2e-4 | `tune_ls_lr2e4` | 72.35 | 4.998 | 7.69M (trained) | 1.19 | not recorded here |
| tune_ls_lr1e4 | `abl_combined` + label smoothing 0.1 + lr 1e-4 | `tune_ls_lr1e4` | 73.02 | 4.998 | 7.69M (trained) | 1.19 | not recorded here |
| tune_ls_lr5e5 | `abl_combined` + label smoothing 0.1 + lr 5e-5 | `tune_ls_lr5e5` | 73.52 | 4.998 | 7.69M (trained) | 1.18 | not recorded here |
| tune_head_lr_half | `tune_ls_lr1e4` + head lr x0.5 (5e-5) | `tune_head_lr_half` | **74.14** | 4.998 | 7.69M (trained) | 1.27 (A100 40 GB) | not recorded here |
| tune_head_lr_quarter | `tune_ls_lr1e4` + head lr x0.25 (2.5e-5) | `tune_head_lr_quarter` | 74.02 | 4.998 | 7.69M (trained) | 1.18 | not recorded here |
| tune_head_lr_double | `tune_ls_lr1e4` + head lr x2 (2e-4) | `tune_head_lr_double` | 71.75 | 4.998 | 7.69M (trained) | 1.28 (A100 40 GB) | not recorded here |
| full_ft_ls | Full fine-tuning + label smoothing 0.1 | `full_ft_ls` | 73.98 | 100 | 7.69M (trained) | 1.13 | not recorded here |
| abl_linear_probe | Head only, backbone frozen (label smoothing 0.1, lr 1e-4, 10 epochs) | `abl_linear_probe` | 62.62 | 0 | 7.69M (trained) | 1.19 (A100 40 GB) | not recorded here |
| imp_droppath | `tune_head_lr_half` + stochastic depth 0.1 | `imp_droppath` | 74.09 | 4.998 | 7.69M (trained) | about 1.32 (A100 40 GB; 1.36 recorded, see section) | not recorded here |
| imp_ema | `tune_head_lr_half` + weight EMA 0.9998 | `imp_ema` | 74.19 | 4.998 | 7.69M (trained) | 1.27 (A100 40 GB) | not recorded here |
| imp_mixup | `tune_head_lr_half` + Mixup 0.8 / CutMix 1.0, 15 epochs | `imp_mixup` | 73.97 | 4.998 | 7.69M (trained) | 1.90 (A100 40 GB) | not recorded here |
| imp_trivialaug | `tune_head_lr_half` + TrivialAugment + random erasing 0.25 | `imp_trivialaug` | 74.28 | 4.998 | 7.69M (trained) | 1.43 (A100 40 GB) | not recorded here |
| imp_trivialaug at test size 288 | same model, scored at 288 px (no retraining; run 22) | `imp_trivialaug` + `--sizes 288` | **75.99** (76.23 with mirror averaging) | 4.998 | 7.69M (trained) | - | not recorded here |
| final_5pct_trivialaug | final 5% model: `imp_trivialaug` recipe on all 45 images per class (no hold-out) | `final_5pct_trivialaug` | - (no hold-out; leaderboard only) | 4.998 | 7.69M (trained) | 1.36 (A100 40 GB) | not recorded here |
| full_ft_llrd | Full fine-tuning + label smoothing + head at the backbone's rate + layer-wise lr decay 0.75, full 2.4 h budget (18.2 epochs) | `full_ft_llrd` | **76.26** | 100 | 7.69M (trained) | 2.27 (A100 40 GB) | not recorded here |

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
  - Still open: does the FFN placement matter, or only the number of adapter parameters? `abl_lora_only` (same budget, attention only) answers that. Answered below: LoRA alone also ties (70.76%).
- **Log:** `results/abl_adaptformer_only/log.jsonl`, `summary.json`.

### Notes that apply to the runs below

- **GPU:** Ryan's runs (`abl_lora_only`, `tune_label_smoothing`, `tune_lr_low`, `tune_lr_high`, `tune_head_lr_half`, `tune_head_lr_double`) ran on an A100 **40 GB**, where the budget is 2.4 h. They took about 1.27 h, so about 1.06 h in 48 GB-equivalent: well inside the budget. The rest ran on an A100 80 GB (budget 1.2 h).
- **Hold-out loss with label smoothing:** a model trained with label smoothing 0.1 is deliberately less confident, so its plain cross-entropy on the hold-out is higher. Compare hold-out loss only between runs that both use label smoothing, or both do not.
- **Latency** of the same LoRA + AdaptFormer design ranges from 11.3 to 12.4 ms across sessions, so gaps of about 1 ms are session noise.
- **Sampling error** is about 0.3 points per run and about 0.45 points for the difference between two runs (see above).

### abl_lora_only (13 Oct 2026, Ryan)

- **Setup:** `configs/abl_lora_only.yaml`: LoRA r=116 on every `qkv` and no AdaptFormer, 4.984% of backbone. Same lr (3e-4), 10 epochs and other settings as `abl_combined`. Run A of the equal-budget ablation.
- **Result:** hold-out top-1 70.76%, top-5 87.47%, hold-out loss 1.420.
- **Efficiency:** 4,276,224 adapter params = 4.984%. Latency 10.4 ms. Peak VRAM 6.8 GiB. Train 1.173 h on an A100 40 GB (0.98 h in 48 GB-equivalent), about 1,130 img/s.
- **Hold-out top-1 by epoch, the three equal-budget designs:**

  | Epoch | A: `abl_lora_only` | Hold-out loss | B: `abl_adaptformer_only` | C: `abl_combined` |
  |---|---|---|---|---|
  | 1 | 63.48 | 1.560 | 64.37 | 64.56 |
  | 2 | 65.97 | 1.494 | 66.47 | 66.97 |
  | 3 | 66.29 | 1.538 | 67.44 | 67.53 |
  | 4 | 67.23 | 1.562 | 68.08 | 68.45 |
  | 5 | 67.71 | 1.564 | 68.69 | 68.77 |
  | 6 | 68.78 | 1.539 | 69.45 | 69.59 |
  | 7 | 69.72 | 1.476 | 70.17 | 70.48 |
  | 8 | 70.30 | 1.449 | 70.62 | 70.77 |
  | 9 | 70.61 | 1.429 | 70.98 | 71.07 |
  | 10 (final) | 70.76 | 1.420 | 71.09 | 71.05 |

- **Training loss (mean per epoch):** 1.93, 1.06, 0.71, 0.48, 0.32, 0.21, 0.14, 0.10, 0.08, 0.07.
- **Interpretation:**
  - All three equal-budget designs tie: LoRA only 70.76, AdaptFormer only 71.09, combined 71.05. The spread (0.33) is inside the sampling error. At 5% of the backbone it does not matter whether the parameters go to attention, to the FFN, or to both.
  - Rank 116 trained without trouble: the first loss lines and the curve look like the other two designs. LoRA alone is slightly behind B and C at every epoch (by 0.3 to 1.2 points) and closes most of the gap by the end.
  - So the gain of the 5% design over LoRA r=8 (68.78) comes from the number of adapter parameters, not from where they sit.
- **Log:** `results/abl_lora_only/log.jsonl`, `summary.json`.

### abl_adaptformer_last6 (16 Oct 2026, Nicole)

- **Setup:** `configs/abl_adaptformer_last6.yaml`: AdaptFormer only in the last 6 of the 12 blocks, bottleneck 464 (twice that of `abl_adaptformer_only`), 4.993% of backbone. Everything else as `abl_adaptformer_only`.
- **Result:** hold-out top-1 70.17%, top-5 87.27%, hold-out loss 1.453.
- **Efficiency:** 4,283,616 adapter params = 4.993%. Latency 8.2 ms. Peak VRAM 3.6 GiB (half of the all-blocks design: no gradients flow through the first 6 blocks). Train 1.011 h on an A100 80 GB, about 1,300 img/s, the fastest adapter run.
- **Hold-out top-1 by epoch:** 63.57, 65.81, 66.75, 67.03, 68.06, 68.75, 69.54, 69.80, 70.02, 70.17 (final); hold-out loss 1.553 falling to 1.453. `abl_adaptformer_only` (all 12 blocks): 64.37, 66.47, 67.44, 68.08, 68.69, 69.45, 70.17, 70.62, 70.98, 71.09.
- **Training loss (mean per epoch):** 1.92, 0.98, 0.61, 0.40, 0.27, 0.18, 0.13, 0.10, 0.08, 0.08.
- **Interpretation:**
  - 0.92 points below AdaptFormer in all 12 blocks (71.09), about 2 times the sampling error of the difference: probably a real loss, but small. It is behind at every epoch from epoch 1.
  - Spending the 5% only in the later layers, where the fine-grained decision is made, does not help. Adapting the early layers too is worth more than a wider adapter late.
  - It is the cheapest design (8.2 ms, 3.6 GiB, 1.01 h). Worth one line in the report as the efficiency trade-off.
- **Log:** `results/abl_adaptformer_last6/log.jsonl`, `summary.json`.

### One-change tuning on `abl_combined`: label smoothing, learning rate, head init (12 to 13 Oct 2026, Ryan and lilian)

Each run is `abl_combined` (71.05%) with exactly one setting changed. All are LoRA r=12 + AdaptFormer b=208 (4,287,936 params, 4.998%), 10 epochs, 33,590 steps, peak VRAM 7.3 GiB.

| Config | Change | Who | Top-1 | Top-5 | vs `abl_combined` | Hold-out loss (final) | Train h (GPU) | Latency |
|---|---|---|---|---|---|---|---|---|
| `tune_label_smoothing` | label smoothing 0.1 | Ryan | 72.17 | 88.03 | +1.12 | 1.604 (smoothed, not comparable) | 1.276 (40 GB) | 11.9 ms |
| `tune_lr_low` | lr 2e-4 | Ryan | 72.37 | 88.52 | +1.32 | 1.258 | 1.282 (40 GB) | 12.3 ms |
| `tune_lr_high` | lr 5e-4 | Ryan | 69.28 | 85.92 | -1.77 | 1.818 | 1.285 (40 GB) | 12.1 ms |
| `tune_head_init` | head init from 43 images per class | lilian | 69.98 | 87.00 | -1.07 | 1.565 | 1.260 (80 GB) | 11.4 ms |

Hold-out top-1 by epoch:

| Epoch | `abl_combined` | `tune_label_smoothing` | `tune_lr_low` | `tune_lr_high` | `tune_head_init` |
|---|---|---|---|---|---|
| 1 | 64.56 | 64.71 | 66.32 | 62.20 | 65.72 |
| 2 | 66.97 | 67.60 | 68.58 | 65.08 | 67.22 |
| 3 | 67.53 | 68.75 | 69.49 | 65.83 | 67.08 |
| 4 | 68.45 | 69.21 | 70.30 | 66.59 | 67.54 |
| 5 | 68.77 | 70.06 | 70.68 | 66.82 | 67.98 |
| 6 | 69.59 | 70.72 | 71.29 | 67.66 | 68.53 |
| 7 | 70.48 | 71.33 | 71.58 | 68.19 | 69.20 |
| 8 | 70.77 | 71.83 | 72.04 | 68.75 | 69.76 |
| 9 | 71.07 | 72.14 | 72.34 | 69.23 | 69.95 |
| 10 | 71.05 | 72.17 | 72.37 | 69.28 | 69.98 |

- **Interpretation:**
  - **Learning rate:** 3e-4 was too high. 2e-4 gains 1.32 points and its hold-out loss keeps falling (1.43 to 1.26, flat only at epochs 2 to 3) instead of rising in the middle; 5e-4 loses 1.77 points and its hold-out loss climbs from 1.57 to 1.88, the clearest over-confidence of any run. Lower is better in this range.
  - **Label smoothing** gains 1.12 points. Its training loss stays high (1.52 at the end, against 0.07 without it), which is the point: the model is kept from becoming over-confident on 43 images per class.
  - Both gains are about 2.5 to 3 times the sampling error of the difference, so they are real. They led to runs 10 to 13 (both changes together).
  - **Head init from 43 images per class** gives the better start (zero-shot 43.30% against 35.70%, and ahead at epochs 1 and 2) but ends 1.07 points lower. From epoch 3 it falls behind and its hold-out loss rises to 1.64 at epoch 5. A likely reason: with all 43 training images used to set the head, the head already fits the training set, the training loss falls faster (1.58 against 1.87 in epoch 1), and the model overfits sooner at lr 3e-4. Not tested directly. Runs 18 and 20 try it again with the tuned recipe (label smoothing, lower learning rate).
- **Logs:** `results/<config>/log.jsonl`, `summary.json`.

### Label smoothing + lower learning rate: `tune_ls_lr2e4`, `tune_ls_lr1e4`, `tune_ls_lr5e5` (16 to 18 Oct 2026, Nicole)

`abl_combined` with label smoothing 0.1 and a lower learning rate (LoRA r=12 + AdaptFormer b=208, 4,287,936 adapter params = 4.998%). `head_lr_mult` is 1, so the head's learning rate drops with the adapters'. 10 epochs, A100 80 GB, about 1.18 to 1.19 h, 1,100 img/s, latency 11.3 to 11.8 ms, peak VRAM 7.3 GiB.

| Epoch | `tune_ls_lr2e4` | `tune_ls_lr1e4` | `tune_ls_lr5e5` | `tune_label_smoothing` (lr 3e-4) |
|---|---|---|---|---|
| 1 | 66.13 | 66.94 | 64.94 | 64.71 |
| 2 | 68.61 | 70.19 | 69.02 | 67.60 |
| 3 | 69.56 | 71.39 | 71.43 | 68.75 |
| 4 | 69.97 | 72.09 | 72.11 | 69.21 |
| 5 | 70.72 | 72.23 | 72.97 | 70.06 |
| 6 | 71.25 | 72.39 | 73.27 | 70.72 |
| 7 | 71.77 | 72.69 | 73.36 | 71.33 |
| 8 | 71.89 | 72.95 | 73.47 | 71.83 |
| 9 | 72.22 | 73.05 | 73.57 | 72.14 |
| 10 (final) | 72.35 | 73.02 | 73.52 | 72.17 |
| Top-5 (final) | 88.36 | 89.33 | 89.78 | 88.03 |
| Hold-out loss (final) | 1.554 | 1.420 | 1.363 | 1.604 |

- **Interpretation:**
  - With label smoothing on, lowering the learning rate keeps helping: 72.17 (3e-4), 72.35 (2e-4), 73.02 (1e-4), 73.52 (5e-5).
  - At 2e-4 the two changes do not add up: 72.35 with both, against 72.17 and 72.37 with each alone. They seem to fix the same problem (over-confidence).
  - 5e-5 is 0.50 above 1e-4, about the sampling error of the difference: level, or a small gain at most. Run 14 (3e-5) was skipped for that reason.
  - The lower the rate, the flatter the end of the curve: at 5e-5 the run gains only 0.25 points after epoch 6, so 10 epochs are enough.
  - Because the head's rate fell with the adapters' rate here, these runs do not say which of the two needed to be slower. The head-rate runs below separate them.
- **Logs:** `results/<config>/log.jsonl`, `summary.json`.

### Head learning rate: `tune_head_lr_half`, `tune_head_lr_quarter`, `tune_head_lr_double` (18 Oct 2026, Ryan and Nicole)

`tune_ls_lr1e4` (adapters at 1e-4, label smoothing 0.1, 73.02%) with only the head's learning rate changed through `head_lr_mult`. All four: LoRA r=12 + AdaptFormer b=208, 4,287,936 adapter params = 4.998%, 10 epochs, peak VRAM 7.3 GiB.

| Config | Head lr | Who | Top-1 | Top-5 | vs `tune_ls_lr1e4` | Hold-out loss | Train h (GPU) | Latency |
|---|---|---|---|---|---|---|---|---|
| `tune_head_lr_double` | 2e-4 (x2) | Ryan | 71.75 | 87.98 | -1.27 | 1.570 | 1.276 (40 GB) | 12.4 ms |
| `tune_ls_lr1e4` | 1e-4 (x1) | Nicole | 73.02 | 89.33 | 0 | 1.420 | 1.192 (80 GB) | 11.5 ms |
| `tune_head_lr_half` | 5e-5 (x0.5) | Ryan | **74.14** | **90.25** | +1.12 | **1.325** | 1.273 (40 GB) | 12.3 ms |
| `tune_head_lr_quarter` | 2.5e-5 (x0.25) | Nicole | 74.02 | 90.17 | +1.00 | 1.335 | 1.181 (80 GB) | 11.7 ms |

Hold-out top-1 by epoch:

| Epoch | x2 | x1 | x0.5 | x0.25 |
|---|---|---|---|---|
| 1 | 65.96 | 66.94 | 65.81 | 62.84 |
| 2 | 68.51 | 70.19 | 69.64 | 67.58 |
| 3 | 69.26 | 71.39 | 71.90 | 70.67 |
| 4 | 69.50 | 72.09 | 72.81 | 71.89 |
| 5 | 70.56 | 72.23 | 73.64 | 72.91 |
| 6 | 70.74 | 72.39 | 73.91 | 73.36 |
| 7 | 71.22 | 72.69 | 73.95 | 73.52 |
| 8 | 71.46 | 72.95 | 74.00 | 73.89 |
| 9 | 71.62 | 73.05 | 74.11 | 73.98 |
| 10 (final) | 71.75 | 73.02 | 74.14 | 74.02 |

- **Interpretation:**
  - The head's learning rate is the most sensitive setting found so far: from x2 to x0.5 is 2.4 points. A slower head is better down to x0.5; x0.25 is level with x0.5 (0.12 apart, inside the noise), so the optimum is around head lr 2.5e-5 to 5e-5.
  - `tune_head_lr_half` (74.14%) is the best 5% run and has the lowest hold-out loss of the label-smoothing runs (1.325). It is ahead of full fine-tuning without label smoothing (72.53) and level with it with label smoothing (73.98).
  - This explains the learning-rate sweep above: `tune_ls_lr5e5` has the head at 5e-5 too, but the adapters also at 5e-5, and reaches 73.52. With the adapters at 1e-4 and the head at 5e-5 it is 74.14. So the head wants a slow rate and the adapters a faster one, and the two should be set apart.
  - A slower head starts worse (x0.25 is 4 points behind x1 after epoch 1) and catches up by epoch 3 to 4. The head starts from class means (nearest-class-mean init), so a slow head keeps that well-placed start for longer while the adapters adapt the features under it.
  - For the report: this is the "classifier head" part of the design, with an ablation of four values.
- **Logs:** `results/<config>/log.jsonl`, `summary.json`.

### full_ft_ls (16 Oct 2026, Nicole)

- **Setup:** `configs/full_ft_ls.yaml`: `full_ft_baseline` with one change, label smoothing 0.1. Backbone lr 3e-5, head lr 3e-4 (`head_lr_mult` 10), weight decay 0.05, auto length.
- **Result:** hold-out top-1 73.98%, top-5 89.39%, hold-out loss 1.580 (smoothed).
- **Efficiency:** 85,798,656 trainable params (100%). Latency 7.1 ms. Peak VRAM 9.6 GiB. Train 1.133 h on an A100 80 GB (1.89 h in 48 GB-equivalent), 9.37 epochs (31,461 steps), about 1,080 img/s.
- **Hold-out top-1 by epoch:** 63.73, 66.40, 68.02, 69.51, 71.20, 72.28, 73.14, 73.89, 73.98, 73.98 (final, epoch 9.37). `full_ft_baseline` at the same epochs: 63.16, 65.58, 66.92, 67.81, 69.80, 70.43, 71.39, 72.16, 72.50, 72.53.
- **Interpretation:**
  - Label smoothing helps full fine-tuning by 1.45 points (72.53 to 73.98), about 3 times the sampling error of the difference. About the same gain as for the 5% design (+1.12).
  - Fair comparison now: full fine-tuning with label smoothing (73.98) against the best 5% run (74.14) is a tie. The 5% design matches full fine-tuning while training 5% of the backbone, but it got two more tuned settings (lower lr, slower head).
  - Its head runs at 3e-4, ten times the backbone. Given the head-rate finding above, run 21 (`full_ft_ls_headlow`, head at 3e-5) is the fair follow-up before the final comparison.
- **Log:** `results/full_ft_ls/log.jsonl`, `summary.json`.

### abl_linear_probe (21 Oct 2026, lilian)

- **Setup:** `configs/abl_linear_probe.yaml`: only the classifier head is trained (backbone frozen, no adapters, 0% of backbone) with the `tune_ls_lr1e4` recipe: label smoothing 0.1, lr 1e-4 (head at x1), weight decay 0.01, head initialised from class means (10 images per class), 10 epochs. Compare with `tune_ls_lr1e4` (same recipe plus the adapters, 73.02%).
- **Result:** hold-out top-1 62.62%, top-5 82.75%, hold-out loss 2.032 (smoothed). Zero-shot 35.70%.
- **Efficiency:** 0 adapter params; head 7.69M trained. Latency 8.5 ms (the same network as full fine-tuning, 7.0 ms; the gap is session noise). Peak VRAM 1.8 GiB. Train 1.192 h on an A100 40 GB (0.99 h in 48 GB-equivalent), 10 epochs (33,590 steps), about 1,120 img/s.
- **Re-run:** the first attempt ran on a slow session (about 645 img/s), was stopped by the budget guard at 6.36 epochs and was discarded. This re-run matches it epoch by epoch (48.62 against 48.60 at epoch 1, 61.57 against 61.55 at epoch 6), so the first attempt was only cut short, not different.
- **Hold-out by epoch:**

  | Epoch | Hold-out top-1 | Top-5 | Hold-out loss | `tune_ls_lr1e4` top-1 (adapters) |
  |---|---|---|---|---|
  | 1 | 48.62 | 73.46 | 2.822 | 66.94 |
  | 2 | 54.45 | 78.00 | 2.454 | 70.19 |
  | 3 | 57.91 | 79.91 | 2.269 | 71.39 |
  | 4 | 59.45 | 81.21 | 2.169 | 72.09 |
  | 5 | 60.79 | 81.83 | 2.106 | 72.23 |
  | 6 | 61.57 | 82.19 | 2.069 | 72.39 |
  | 7 | 62.20 | 82.59 | 2.047 | 72.69 |
  | 8 | 62.53 | 82.67 | 2.037 | 72.95 |
  | 9 | 62.64 | 82.73 | 2.033 | 73.05 |
  | 10 (final) | 62.62 | 82.75 | 2.032 | 73.02 |

- **Training loss (mean per epoch):** 4.18, 3.50, 3.15, 2.94, 2.79, 2.70, 2.62, 2.58, 2.55, 2.54 (`tune_ls_lr1e4`: 1.64 at the end).
- **Interpretation:**
  - Training the head alone adds 26.9 points over the class-mean head (35.70 to 62.62). The adapters add another 10.4 on the same recipe (73.02), and the tuned head rate 1.1 more (74.14). Of the 38.4 points from zero-shot to the best 5% run, about 70% come from training the head and 30% from adapting the backbone. For the report: the adapters are a clear, necessary gain, but most of the accuracy comes from a well-trained 10,000-class head on the pre-trained features.
  - The run converged: 62.53, 62.64, 62.62 over the last three epochs. Longer training would not close the gap.
  - It underfits, it does not overfit: its training loss stays at 2.54 against 1.64 with adapters (same label smoothing), and the hold-out loss falls to the end. The frozen features cannot separate the 10,000 classes well enough, which is exactly what the adapters fix.
  - The gap is wider at the start (18.3 points at epoch 1) than at the end (10.4): the adapters adapt the features fast, in the first epochs.
  - Speed: about 1,120 img/s, no faster than the adapter runs (about 1,100), although nothing is trained in the backbone. So these runs are limited by the data loader and the frozen forward pass, not by the backward pass. Heavier augmentation (`imp_trivialaug`, run 26) will probably slow the runs down; watch its img/s.
  - The head ran at 1e-4 (x1). The slower-head finding does not carry over to a probe, where the head is the only part that learns, so this number is a fair baseline as it is.
- **Log:** `results/abl_linear_probe/log.jsonl`, `summary.json`.

### imp_droppath (21 Oct 2026, lilian)

- **Setup:** `configs/imp_droppath.yaml`: `tune_head_lr_half` (run 15, 74.14%) with one change, stochastic depth 0.1 (`drop_path_rate`). LoRA r=12 + AdaptFormer b=208, lr 1e-4, head x0.5, label smoothing 0.1, 10 epochs. Queue run 23, the first of the section 5 improvement runs.
- **Result:** hold-out top-1 74.09%, top-5 90.19%, hold-out loss 1.251. Zero-shot 35.70%.
- **Efficiency:** 4,287,936 adapter params = 4.998%. Latency 11.4 ms. Peak VRAM 7.3 GiB. A100 40 GB, 10 epochs (33,590 steps), about 990 img/s (run 15: about 1,030, so stochastic depth costs no real speed).
- **Two processes in one folder:** at 76 minutes, while the run was at step ~33,000, a second training process started in the same folder (most likely the training cell was run again after the VS Code connection to Colab dropped, while the first process kept running on the VM). It resumed from the step-32,000 checkpoint and both ran side by side to the end, each at about half speed; both wrote to `log.jsonl` (the steps jump backwards from line 669) and to the results. Both trained the same model from the same checkpoint and ended at the same score (74.085% and 74.095% at epoch 10), so the result stands. The recorded 1.356 h is inflated by the shared GPU in the last minutes; the run alone would have taken about 1.32 h (step 32,900 at 77.7 min at full speed, then 690 steps at 990 img/s), 0.55 of the 2.4 h budget on a 40 GB card, so inside the budget either way.
- **Hold-out by epoch:**

  | Epoch | `imp_droppath` top-1 | Hold-out loss | `tune_head_lr_half` top-1 | Hold-out loss |
  |---|---|---|---|---|
  | 1 | 63.67 | 1.749 | 65.81 | 1.715 |
  | 2 | 68.44 | 1.495 | 69.64 | 1.484 |
  | 3 | 71.23 | 1.375 | 71.90 | 1.413 |
  | 4 | 72.17 | 1.324 | 72.81 | 1.355 |
  | 5 | 73.03 | 1.296 | 73.64 | 1.346 |
  | 6 | 73.48 | 1.270 | 73.91 | 1.327 |
  | 7 | 73.75 | 1.261 | 73.95 | 1.331 |
  | 8 | 73.94 | 1.255 | 74.00 | 1.324 |
  | 9 | 74.09 | 1.251 | 74.11 | 1.325 |
  | 10 (final) | 74.09 | 1.251 | 74.14 | 1.325 |

- **Training loss (mean per epoch):** 3.55, 2.88, 2.60, 2.43, 2.31, 2.23, 2.17, 2.13, 2.10, 2.09 (run 15: 1.79 at the end).
- **Interpretation:**
  - No accuracy gain: 74.09 against 74.14, a tie (the noise on a difference is about 0.45). Top-5 also ties (90.19 against 90.25).
  - It does make the model better calibrated: the hold-out loss is clearly lower (1.251 against 1.325, both with label smoothing) and keeps falling to the end, while run 15's flattens from epoch 6. The training loss stays higher (2.09 against 1.79). So stochastic depth works as a regulariser, but over-confidence was no longer what limited top-1 once label smoothing and the slow head were in.
  - It starts slower (2.1 points behind after epoch 1) and catches up by epoch 9. With a longer schedule it might pull ahead; that is only worth testing inside a longer run (Mixup, run 25), not on its own.
  - For the combined run (29): it does not meet the "about 0.5 points or more" rule, so leave it out of the 10-epoch recipe. If the combined run uses Mixup and 15 epochs, adding it costs nothing and may help; decide after run 25.
- **Log:** `results/imp_droppath/log.jsonl` (interleaved at the end, see above), `summary.json`.

### imp_ema (21 Oct 2026, lilian)

- **Setup:** `configs/imp_ema.yaml`: `tune_head_lr_half` (run 15, 74.14%) with one change, an exponential moving average of the trainable weights (decay 0.9998, a time constant of about 5,000 steps = 1.5 epochs). The per-epoch hold-out scores, the final score and `final.pt` all use the EMA weights. Queue run 24.
- **Result:** hold-out top-1 74.19%, top-5 90.28%, hold-out loss 1.325. Zero-shot 35.70%.
- **Efficiency:** 4,287,936 adapter params = 4.998%. Latency 11.9 ms (EMA adds nothing at inference). Peak VRAM 7.3 GiB. Train 1.271 h on an A100 40 GB (1.06 h in 48 GB-equivalent), 10 epochs (33,590 steps), about 1,020 img/s: the averaging step costs no measurable speed. One clean process this time.
- **Hold-out by epoch (EMA weights):**

  | Epoch | `imp_ema` top-1 | Hold-out loss | `tune_head_lr_half` top-1 | Gap |
  |---|---|---|---|---|
  | 1 | 66.42 | 1.704 | 65.81 | +0.61 |
  | 2 | 70.86 | 1.467 | 69.64 | +1.22 |
  | 3 | 72.77 | 1.375 | 71.90 | +0.87 |
  | 4 | 73.65 | 1.334 | 72.81 | +0.84 |
  | 5 | 73.90 | 1.317 | 73.64 | +0.26 |
  | 6 | 73.84 | 1.313 | 73.91 | -0.07 |
  | 7 | 73.95 | 1.315 | 73.95 | 0.00 |
  | 8 | 74.06 | 1.318 | 74.00 | +0.06 |
  | 9 | 74.12 | 1.325 | 74.11 | +0.01 |
  | 10 (final) | 74.19 | 1.325 | 74.14 | +0.05 |

- **Training loss (mean per epoch):** 3.19, 2.53, 2.27, 2.10, 1.99, 1.91, 1.86, 1.82, 1.80, 1.79, identical to run 15 to two decimals. The EMA does not touch training, and the same seed on the same GPU type gave the same trajectory, so this is an unusually clean comparison: the +0.05 at the end is the effect of the averaging alone, not run-to-run noise.
- **Interpretation:**
  - No gain at the end: 74.19 against 74.14, and the same hold-out loss (1.325).
  - A clear gain in the middle of training: up to +1.2 points at epoch 2, +0.8 at epochs 3 and 4, while the learning rate is high and the weights are noisy. From epoch 6 the cosine schedule has lowered the learning rate enough that the live weights are as smooth as their average, and the two curves meet. With a cosine decay to 1% of the peak rate, the schedule already does what the EMA would do.
  - So the EMA is worth having only for runs that stop early or use a flat learning rate; for our 10-epoch cosine runs it is free but useless. Leave it out of the combined run (29).
- **Pattern after two runs:** stochastic depth and EMA both tie with run 15. Both act on the weights and the training noise; neither changes what the model sees. The 10-epoch recipe with label smoothing and the slow head looks saturated on that side, so the remaining chances are on the data side: Mixup / CutMix (run 25), stronger augmentation (run 26), the test size (run 22) and training on all 45 images per class for the final model (run 30).
- **Log:** `results/imp_ema/log.jsonl`, `summary.json`.

### imp_mixup (21 to 22 Oct 2026, lilian)

- **Setup:** `configs/imp_mixup.yaml`: `tune_head_lr_half` (run 15, 74.14%) with Mixup 0.8 / CutMix 1.0 (timm `Mixup`, each batch is either blended or has a patch pasted from another image, labels mixed in proportion; label smoothing 0.1 applied inside the mixer), trained for 15 epochs instead of 10 because mixing makes the task harder. Queue run 25.
- **Result:** hold-out top-1 73.97%, top-5 90.42%, hold-out loss 1.357. Zero-shot 35.70%.
- **Efficiency:** 4,287,936 adapter params = 4.998%. Latency 11.9 ms. Peak VRAM 7.3 GiB. Train 1.895 h on an A100 40 GB (1.58 h in 48 GB-equivalent, inside the 2.4 h budget), 15 epochs (50,385 steps), about 1,020 img/s: mixing on the GPU costs no speed. One clean process.
- **Hold-out top-1 by epoch:** 61.59, 66.58, 69.25, 70.62, 71.67, 71.92, 72.81, 73.06, 73.25, 73.50, 73.78, 73.86, 74.00, 74.04, 73.97 (final). Run 15 (10 epochs) for comparison: 65.81 after epoch 1, 74.14 at the end.
- **Training loss (mean per epoch):** 4.92 falling to 3.55. It is the loss on mixed soft targets, so it cannot be compared with the other runs.
- **Interpretation:**
  - No gain: 73.97 against 74.14, 0.17 lower, inside the noise of a difference (about 0.45). Top-5 is 0.17 higher (90.42 against 90.25), also noise. The hold-out loss is slightly worse (1.357 against 1.325).
  - Not a case of too few epochs: the curve is flat over the last three epochs (74.00, 74.04, 73.97) while the learning rate decays, so 15 epochs were enough to converge. It needed 13 epochs to reach what run 15 reaches in 9 (74.0), and 50% more training time for the same result.
  - Why it does not help here, two likely reasons. First, Mixup and CutMix are cures for overfitting, and after label smoothing and the slow head our model no longer overfits much (run 15's hold-out loss keeps falling to the end). Second, for fine-grained classes the details that separate two classes are small, and blending two images or pasting a patch over the part that matters can hide them while the label still says the class is present; this mismatch is the motivation of SnapMix (Huang et al., AAAI 2021), which reports plain CutMix to be unreliable on fine-grained benchmarks.
  - The control run (28, 15 epochs without Mixup) is not needed: it was only planned in case Mixup won.
- **Pattern after three runs:** stochastic depth (74.09), weight EMA (74.19) and Mixup (73.97) all tie with run 15 (74.14). Three different regularisers giving the same number means the recipe has converged: what limits top-1 now is what the 5% adapters can learn from 43 images per class, not overfitting. The remaining levers that do not regularise are the test size (run 22) and more training data (all 45 images per class, run 30).
- **Log:** `results/imp_mixup/log.jsonl`, `summary.json`.

### imp_trivialaug (22 Oct 2026, lilian)

- **Setup:** `configs/imp_trivialaug.yaml`: `tune_head_lr_half` (run 15, 74.14%) with stronger augmentation in the data loader: TrivialAugment (one random photometric or geometric operation per image) and random erasing with probability 0.25, both on the 224 px crop. 10 epochs. Queue run 26.
- **Result:** hold-out top-1 74.28%, top-5 90.47%, hold-out loss 1.311. Zero-shot 35.70%.
- **Efficiency:** 4,287,936 adapter params = 4.998%. Latency 11.8 ms (augmentation is training-only). Peak VRAM 7.3 GiB. Train 1.429 h on an A100 40 GB (1.19 h in 48 GB-equivalent, inside the 2.4 h budget), 10 epochs (33,590 steps), about 915 img/s against about 1,020 without the extra augmentation: as predicted, the CPU-side augmentation makes the run partly loader-bound and about 12% slower.
- **One resume:** the session dropped after step 16,000 (42.6 min) and the run resumed from the step-14,000 checkpoint, so steps 14,050 to 16,000 appear twice in `log.jsonl` (lines 324 to 362 repeat them). One process at a time, the same data order, so the result stands. The recorded time continues from the checkpoint and does not count the lost 4 minutes.
- **Hold-out top-1 by epoch:** 64.97, 69.25, 71.52, 72.66, 73.56, 73.72, 73.91, 74.23, 74.24, 74.28 (final). Run 15: 65.81, 69.64, 71.90, 72.81, 73.64, 73.91, 73.95, 74.00, 74.11, 74.14.
- **Training loss (mean per epoch):** 3.44 falling to 2.04 (run 15: 1.79 at the end): the harder inputs keep the training loss higher.
- **Interpretation:**
  - The highest 5% number so far, but a tie: +0.14 top-1 (74.28 against 74.14), +0.22 top-5 (90.47 against 90.25), slightly lower hold-out loss (1.311 against 1.325). All inside the noise of a difference (about 0.45).
  - Slightly behind for the first 7 epochs, ahead from epoch 8 (+0.23 at epoch 8), still creeping up at the end. Augmentation pays off late, so a longer schedule might widen the gap a little, but Mixup at 15 epochs showed that longer training alone does not help this recipe.
  - Cost: about 12% more training time (loader-bound), nothing at inference.
- **Pattern after four runs:** stochastic depth 74.09, EMA 74.19, Mixup 73.97, TrivialAugment 74.28, against 74.14 for run 15. Four different regularisers land within 0.3 points of each other. The recipe has converged and none of them clears the "about 0.5 points" rule for the combined run (29).
- **Log:** `results/imp_trivialaug/log.jsonl` (steps 14,050 to 16,000 logged twice, see above), `summary.json`.

### Test size, run 22 (22 Oct 2026, lilian)

- **Setup:** no training. `python -m common.evaluate <imp_trivialaug run folder> --sizes 224 256 288`: the finished `imp_trivialaug` model (trained at 224 px) scored on the same 20,000 hold-out images at three test sizes (resize to size x 248/224, centre crop; position embeddings interpolated by timm), each with and without mirror averaging.
- **Result:**

  | Test size | Top-1 | Top-5 | Hold-out loss | Top-1 with mirror averaging | Latency (batch 1) |
  |---|---|---|---|---|---|
  | 224 (as trained) | 74.28 | 90.47 | 1.311 | 74.67 (+0.39) | 11.4 ms |
  | 256 | 75.78 (+1.50) | 91.25 | 1.238 | 76.00 (+0.22) | 11.7 ms |
  | 288 | **75.99 (+1.71)** | **91.50** | **1.221** | **76.23** (+0.25) | 11.7 ms |
  | 320 | 75.28 (+1.00) | 91.16 | 1.260 | 75.55 (+0.27) | 38.9 ms (an outlier of the timing, see below) |
  | 352 | 74.06 (-0.22) | 90.56 | 1.326 | 74.35 (+0.29) | 11.6 ms |

- **Interpretation:**
  - The largest gain since the head learning rate, and free: +1.71 top-1 at 288 px without retraining. Every metric improves together (top-5 +1.0, hold-out loss 1.311 to 1.221), so it is not a fluke of top-1.
  - It is a paired comparison (the same model on the same 20,000 images), so its noise is far smaller than the 0.45 points between two separate runs: the gain is certain.
  - Why: training uses random crops covering 35 to 100% of the image, resized to 224 px, so objects look larger in training than in the 224 px centre crop used at test. A larger test image restores the object size the model learned on (FixRes, Touvron et al. 2019). Fine-grained classes also gain from the extra detail.
  - The optimum is 288 px (second check, 22 Oct): +1.50 at 256, +1.71 at 288, then down to +1.00 at 320 and -0.22 at 352. Too large a test image makes objects look larger than in training, and the interpolated position embeddings drift further from the trained 14 x 14 grid. Sizes between (272, 304) would differ by a few tenths at most; not worth checking.
  - The 38.9 ms latency at 320 px is a timing outlier (352 px, with more patches, took 11.6 ms), most likely a slow GPU kernel choice for that one shape; re-measure before quoting any latency other than 224 and 288.
  - Mirror averaging adds +0.22 to +0.39 on top, at twice the inference cost; at 288 px both together give 76.23%.
  - Cost: latency at batch 1 barely moves (11.4 to 11.7 ms) because a single image does not fill the A100; the compute per image grows with the number of patches (196 at 224, 324 at 288: about 1.7 times), which shows in throughput at large batch sizes. Report the latency at the size used, and say how it was measured.
  - It applies to every model, not only the 5% one. For a fair comparison in the report, score the full fine-tuning models (`full_ft_ls`, run 21, run 27) at the same test size before comparing the two tracks.
- **Decision:** submit the final 5% model (run 30) at **288 px** (`python -m common.submit <run> --team XX --split val --size 288`). Mirror averaging is optional (+0.25, twice the latency).
- **Follow-up still open:** the same check on `full_ft_ls` (Nicole's Drive) and on run 27 when it finishes, for the fair track comparison.

### final_5pct_trivialaug, run 30 (22 Oct 2026, lilian)

- **Setup:** `configs/final_5pct_trivialaug.yaml`: the `imp_trivialaug` recipe (LoRA r=12 + AdaptFormer b=208, lr 1e-4, head x0.5, label smoothing 0.1, TrivialAugment + random erasing 0.25, 10 epochs) trained on all 45 images per class (`holdout_per_class: 0`, 450,000 images, 4.7% more than the hold-out runs). This is the final model of the efficient-adaptation track.
- **Result:** no local score: there is no hold-out, so the hold-out columns are empty and `common.evaluate` refuses the run. It is judged on the CodaBench leaderboard only (scores are not written in this public repo).
- **Efficiency:** 4,287,936 adapter params = 4.998% of the backbone; head 7.69M (not counted). Latency 11.8 ms at 224 px (about 11.7 ms at 288 px, see run 22). Peak VRAM 7.3 GiB. Train 1.364 h on an A100 40 GB (1.14 h in 48 GB-equivalent, 0.57 of the 2.4 h budget), 10 epochs (35,150 steps), about 950 img/s. One clean process, no resume.
- **Training loss (mean per epoch):** 3.43, 2.78, 2.52, 2.37, 2.26, 2.18, 2.12, 2.08, 2.06, 2.04, the same curve as `imp_trivialaug` (3.44 to 2.04) to within 0.01. With 4.7% more images the training behaves identically, which is the only check available without a hold-out: the run trained as expected.
- **Expected accuracy:** the same recipe scored 74.28% at 224 px and 75.99% at 288 px on the hold-out; the extra 2 images per class should add a little on top. The leaderboard is the real number.
- **Submission:** `python -m common.submit <run folder> --team XX --split val --size 288` (288 px is the best test size, run 22); `--tta flip` adds about 0.25 at twice the inference cost. For Phase 2 (test set from 31 Oct) use the same command with `--split test`.
- **Log:** `results/final_5pct_trivialaug/log.jsonl`, `summary.json`.

### full_ft_llrd, run 27 (22 to 23 Oct 2026, lilian)

- **Setup:** `configs/full_ft_llrd.yaml`: full fine-tuning (every weight trained) with label smoothing 0.1, the head at 3e-5 (the backbone's old flat rate) and layer-wise learning-rate decay 0.75 (top block 7.5e-5, first block 3.2e-6, embeddings 2.4e-6). Auto length (`epochs: null`). It ran on an A100 40 GB **without** the step cap, so it filled that card's 2.4 h budget: 18.2 epochs, about twice `full_ft_ls` (9.37 epochs on an 80 GB card).
- **Result:** hold-out top-1 **76.26%**, top-5 90.91%, hold-out loss 1.468 (smoothed). Zero-shot 35.70%. The best run of either track at 224 px.
- **Efficiency:** 85,798,656 trainable params (100%). Latency 6.9 ms at 224 px. Train 2.274 h on an A100 40 GB = 1.895 h in 48 GB-equivalent, inside the 2 h x 48 GB budget (95% of it). 61,131 steps, about 1,040 img/s. One training process. The training cell was run once more after the run had finished: that process loaded the finished checkpoint, evaluated it again (the duplicate last eval line, same score) and rewrote `summary.json`, whose peak VRAM (2.5 GiB) therefore comes from evaluation only; the training peak is about 9.6 GiB as in `full_ft_ls`.
- **Hold-out top-1 by epoch:**

  | Epoch | `full_ft_llrd` | Hold-out loss | `full_ft_ls` | Hold-out loss |
  |---|---|---|---|---|
  | 1 | 67.32 | 1.596 | 63.73 | 1.658 |
  | 2 | 71.30 | 1.384 | 66.40 | 1.551 |
  | 3 | 73.29 | 1.301 | 68.02 | 1.558 |
  | 4 | 73.68 | 1.283 | 69.51 | 1.540 |
  | 5 | 74.40 | 1.265 | 71.20 | 1.516 |
  | 6 | 74.53 | 1.289 | 72.28 | 1.520 |
  | 7 | 74.68 | 1.313 | 73.14 | 1.534 |
  | 8 | 74.81 | 1.325 | 73.89 | 1.557 |
  | 9 | 75.17 | 1.342 | 73.98 (final, 9.37) | 1.580 |
  | 10 | 75.40 | 1.357 | | |
  | 11 | 75.39 | 1.380 | | |
  | 12 | 75.61 | 1.393 | | |
  | 13 | 75.87 | 1.409 | | |
  | 14 | 75.92 | 1.430 | | |
  | 15 | 76.06 | 1.440 | | |
  | 16 | 76.18 | 1.460 | | |
  | 17 | 76.20 | 1.465 | | |
  | 18 | 76.24 | 1.465 | | |
  | 18.2 (final) | 76.26 | 1.468 | | |

- **Training loss (mean per epoch):** 3.01, 2.37, 2.08, 1.89, 1.75, ... 1.39 at the end (`full_ft_ls`: 1.45 at the end).
- **Interpretation:**
  - +2.28 points over `full_ft_ls` (73.98). Two things changed besides layer-wise decay: the head's rate (3e-4 to 3e-5) and the length (9.37 to 18.2 epochs). The per-epoch curves separate part of it: at epoch 9 this run is already 1.19 points ahead (75.17 against 73.98), although its learning rate is still near the middle of its cosine schedule there, while `full_ft_ls` has annealed fully. So at equal epochs layer-wise decay plus the slower head are worth at least about 1.2 points; the second half of the budget adds about 1.1 more (75.17 to 76.26).
  - Which of the two (layer-wise decay or the slower head) does the work is open until Nicole's run 21 (`full_ft_ls_headlow`, slower head only) is in.
  - It learns much faster at the start: +3.6 points after epoch 1, +4.9 after epoch 2. Keeping the early layers close to the pre-trained weights lets the late layers and the head adapt quickly (Kumar et al. 2022; the BEiT / MAE recipe).
  - Over-confidence comes back late: the hold-out loss is lowest at epoch 5 (1.265) and rises to 1.468 while top-1 keeps climbing. More epochs still raise accuracy, but the model grows more confident on the errors it makes.
  - Close to converged: +0.20 over the last three epochs.
  - Against the 5% track: at 224 px it is now 1.98 points above the best 5% run (`imp_trivialaug`, 74.28) and 2.12 above `tune_head_lr_half`. The earlier "tie" was against a full fine-tuning baseline without layer-wise decay. Caveat for fairness: this run used 95% of the budget, the 5% runs only about 55% (10 epochs on a 40 GB card), and the 5% recipe was never run for the whole budget without Mixup (`imp_ep15_control` was skipped; Mixup at 15 epochs did not gain, and `abl_combined` at 10 epochs did not beat 7.78).
  - Inference: 6.9 ms against about 11.8 ms for the 5% model at 224 px (the adapters add layers), but 85.8M weights to store per task instead of 4.3M.
- **Next:** (a) the test-size check on this run (`--sizes 224 288`), so both tracks are compared at their best test size; (b) a final full fine-tuning model on all 45 images per class with this recipe (`final_full_ft_llrd`), for the Phase 2 submission of the full fine-tuning track.
- **Log:** `results/full_ft_llrd/log.jsonl` (the last eval line is repeated, see above), `summary.json` (peak VRAM from the evaluation-only re-run).

## Next runs (planned 11 Oct 2026)

### Run queue

Take a run marked "open" and tell the team before you start, so no run is done twice. How to run one is in `README.md`. Every run uses `RUN = ""`; leave `EXTRA` empty unless a value is given here. Each takes about 1.2 h on an A100 80 GB.

| # | `CONFIG` | `EXTRA` | What it is | Owner | Status |
|---|---|---|---|---|---|
| 1 | `full_ft_baseline` | `ckpt_every_steps=1000000` | required baseline: the whole model is trained | Nicole | done 11 Oct: 72.53% |
| 2 | `abl_combined` | | our design for 10 epochs: the comparison point for runs 3 to 8 | Nicole | done 11 Oct: 71.05% |
| 3 | `tune_label_smoothing` | | run 2 + label smoothing 0.1 | Ryan | done 13 Oct: 72.17% |
| 4 | `tune_lr_high` | | run 2 + learning rate 5e-4 | Ryan | done 13 Oct: 69.28% |
| 5 | `tune_lr_low` | | run 2 + learning rate 2e-4 | Ryan | done 13 Oct: 72.37% |
| 6 | `tune_head_init` | | run 2 + classifier start from 43 images per class (about 1.25 h) | lilian | done 13 Oct: 69.98% |
| 7 | `abl_adaptformer_only` | | AdaptFormer alone at 5%, for the report | lilian | done 12 Oct: 71.09% |
| 8 | `abl_lora_only` | | LoRA alone at 5%, for the report | Ryan | done 13 Oct: 70.76% |
| 9 | `abl_adaptformer_last6` | | AdaptFormer in the last 6 blocks only, twice the bottleneck; compare with run 7 | Nicole | done 16 Oct: 70.17% |
| 10 | `tune_ls_lr2e4` | | run 2 + label smoothing 0.1 + learning rate 2e-4 (the two changes that each helped) | Nicole | done 16 Oct: 72.35% |
| 11 | `tune_ls_lr1e4` | | run 10 with learning rate 1e-4 (does lower still help?) | Nicole | done 16 Oct: 73.02% |
| 12 | `full_ft_ls` | | full fine-tuning + label smoothing 0.1, so the baseline gets the same trick | Nicole | done 16 Oct: 73.98% |
| 13 | `tune_ls_lr5e5` | | run 11 with learning rate 5e-5 (does lower still help?) | Nicole | done 18 Oct: 73.52%, level with run 11 |
| 14 | `tune_ls_lr3e5` | | run 11 with learning rate 3e-5; only if run 13 beats run 11 | - | skipped: run 13 was level with run 11 |
| 15 | `tune_head_lr_half` | | run 11 with the head at half the adapters' learning rate (head 5e-5) | Ryan | done 18 Oct: 74.14%, best 5% run so far |
| 16 | `tune_head_lr_double` | | run 11 with the head at twice the adapters' learning rate (head 2e-4) | Ryan | done 18 Oct: 71.75%, clearly worse |
| 17 | `abl_linear_probe` | | only the head is trained (backbone frozen, no adapters); compare with run 11 to see what the adapters add | lilian | done 21 Oct: 62.62% (re-run; the first attempt stopped at 6.36 epochs on a slow session and was discarded) |
| 18 | `tune_ls_lr1e4_init43` | | run 11 + head start from all 43 images per class (about 1.25 h) | Nicole | running 18 Oct |
| 19 | `tune_head_lr_quarter` | | run 15 with the head at a quarter of the adapters' rate (head 2.5e-5) | Nicole | done: 74.02%, level with run 15 |
| 20 | `tune_head_lr_half_init43` | | run 15 + head start from all 43 images per class (about 1.25 h) | open (Nicole) | |
| 21 | `full_ft_ls_headlow` | | run 12 with the head at the backbone's rate (3e-5 instead of 3e-4), fair comparison | open (friend) | |
| 22 | no training | | test-size check on a finished run: `python -m common.evaluate <run folder> --sizes 224 256 288` (minutes; see section 5) | lilian | done 22 Oct on `imp_trivialaug`: 288 px gives 75.99% (+1.71), 76.23% with mirror averaging; 288 is the optimum (320: +1.00, 352: -0.22). Follow-up: the same check on `full_ft_ls` |
| 23 | `imp_droppath` | | run 15 + stochastic depth 0.1 | lilian | done 21 Oct: 74.09%, tie with run 15 (lower hold-out loss) |
| 24 | `imp_ema` | | run 15 + weight EMA 0.9998 (hold-out and final.pt use the averaged weights) | lilian | done 21 Oct: 74.19%, tie with run 15 (ahead only mid-training) |
| 25 | `imp_mixup` | | run 15 + Mixup 0.8 / CutMix 1.0, 15 epochs. **A100 40 GB only** (about 1.9 h of the 2.4 h budget) | lilian | done 22 Oct: 73.97%, tie with run 15 (converged; 50% more time) |
| 26 | `imp_trivialaug` | | run 15 + TrivialAugment + random erasing 0.25; stop if img/s falls well below 1,000 | lilian | done 22 Oct: 74.28%, tie with run 15 (about 915 img/s, 12% slower) |
| 27 | `full_ft_llrd` | none on an A100 80 GB; `max_train_steps=31461` on an A100 40 GB | run 21 + layer-wise lr decay 0.75. The config fills the budget (`epochs: null`), so on a 40 GB card (2.4 h) it would train about 19 epochs instead of `full_ft_ls`'s 9.37; the step cap keeps the length equal. Checkpoints are already once per epoch in the config | lilian | done 23 Oct: **76.26%**, run on a 40 GB card without the cap (18.2 epochs, 2.27 h, inside the budget) |
| 28 | `imp_ep15_control` | | run 15 for 15 epochs, A100 40 GB; only if run 25 beats run 15 | - | skipped: run 25 did not beat run 15 |
| 29 | (written after 22 to 27) | | run 15 + every change from 22 to 26 that gained about 0.5 points or more | - | not needed: none of runs 23 to 26 cleared 0.5 points (all within 0.3 of run 15). The final 5% recipe stays run 15, optionally with TrivialAugment (the nominal best, 74.28, no inference cost), plus the test size from run 22 |
| 30 | `final_5pct_trivialaug` | | final 5% model: run 26's recipe (TrivialAugment, 74.28%) on all 45 images per class (`holdout_per_class: 0` is in the config); no hold-out, leaderboard only; A100 40 GB (about 1.5 h) | lilian | done 22 Oct: trained cleanly (1.36 h); submit at `--size 288` |
| 31 | no training | | fair evaluation of full fine-tuning: `python -m common.evaluate <full_ft_llrd folder> --sizes 224 288` (single pass and mirror averaging at both sizes, like run 22 for the 5% model) | open (lilian) | |
| 32 | `full_ft_llrd_10ep` | | equal length: `full_ft_llrd` for 10 epochs (33,590 steps), the length of the 5% runs. A100 40 GB preferred (about 1.2 h); then `--sizes 224 288` | open (lilian) | |
| 33 | `imp_trivialaug_fullbudget` | | equal budget: `imp_trivialaug` filling the 2.4 h budget of an A100 40 GB (about 16 epochs), like `full_ft_llrd`. A100 40 GB only; then `--sizes 224 288` | open (lilian) | |

When a run finishes, copy `log.jsonl` and `summary.json` from `My Drive/cs604_runs/<CONFIG>/` into `results/<CONFIG>/` and push those two files. Do not upload to CodaBench; one person handles submissions.

### 1. Full fine-tuning (required track): done, 72.53%

`CONFIG = "full_ft_baseline"`, `EXTRA = "ckpt_every_steps=1000000"`: each checkpoint is the whole model (about 1.1 GB), so this turns off the saves every 2,000 steps and keeps the one at the end of each epoch. It is the brief's Task 1 and the 100% baseline that the 5% design is compared with (brief reference 70.8%). The 5% limit does not apply to it. Result and interpretation: see the `full_ft_baseline` run section above.

### 2. Equal-budget ablation: which part of the design earns the gain?

All three use about 5% of the backbone, lr 3e-4, 10 epochs (about 1.18 h on an A100 80 GB, inside the 1.2 h budget) and identical settings otherwise. Only the adapter design differs.

| Run | Config | Design | Adapter % | Hold-out top-1 |
|---|---|---|---|---|
| A | `abl_lora_only` | LoRA r=116 on qkv | 4.984 | 70.76 |
| B | `abl_adaptformer_only` | AdaptFormer b=232 on FFN | 4.998 | 71.09 |
| C | `abl_combined` | LoRA r=12 on qkv + AdaptFormer b=208 on FFN | 4.998 | 71.05 |

How to read it: if C beats both A and B by about 1 point or more, adapting attention and FFN together is better than spending the same budget on one of them. If A or B matches C, the simpler design is the better story. Rank 116 has never been run, so watch its first loss lines.

Reading so far (12 Oct): B and C tie (71.09 against 71.05), so adding LoRA on the attention to the FFN adapter gains nothing at an equal budget, and B is the simpler and faster design. Run A decides whether the FFN placement matters or only the number of adapter parameters.

Result (13 Oct, files added 21 Oct): A ties too (70.76). All three designs are within 0.33 points, inside the sampling error, so at 5% of the backbone the placement (attention, FFN or both) does not matter; the number of adapter parameters does.

### 3. One-change tuning on `abl_combined` (one config per change)

Each config is `abl_combined` with exactly one setting changed, so it is compared with the `abl_combined` run (C above). Leave `RUN` empty so each config gets its own output folder.

| Config | The one change (baseline value) | Why | Hold-out top-1 |
|---|---|---|---|
| `abl_combined` | none: the baseline, 10 epochs (the earlier run stopped at 7.8) | uses the whole time budget | 71.05 (the 7.8-epoch run: 71.29, so no gain from training longer) |
| `tune_label_smoothing` | label smoothing 0.1 (0.0) | the LoRA run got over-confident (train loss 0.12, hold-out loss rising); the brief hints at loss design | 72.17 (+1.12) |
| `tune_lr_low` | lr 2e-4 (3e-4) | 3e-4 was never tuned | 72.37 (+1.32) |
| `tune_lr_high` | lr 5e-4 (3e-4) | 3e-4 was never tuned | 69.28 (-1.77) |
| `tune_head_init` | head init from 43 images per class (10) | zero-shot 35.7% vs the brief's 43.7% | 69.98 (-1.07): zero-shot 43.30%, ahead at epochs 1 and 2, behind from epoch 3, hold-out loss rises faster. The head-init pass added about 5 minutes. |

Status on 13 Oct: runs 3, 4, 5 and 8 are finished and were scored on the leaderboard (scores are not written here). Label smoothing and the lower learning rate (2e-4) each gave a clear gain over `abl_combined`, the higher learning rate (5e-4) was clearly worse, and LoRA only was about level with AdaptFormer only and the combination. Their hold-out numbers are added when the run files arrive. Runs 10 to 12 follow from this. Status on 16 Oct: on the leaderboard, combining label smoothing with lr 2e-4 (run 10) was no better than either change alone, but label smoothing with lr 1e-4 (run 11) gave a further clear gain and is the best 5% run so far, ahead of the full fine-tuning baseline. Runs 13 and 14 continue the learning-rate sweep downwards. Note that `lr` also sets the classifier head's learning rate (`head_lr_mult` 1).

Status on 18 Oct: the head's learning rate is the most sensitive setting found so far. A slower head (half the adapters' rate, run 15) gave the best 5% run, a faster head (twice, run 16) was clearly worse, and lowering the adapters' rate from 1e-4 to 5e-5 (run 13) made little difference. Full fine-tuning gained clearly from label smoothing (run 12) and is now level with the best 5% run, so it gets the slower head too (run 21). Adapters in the last 6 blocks only (run 9) did worse than in all 12. Runs 19 to 21 follow from this; run 14 was skipped.

Status on 21 Oct: the hold-out files of runs 3 to 13, 15 to 17 and 19 are in `results/` and have run sections above. The hold-out numbers agree with the leaderboard reading: the best 5% run is `tune_head_lr_half` (74.14%), with `tune_head_lr_quarter` level (74.02%), against 73.98% for full fine-tuning with label smoothing. Still open: runs 18, 20 and 21. Run 17 (linear probe, re-run) finished on 21 Oct: 62.62%, so the adapters add about 10 points over a trained head alone.

How to decide: one change worth a few tenths of a point cannot be told apart from noise on its own (about 0.3 points of sampling error per run). So after these runs, make one config with every change that went up, run it once, and keep it only if it beats `abl_combined` clearly (about 1 point). That combined run is the one to submit.

`tune_head_init` ends near 1.25 h on an A100 80 GB (estimate: the extra feature pass adds 4 to 5 minutes), which is over the 1.2 h budget but inside the +-10% tolerance. Cheap check first: `EXTRA = "max_train_steps=50"` and read the zero-shot line.

### 4. Later-layer adapters and final-model extras (written 12 Oct)

**Later layers (queue run 9).** `abl_adaptformer_last6`: AdaptFormer only in the last 6 of the 12 blocks with twice the bottleneck (464), 4.993% of the backbone. Everything else equals `abl_adaptformer_only` (all 12 blocks, 71.09%), so the pair shows whether the 5% is better spent where the fine-grained decision is made. Result (16 Oct): 70.17%, 0.92 below all 12 blocks, so no; see its run section. The final-model extras below have not been run yet.

**Final-model extras.** Use them on the design that is finally chosen, not as separate experiments:

| Extra | How | What to know |
|---|---|---|
| Train on all 45 images per class | add `holdout_per_class=0` to `EXTRA`, with a new `RUN` name | about 5% more training data; the run has no hold-out, so it can only be scored on the leaderboard and its hold-out columns stay empty |
| Mirror averaging at prediction time | add `--tta flip` to the `common.submit` command | each image and its mirror image are predicted and the probabilities averaged; two forward passes, so report twice the single-pass latency |
| Check mirror averaging first | `python -m common.evaluate <run folder>` | prints hold-out top-1 with and without mirror averaging for a finished run, in about a minute and without training |

Moving the 5% between attention and FFN is covered by runs A to C above.

### 5. Regularisation and fine-tuning recipe (planned 21 Oct 2026, queue runs 22 to 30)

Why: at an equal 5% budget the adapter design did not matter (LoRA, AdaptFormer and both tie), and the PEFT literature on few-shot fine-grained benchmarks (VTAB-1k, FGVC) shows the same: methods land within about 1 point of each other. Every gain so far came from regularisation and from how the head is trained (label smoothing, lower lr, slower head). Our recipe still lacks the regularisers of the standard ViT fine-tuning recipes (DeiT; Steiner et al., "How to train your ViT", the paper behind the *augreg* checkpoint; BEiT and MAE fine-tuning): training used only random crops, flips and label smoothing.

Each improvement run is `tune_head_lr_half` (run 15, 74.14%) with one change. Gains under about 0.5 points are inside the noise of one run, so the winners are combined in one run (29) and only kept if it clearly beats run 15.

| Run | Change | What it does | Typical gain in the literature |
|---|---|---|---|
| 22 | test at 256 / 288 px (no training) | random crops make objects look larger in training than the 224 px centre crop at test; a slightly larger test size fixes the mismatch (FixRes, Touvron et al. 2019). Latency grows with the size. | +0.3 to 1. **Result: +1.71 at 288 px** (75.99%; 76.23% with mirror averaging), the only clear gain of section 5 |
| 23 | stochastic depth 0.1 | randomly skips a block's attention or FFN branch while training | +0.3 to 1. **Result: 74.09%, a tie** (hold-out loss lower: 1.251 against 1.325) |
| 24 | weight EMA 0.9998 | averages the weights over the last 1.5 epochs or so (a cheap cousin of model soups, Wortsman et al. 2022) | +0.2 to 0.5. **Result: 74.19%, a tie** (up to +1.2 mid-training, none at the end) |
| 25 | Mixup / CutMix, 15 epochs | trains on blended pairs of images and labels; needs more epochs, so A100 40 GB only (budget 2.4 h) | +0.5 to 2. **Result: 73.97%, a tie** (converged at 15 epochs) |
| 26 | TrivialAugment + random erasing | more variety from the same 45 images | +0.5 to 1.5 (if the loader keeps up). **Result: 74.28%, a tie** (12% slower, loader-bound) |
| 27 | layer-wise lr decay 0.75 (full fine-tuning) | early layers change less than late ones, so general features are kept (Kumar et al. 2022; BEiT / MAE recipe) | +0.5 to 1. **Result: 76.26%, +2.28 over `full_ft_ls`** (with the slower head and twice the epochs; at least about +1.2 at equal epochs) |

Run 22 needs a finished run's folder with `final.pt`: the best one (run 15) is on Ryan's Drive, but any finished 5% run shows whether a larger test size helps. If it does, submit with `python -m common.submit <run> --team XX --size <best size>` and report the latency at that size (it is printed per size).

Code: the new settings (`drop_path_rate`, `ema_decay`, `mixup_alpha`, `cutmix_alpha`, `auto_augment`, `re_prob`, `layer_decay`, and `--sizes` / `--size` in `common/evaluate.py` and `common/submit.py`) are opt-in. Their defaults reproduce the old behaviour exactly (checked on CPU on 21 Oct: the same parameter groups and the same outputs at 224 px), so earlier results and configs are unaffected. They live in `common/`, so the team agrees before the branch is merged.

Not planned: other adapter types (DoRA, SSF, VPT, FacT; expected under 0.5 points), a larger training size (about 3.5 times the cost per step at 384 px), pseudo-labels on val or test (forbidden), another checkpoint (ruled out).

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
- Full fine-tuning (72.53%) beats the 5% design like for like before tuning (71.05% at 10 epochs; 71.29% at 7.78). After tuning (21 Oct), the best 5% run (`tune_head_lr_half`, 74.14%) ties full fine-tuning with label smoothing (73.98%). The story for the report is that 5% of the parameters, tuned, matches full fine-tuning, and what it costs or saves (latency about 11.5 ms against 7.0 ms; 4.3M trained weights to store per task instead of 85.8M). Run 21 (full fine-tuning with a slower head) is still needed before calling it a tie for the report. Update 23 Oct: with layer-wise lr decay and the whole 2.4 h budget, full fine-tuning reaches 76.26% (`full_ft_llrd`), about 2 points above the best 5% run at 224 px, so it is no longer a tie. The story becomes: 5% of the parameters gets within about 2 points of a well-tuned full fine-tuning, with 20 times fewer weights to store per task. Compare both tracks at their best test size and say that the 5% runs used about half the budget.
- Ablation finding: at the same 5% budget, AdaptFormer alone (71.09%), LoRA alone (70.76%) and the combination (71.05%) tie, and AdaptFormer alone is the fastest (9.2 against 11.4 ms). Placing the adapters only in the last 6 blocks is worse (70.17%). All tuning runs used the combined design, so the final 5% model is the combined design with the tuned recipe. Either report it as is, or run AdaptFormer alone with the tuned recipe (label smoothing, lr 1e-4, head x0.5) and present the simpler design if it ties.
- Tuning findings for the report, in order of size: the head's learning rate (2.4 points from x2 to x0.5), the adapters' learning rate (3e-4 to 1e-4 with label smoothing: +0.85; 5e-4 is clearly worse), label smoothing (+1.1 for the 5% design, +1.45 for full fine-tuning). Head init from 43 images per class raised zero-shot but lowered the final accuracy at lr 3e-4.
- What the parts add (hold-out top-1): class-mean head 35.70 (43.30 with 43 images per class) -> trained head only 62.62 (`abl_linear_probe`) -> adapters 73.02 -> tuned head rate 74.14. About 70% of the gain over zero-shot comes from training the head, 30% from the adapters.
- Open question for the professor: which model's predictions go to the leaderboard and to the 2 Phase 2 submissions: the full fine-tuning model, the 5% model, or one of each?
- Open question for the professor: an 80 GB A100 is counted here as 80 GB (budget 1.2 h). The LoRA run peaked at 6.7 GiB, so ask whether such a run may use the full 2 h.
- Zero-shot baseline for the report (0% trained, head set from class means): 43.30% with all 43 training images per class, 35.70% with 10 per class. The brief reports 43.7%.
- Base model: the professor ruled on 12 Oct 2026 that a different base checkpoint is not allowed (the IN21K-only weights were considered and dropped). Whether a larger input size than 224 px is allowed has not been asked yet.
- Hold-out numbers are not leaderboard numbers. Report both and say which is which.
- Leaderboard scores are not published by teams, and this repo is public, so they are not recorded here. Keep them in the report only.
