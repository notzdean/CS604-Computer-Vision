# Experiment log

Running record of every experiment for the ablation study and the report. One section per run, plus a summary table.
Source of truth for numbers is `results/results.csv` (one row per run); this file adds the interpretation.
Brief marks: motivation of design (most important), design, **ablation of each part**, top-1, trainable params, latency, GPU-hours, loss curves.

**Counting rule:** trainable % = adapter params / backbone params (85.8M). The classifier head (7.69M) is excluded and shown separately.
**Hold-out:** 2 labelled train images per class (20,000 images). These are *not* the val set. The CodaBench val leaderboard is the real check.

## Summary

| Run | Method | Config | Hold-out top-1 | Trainable % (adapter) | Head params | GPU-h (A100 80GB) | Leaderboard val top-1 (not published) |
|---|---|---|---|---|---|---|---|
| zero-shot (head init only) | none | `lora_baseline` before training | 35.70 | 0 | 7.69M (init only) | - | - |
| lora_r8_full | LoRA r=8, qkv | `lora_baseline` | 68.78 | 0.344 | 7.69M (trained) | 0.76 | not recorded here |

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

- Trainable % is measured against the backbone and the head is excluded. This is inferred from the brief's 0.34% LoRA figure (rank 8 on qkv matches exactly), and it is not stated in the brief. Confirm with the professor or state it explicitly in the report.
- Hold-out numbers are not leaderboard numbers. Report both and say which is which.
- Leaderboard scores are not published by teams, and this repo is public, so they are not recorded here. Keep them in the report only.
