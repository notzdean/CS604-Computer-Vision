# Ablation story

The ablation section of the presentation and report, told as one line of reasoning: each experiment answered one
question and raised the next. All numbers are hold-out top-1 (20,000 labelled training images kept out of training,
2 per class); differences under about 0.5 points are within noise. Run names point to `results/` and `EXPERIMENT_LOG.md`.

**One sentence for the slide:** we started by asking *where* to put the adapters, discovered the real problem was
*over-confidence*, and found the fix in the part nobody counts: *the classifier head*.

## Chapter 1: "We have 5%. Where should it go?"

The first design put LoRA on the attention (`qkv`) and AdaptFormer on every FFN, 4.998% of the backbone: 71.1%.
Do we need both? Same 5% budget, same settings:

| Design | Run | Hold-out |
|---|---|---|
| LoRA only (rank 116) | `abl_lora_only` | 70.8 |
| AdaptFormer only (bottleneck 232) | `abl_adaptformer_only` | 71.1 |
| Both (LoRA r=12 + AdaptFormer b=208) | `abl_combined` | 71.1 |
| AdaptFormer in the last 6 layers only | `abl_adaptformer_last6` | 70.2 |

**Lesson:** where the parameters sit barely matters, as long as they are spread over all layers. Adding adapters was
not the bottleneck, so we looked for the real one.

## Chapter 2: "Then what is holding us back?"

The training curves: the training loss fell to almost zero (0.07) while the hold-out loss started rising after epoch 2.
With 45 images per class the model memorises the training images and becomes over-confident.

**Lesson:** the problem is over-confidence, not capacity. The next experiments target that.

## Chapter 3: "Calm the model down"

- **Label smoothing** (never be 100% sure): +1.1 (`tune_label_smoothing`).
- **Lower learning rate** (gentler steps): 3e-4, 2e-4, 1e-4, each better (`tune_lr_*`, `tune_ls_lr*`).

The learning rate also drives the classifier head, and the head has 7.7M parameters, more than all the adapters
together (4.3M). So the head got its own speed, with the adapters fixed at 1e-4:

| Head learning rate | Run | Hold-out |
|---|---|---|
| 2e-4 (faster) | `tune_head_lr_double` | 71.8 |
| 1e-4 (same as the adapters) | `tune_ls_lr1e4` | 73.0 |
| 5e-5 (half) | `tune_head_lr_half` | **74.1** |
| 2.5e-5 (quarter) | `tune_head_lr_quarter` | 74.0 |

**Lesson (key finding):** the head memorises first. Training it slowly was the biggest single improvement, about
+3 points over the first design.

## Chapter 4: "Can more anti-memorising tricks push further?"

Four standard tricks on top of the slow head:

| Trick | Run | Hold-out |
|---|---|---|
| Stochastic depth 0.1 | `imp_droppath` | 74.1 |
| Weight averaging (EMA) | `imp_ema` | 74.2 |
| Mixup / CutMix, 15 epochs | `imp_mixup` | 74.0 |
| TrivialAugment + random erasing | `imp_trivialaug` | 74.3 |

**Lesson:** all tie. At 224 px we reached the ceiling; the slow head already removed most of the over-confidence.

## Chapter 5: "What about how we test?"

Random training crops make objects look larger than the 224 px centre crop used at test time. The same model
(`imp_trivialaug`), tested on slightly larger images without retraining: 288 px gives 76.0 (+1.7); 320 px +1.0;
352 px -0.2. Mirror averaging adds +0.25 more, at twice the inference cost.

**Lesson:** a train / test mismatch was costing points. *Open: use the larger test size only if the professor approves.*

## Ending: "How close is 5% to 100%?"

| | Best 5% (`imp_trivialaug`) | Full fine-tuning (`full_ft_llrd`) |
|---|---|---|
| Hold-out top-1 | 74.3 | 76.3 |
| Trained share of the backbone | 5% | 100% |
| Stored per task | 12M parameters | 93.5M parameters |
| Latency (batch 1) | 11.8 ms | 6.9 ms |
| Training | 10 epochs | 18.2 epochs |

Full fine-tuning was given the same care (label smoothing, a learning rate that shrinks towards the early layers). The
5% model reaches about 97% of its accuracy. *Pending: the equal-length (`full_ft_llrd_10ep`) and equal-budget
(`imp_trivialaug_fullbudget`) runs show how much of the 2-point gap is training time rather than method.*

## Reference points

| Model | Trained | Hold-out |
|---|---|---|
| Zero-shot: head from class averages, 10 / 43 images per class | 0% | 35.7 / 43.3 (brief: 43.7) |
| Linear probe: head only (`abl_linear_probe`) | 0% | 62.6 |
| LoRA rank 8 on `qkv` (`lora_r8_full`) | 0.34% | 68.8 (brief: 68.3) |
| Full fine-tuning, first run (`full_ft_baseline`) | 100% | 72.5 (brief: 70.8) |
