# Ablation story

The ablation section of the presentation and report, told as one line of reasoning: each experiment answered one
question and raised the next. All numbers are hold-out top-1 (20,000 labelled training images kept out of training,
2 per class); differences under about 0.5 points are within noise. Run names point to `results/` and
`EXPERIMENT_LOG.md`; the "Who" column says which team member ran each experiment.

**One sentence for the slide:** we started by asking *where* to put the adapters, discovered the real problem was
*over-confidence*, and found the fix in the part nobody counts: *the classifier head*.

## Chapter 1: "We have 5%. Where should it go?"

The first design put LoRA on the attention (`qkv`) and AdaptFormer on every FFN, 4.998% of the backbone: 71.1%.
Do we need both? Same 5% budget, same settings:

| Design | Run | Who | Hold-out |
|---|---|---|---|
| LoRA only (rank 116) | `abl_lora_only` | Ryan | 70.8 |
| AdaptFormer only (bottleneck 232) | `abl_adaptformer_only` | Lilian | 71.1 |
| Both (LoRA r=12 + AdaptFormer b=208) | `abl_combined` | Nicole | 71.1 |
| AdaptFormer in the last 6 layers only | `abl_adaptformer_last6` | Nicole | 70.2 |

- The three equal-budget designs tie (spread 0.3 points). Ryan's LoRA-only run was slightly behind at every epoch
  but closed most of the gap by the end.
- All three are 2 to 2.3 points above LoRA rank 8 (68.8, 0.34%). So the gain of the 5% design comes from the
  *number* of adapter parameters, not from where they sit.

**Lesson:** where the parameters sit barely matters, as long as they are spread over all layers. Adding adapters was
not the bottleneck, so we looked for the real one.

## Chapter 2: "Then what is holding us back?"

The training curves: the training loss fell to almost zero (0.07) while the hold-out loss started rising after epoch 2.
With 45 images per class the model memorises the training images and becomes over-confident. Ryan's learning-rate
runs make it visible: at 5e-4 the hold-out loss climbs from 1.57 to 1.88 (the clearest over-confidence of any run),
at 2e-4 it keeps falling.

**Lesson:** the problem is over-confidence, not capacity. The next experiments target that.

## Chapter 3: "Calm the model down"

One change at a time on `abl_combined` (71.1):

| Change | Run | Who | Hold-out | vs baseline |
|---|---|---|---|---|
| Label smoothing 0.1 (never be 100% sure) | `tune_label_smoothing` | Ryan | 72.2 | +1.1 |
| Lower learning rate, 2e-4 | `tune_lr_low` | Ryan | 72.4 | +1.3 |
| Higher learning rate, 5e-4 | `tune_lr_high` | Ryan | 69.3 | -1.8 |
| Better head start (head built from all 43 images per class) | `tune_head_init` | Lilian | 70.0 | -1.1 |

- Label smoothing and a lower rate fix the *same* problem: together at 2e-4 they give 72.4, no more than either alone
  (`tune_ls_lr2e4`). Lowering the rate further keeps helping: 73.0 at 1e-4, 73.5 at 5e-5 (`tune_ls_lr*`, Nicole).
- Lilian's head-start run is the first clue that the head is the weak point. The better start (zero-shot 43.3% against
  35.7%) leads for two epochs, then falls behind and its hold-out loss rises: a head built from all the training
  images already fits them, so it memorises even sooner.

The learning rate also drives the classifier head, and the head has 7.7M parameters, more than all the adapters
together (4.3M). So the head got its own speed, with the adapters fixed at 1e-4:

| Head learning rate | Run | Who | Hold-out |
|---|---|---|---|
| 2e-4 (faster) | `tune_head_lr_double` | Ryan | 71.8 |
| 1e-4 (same as the adapters) | `tune_ls_lr1e4` | Nicole | 73.0 |
| 5e-5 (half) | `tune_head_lr_half` | Ryan | **74.1** |
| 2.5e-5 (quarter) | `tune_head_lr_quarter` | Nicole | 74.0 |

- The head's learning rate is the most sensitive setting found: 2.4 points from x2 to x0.5.
- The head and the adapters want *different* speeds: both at 5e-5 give 73.5, head 5e-5 with adapters 1e-4 give 74.1.
- A slow head starts worse but catches up by epoch 3 to 4: it keeps its well-placed start (the class averages) while
  the adapters reshape the features underneath it.

How much does each part contribute? Lilian's linear probe (head only, backbone frozen, `abl_linear_probe`) reaches
62.6%. From zero-shot (35.7) to the best 5% run (74.1), about **70% of the gain comes from training the head** and
30% from the adapters. The linear probe underfits (its training loss stays high): the frozen features cannot separate
10,000 classes on their own, and the adapters fix that mostly in the first epochs.

**Lesson (key finding):** the head is the largest trained part, it carries most of the gain, and it memorises first.
Training it slowly was the biggest single improvement, about +3 points over the first design.

## Chapter 4: "Can more anti-memorising tricks push further?"

Lilian added four standard tricks on top of the slow head (`tune_head_lr_half`, 74.1):

| Trick | Run | Hold-out | What else it showed |
|---|---|---|---|
| Stochastic depth 0.1 | `imp_droppath` | 74.1 | better calibrated (hold-out loss 1.25 against 1.33), but no accuracy gain |
| Weight averaging (EMA) | `imp_ema` | 74.2 | up to +1.2 in the middle of training, gone by the end (the cosine schedule already smooths the weights) |
| Mixup / CutMix, 15 epochs | `imp_mixup` | 74.0 | 50% more training time for the same result; blending two images can hide the small details that separate fine-grained classes |
| TrivialAugment + random erasing | `imp_trivialaug` | 74.3 | ahead only from epoch 8, 12% slower training |

**Lesson:** all tie. At 224 px we reached the ceiling; the slow head already removed most of the over-confidence.

## Chapter 5: "What about how we test?"

Random training crops make objects look larger than the 224 px centre crop used at test time. Lilian scored the same
model (`imp_trivialaug`) on slightly larger images, without retraining: 288 px gives 76.0 (+1.7); 320 px +1.0;
352 px -0.2. Every metric improves together (top-5 +1.0, lower hold-out loss), so it is not a fluke. Mirror averaging
adds +0.25 more, at twice the inference cost. At batch 1 the latency barely moves (11.4 to 11.7 ms), but each image
costs about 1.7 times the compute (324 patches instead of 196).

**Lesson:** a train / test mismatch was costing points. *Open: use the larger test size only if the professor approves.*

## Ending: "How close is 5% to 100%?"

Lilian gave full fine-tuning the same care (label smoothing, a slower head, and a learning rate that shrinks towards
the early layers, `full_ft_llrd`):

| | Best 5% (`imp_trivialaug`) | Full fine-tuning (`full_ft_llrd`) |
|---|---|---|
| Hold-out top-1 | 74.3 | 76.3 |
| Trained share of the backbone | 5% | 100% |
| Stored per task | 12M parameters | 93.5M parameters |
| Latency (batch 1) | 11.8 ms | 6.9 ms |
| Training | 10 epochs, about 55% of the budget | 18.2 epochs, 95% of the budget |

- Keeping the early layers close to the pre-trained weights lets full fine-tuning learn much faster at the start
  (+3.6 points after epoch 1 against `full_ft_ls`), and it is already 1.2 points ahead at epoch 9.
- Over-confidence comes back late even here: its hold-out loss is lowest at epoch 5 and then rises while accuracy
  still climbs.
- The 5% model reaches about 97% of its accuracy while storing far fewer weights per task, but it is slower at
  inference because the adapters add layers.

*Pending:* the equal-length (`full_ft_llrd_10ep`) and equal-budget (`imp_trivialaug_fullbudget`) runs show how much of
the 2-point gap is training time rather than method; the test-size check on full fine-tuning compares both tracks at
their best test size; `full_ft_ls_headlow` would separate the slower head from the layer-wise rate.

## Reference points

| Model | Trained | Hold-out |
|---|---|---|
| Zero-shot: head from class averages, 10 / 43 images per class | 0% | 35.7 / 43.3 (brief: 43.7) |
| Linear probe: head only (`abl_linear_probe`, Lilian) | 0% | 62.6 |
| LoRA rank 8 on `qkv` (`lora_r8_full`, Nicole) | 0.34% | 68.8 (brief: 68.3) |
| Full fine-tuning, first run (`full_ft_baseline`, Nicole) | 100% | 72.5 (brief: 70.8) |
| Full fine-tuning + label smoothing (`full_ft_ls`, Nicole) | 100% | 74.0 |

## Who did what

- **Nicole:** pipeline and baselines, the first LoRA + AdaptFormer design and its 10-epoch run, the later-layers
  ablation, the learning-rate sweep with label smoothing, the quarter-speed head, full fine-tuning with label smoothing.
- **Ryan:** LoRA-only ablation, label smoothing, the learning-rate pair (2e-4 / 5e-4), the half- and double-speed head
  (the best 5% recipe); found and fixed the head-init memory leak on 40 GB GPUs.
- **Lilian:** AdaptFormer-only ablation, the better head start, the linear probe, the four regulariser runs, the
  test-size study, full fine-tuning with layer-wise learning rates (the best run), the final 5% model on all
  45 images; the code for the regularisers, the test-size option and the head-init memory fix.
