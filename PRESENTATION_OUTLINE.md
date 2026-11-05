# Presentation outline (7 Nov 2026, 15 min + 5 min Q&A)

Built against `project_requirements.md`: cover the investigation, the motivation, the design, the comparison with baselines and the ablations; focus on adaptation methods; show the process, including failed runs; visually rich slides. All scores are hold-out top-1 (2 images per class, 20,000 images), at 224 px unless a size is given. Differences under 0.5 points are noise. Leaderboard and test scores are not written here (public repo): add them to the slides only.

**Storyline:** we set out to find the best adaptation method under 5%. The method type mattered less than how the classifier head adapts and how many pixels the model sees, and, contrary to the brief's expectation, tuned full fine-tuning still beat the 5% design on this few-shot task. For the final models we applied one rule: among options within 0.5 points of the best, take the cheapest. That picked 320 px over 352 px for both tracks, and at a 224 px training size AdaptFormer alone over LoRA + AdaptFormer. Training at 288 px then added about 2 points to both tracks, and at 288 px the tie broke: the combined design led AdaptFormer alone by 0.87 (run 49), so we kept it. Both submitted models are the measured hold-out models (runs 39 and 42), trained and tested under the same conditions.

## Slides

| # | Slide | Main visual | Hypotheses | Time |
|---|---|---|---|---|
| 1 | Title | Grid of sample training images | | 0:15 |
| 2 | The problem | Similar-looking pairs from different classes; 10k classes x 45 images; 96 GB-h budget | | 0:45 |
| 3 | Baselines and how we measure | Brief's reference rows next to ours; hold-out diagram; iteration loop (one hypothesis per run) | | 0:45 |
| 4 | Design considerations | 2 x 3 icon grid | | 1:00 |
| 5 | The adaptation methods | ViT block diagram: where each method attaches | | 1:00 |
| 6 | Iteration 1: equal 5% budget | `results/hypothesis_charts/h1_h4_design.png` + 12-block strip | H1, H2 + H3, H4 | 1:15 |
| 7 | Params vs accuracy | Scatter: trainable % (log) vs top-1 | H1 | 0:45 |
| 8 | Iteration 2: diagnosing the limit | Loss curves (train vs hold-out) + `h5_h8_overconfidence.png` | H5, H6 + H8, H7 | 1:00 |
| 9 | Iteration 3: the head | Head-lr curve; 70/30 donut; failed head starts (`h9_h11_head.png`) | H9, H10 (H11 as finding) | 1:15 |
| 10 | Iteration 4: refining the recipe | `h12_h14_recipe.png` | H12, H14 | 0:45 |
| 11 | How we selected the models | Selection funnel (see "Model selection process"); the "cheapest within 0.5 points" rule | H2 + H3 | 1:15 |
| 12 | Iteration 5: more pixels | Test-size curve 224 to 416; crop comparison; `h17_h19_pixels.png` | H17, H19 | 1:00 |
| 13 | Full fine-tuning track | Framework diagram + progression bars | H13 + H15 | 0:45 |
| 14 | Comparing the tracks | Paired bars: accuracy, stored params, latency, GPU-hours; `h15_h16_tracks.png` | H16, H20 | 1:00 |
| 15 | Lessons and novelty | 4 icons; test score; hypothesis scoreboard (6 supported / 7 rejected / 2 split) | | 0:45 |

Backup: full results table; parameter-count rule; latency, GPU-hours and VRAM per final model; a training log; why our zero-shot is lower than the brief's.

Speaker split (suggestion): person 1 slides 1-6, person 2 slides 7-11, person 3 slides 12-15.

## Hypotheses (numbers as in `EXPERIMENT_LOG.md`)

Show the hypothesis as the title or subtitle of each iteration slide, and the verdict as a stamp (supported / rejected / split). The wording is reconstructed from the config notes, not pre-registered: say so if asked. Of the 20 in the log, 18 are tested hypotheses, shown in 15 rows (three pairs merged), H11 is shown as a finding and H18 goes to backup. Scoreboard for slide 15: 6 supported, 6 rejected, 3 split (H2 + H3: a tie at 224 px, the combined design ahead when trained at 288 px; H13 + H15: helps one track, hurts the other; H20: lifts the 5% design, but the gap stays).

| # | Hypothesis | Verdict | Slide |
|---|---|---|---|
| H1 | More adapter parameters beat the brief's small LoRA | Supported, +2 to +2.3; **confounded**: the small LoRA also trained 7 epochs against 10 (show the caveat) | 6, 7 |
| H2 + H3 | Adapting attention and FFN beats either alone at equal 5% | Split: a tie untuned (all within 0.33) and on the tuned recipe at 224 px (AdaptFormer alone -0.50; -0.37 tested at 288 px), but trained at 288 px the combined design leads by 0.87 (run 49). It pays only at the final resolution | 6 |
| H4 | Late blocks carry the adaptation, so spend the 5% on the last 6 | Rejected: -0.92 | 6 |
| H5 | Label smoothing helps, because the model is over-confident on 45 images | Supported: +1.12 (5%), +1.45 (full FT) | 8 |
| H6 + H8 | A lower lr helps, a higher one hurts (one sweep: 5e-4 to 5e-5) | Supported down to about 1e-4: 69.28 at 5e-4, 71.05 at 3e-4, 72.37 at 2e-4, 73.02 at 1e-4 (with label smoothing); 5e-5 level | 8 |
| H7 | Label smoothing and a lower lr add up | Rejected: no more than either alone, so they fix the same problem | 8 |
| H9 | A better head start gives a better final model | Rejected three times: -1.07, -1.03, -4.00, though zero-shot rises | 9 |
| H10 | The head should learn more slowly than the adapters | Supported: +1.12 at 0.5x; the most sensitive setting | 9 |
| H12 | A stronger regulariser adds accuracy on top of the slow head | Rejected: four ties | 10 |
| H14 | Using the whole budget improves on 10 epochs | Rejected: -0.10 (5%), +0.20 (full FT) | 10 |
| H13 + H15 | Layer-wise lr decay helps both tracks | Split: helps full FT modestly (+0.51), hurts the 5% design (-1.50). A technique does not carry over between tracks | 13 |
| H16 | Tuned, the 5% design matches full fine-tuning | Rejected: gap of 1.7 to 2.1 at equal epochs and equal budget | 14 |
| H17 | Testing above the training size helps (FixRes) | Supported: +1.61 to +1.71, no retraining | 12 |
| H19 | Training at a higher resolution beats only testing at it | Supported: +2.02, the largest single gain | 12 |
| H20 | Training at 288 px lifts the 5% design too, and so changes the track gap | Split: +1.95 for the 5% design (77.94 at 288 px), as much as full FT, so the gap stays at 1.75 | 14 |

Not shown as hypotheses:

- **H11 (finding, slide 9):** training the head alone gives about 70% of the gain over zero-shot, the adapters the other 30% (donut). It is a measurement that could not have been rejected, so no verdict stamp.
- **H18 (backup):** averaging two full fine-tuning models gives +0.39 for twice the inference cost. Not an adaptation method; it stays in the "rejected" bullet of slide 11.

Open (added 5 Nov; add a row above with its verdict when the result is in, and update the scoreboard):

| # | Hypothesis | Run | Compared against | Slide |
|---|---|---|---|---|
| H1 re-test | At equal epochs and lr, rank 116 beats rank 8 (removes the H1 caveat) | 50 `abl_lora_r8` | `abl_lora_only` 70.76 | 6, 7 |
| H21 | An early-fitting head starves the adapters, so delaying the head helps (novel; the reverse of LP-FT) | 52 `imp_head_delay` | `imp_trivialaug` 74.28 | 9 |
| H22 | Smoothing towards similar classes beats uniform smoothing (novel) | 53 `imp_ls_neighbours` | `imp_trivialaug` 74.28 | 8 |
| H23 | The biases alone (BitFit, 0.12%) recover most of the adapters' gain over the linear probe | 51 `imp_bitfit` | `abl_linear_probe` 62.62, `imp_trivialaug` 74.28 | 7 |
| H24 | Another seed moves the result by less than the 0.5-point noise bar | 47, 48 (seed 1 of runs 42, 39) | runs 42 and 39 | 14 |

## Design considerations (slide 4, and one caption per design slide)

| Consideration | What we weighed | How it shaped the design |
|---|---|---|
| Parameter budget | <5% of the backbone, head not counted | Spend the whole 5%: LoRA r=12 + AdaptFormer b=208 = 4.998% (4.992% at 288 px, where the backbone's position embeddings grow). AdaptFormer alone, b=232, ties it at 224 px training but is 0.87 lower at 288 px training (run 49) |
| Compute budget | 96 GB-h; A100 about 1,100 img/s in bf16 | Fixed 10-epoch runs keep comparisons fair; heavy augmentation makes the data loader the limit |
| Few-shot overfitting | 45 images per class, 10k classes | Choices aimed at overconfidence: label smoothing, lower lr, slower head |
| Inference latency | Added modules cost time on every forward pass | LoRA can be merged into the weights; AdaptFormer cannot. The combined model cannot merge its LoRA either, because AdaptFormer stays: 11.8 ms, against 9.5 ms for AdaptFormer alone and about 7 ms for full FT (batch 1; the same at 288 to 352 px, run 46). AdaptFormer alone would be the faster final model (9.8 against 12.6 ms logged at 288 px), but at 288 px training it is 0.87 lower, outside the 0.5-point tie rule |
| Storage per task | The point of PEFT | 12M against 93.5M params stored per task, head included |
| Keep the pre-trained features at step 0 | Avoid damaging them before training starts | Zero-initialised `lora_B` and `up`; head from class means, calibrated |

Per slide:

- **5 Methods:** LoRA changes what tokens attend to; AdaptFormer changes what the features are. We expected them to be complementary, so we tested each alone and combined at equal budget. The linear probe is the "new head only" contrast.
- **6 Equal budget:** later blocks are more task-specific, so putting twice the width in the last 6 blocks seemed plausible. Rejected: early blocks need adapting too.
- **7 Params vs accuracy:** pick the design at the knee of the curve; going from 0.34% to 5% bought about +2 points.
- **8 Diagnosis:** the brief hints at loss design; overconfident memorisation pointed to label smoothing.
- **9 Head:** the head has 7.7M params against the adapters' 4.3M, so it gets its own lr. A stronger head start raises zero-shot accuracy but leaves less for the adapters to learn, so we chose the gentle start.
- **10 Recipe:** layer-wise lr decay protects pre-trained layers in full fine-tuning; with a frozen backbone it only slows the early adapters. Kept only regularisers that cost nothing at inference.
- **12 Pixels:** training at 288 px costs about 1.65x per epoch, affordable because 10 epochs already fit. Position embeddings are interpolated, so no new parameters.
- **13 Full fine-tuning:** protect the pre-trained features with layer-wise decay 0.75 and a slow head, rather than freezing anything.
- **11 Selection:** a tie goes to the cheaper option, and a measured result beats an assumed one. 320 px ties 352 px on both tracks (-0.31 full FT, +0.02 5%) with about 17% fewer patches (400 against 484): less compute per image, though the same batch-1 latency on an A100 (run 46). AdaptFormer alone tied the combined design at a 224 px training size and is about 20% faster, but trained at 288 px it is 0.87 lower (77.07 against 77.94, run 49), so the tie rule no longer picks it.
- **14 Comparison:** PEFT saves storage; it does not save training time (the frozen backbone still runs forward and backward) and costs latency unless merged.

## Results explained

Each finding: what we saw, our explanation, how sure we are.

### 1. Adapter type does not matter at 5%; coverage and size do (slide 6; H1, H2 + H3, H4)

| Design (5%, 10 epochs, lr 3e-4) | Top-1 |
|---|---|
| LoRA only | 70.76 |
| AdaptFormer only | 71.09 |
| LoRA + AdaptFormer | 71.05 |
| AdaptFormer, last 6 blocks only | 70.17 |
| LoRA r=8 (0.34%, 7 epochs) | 68.78 |

- **Explanation:** with 5% of the parameters, any of these designs has enough capacity to fit 45 images per class; the limit is generalisation, not capacity. Concentrating on the last blocks loses about a point, which suggests the early layers also need to shift for fine-grained detail.
- **Confidence:** the tie is solid (all within 0.33). Re-tested on the tuned recipe, AdaptFormer alone was 0.50 lower at 224 px (73.78 against 74.28) and 0.37 lower at 288 px (75.62 against 75.99): still a tie, and AdaptFormer alone is about 17% faster (9.7 against 11.7 ms at 288 px). **But trained at 288 px, the final resolution, it is 0.87 lower (77.07 against 77.94, run 49)**: with more pixels, adapting attention as well starts to pay. One seed each, so the seed repeats (H24) show how firm this is.
- **The small-LoRA comparison is confounded:** it also trained for fewer epochs.

### 2. The limit was memorisation (slide 8; H5, H6 + H8, H7)

- **Seen:** at the end of training, train loss 0.07 against hold-out loss 1.42.
- **Label smoothing 0.1:** 72.17 (+1.12). **Lower lr 2e-4:** 72.37 (+1.32). **Higher lr 5e-4:** 69.28. **Both together:** 72.35, no more than either.
- **Explanation:** both changes stop the model from becoming over-confident on its 45 images, so they fix the same problem and do not add up. With label smoothing, lowering the lr further to 1e-4 reached 73.02; 5e-5 was level (73.52, at the noise bar).

### 3. The head is the key adaptation lever (slide 9; H9, H10, finding H11)

| Head lr (adapters at 1e-4) | Top-1 | Hold-out loss |
|---|---|---|
| 2x | 71.75 | 1.570 |
| 1x | 73.02 | 1.420 |
| 0.5x | **74.14** | 1.325 |
| 0.25x | 74.02 | 1.335 |

- **Better head starts all ended lower:** class means from 43 images (-1.07, and -1.03 on the tuned recipe); a head already trained by a linear probe (-4.00).
- **Explanation (ours, not tested directly):** a head that fits the training images early takes the easy part of the loss, so the adapters get a weaker learning signal, and the head then overfits. In the worst case, train loss ended lower (1.90 against 2.04) while hold-out loss ended higher (1.54 against 1.31).
- **Head vs adapters:** training the head alone takes 35.7 to 62.62, about 70% of the gain to 73.02; the adapters give the other 30%.
- **Class means of the adapted features (run 45, no training):** 69.82% at 288 px with no trained head, against 43.30% on the frozen features (224 px, so the gap of about 25 points is approximate). Most of the gain is already in the features; the trained head adds about 6 points on top. Combining the head with the class-mean classifier is a tie (-0.15, cross-fitted): the head no longer overfits in a way a parameter-free classifier can fix.
- **Zero-shot against the brief's 43.7:** our 35.7 uses class means from 10 images per class (the hold-out images are excluded); from 43 images it is 43.30, close to the brief's figure.

### 4. The recipe converged (slide 10; H12, H14; H13 on slide 13)

- Four regularisers on top of the 74.14 model all tie: stochastic depth 74.09, EMA 74.19, Mixup/CutMix (15 epochs) 73.97, TrivialAugment + random erasing 74.28.
- The whole budget instead of 10 epochs: -0.10 on the 5% track, +0.20 on full fine-tuning.
- Layer-wise lr decay on the 5% design: 72.78 (-1.50). It helps full fine-tuning (+0.51), which shows that a technique does not carry over between tracks automatically.
- **Explanation:** once the head is slow and the loss is smoothed, the model no longer overfits much, so extra regularisation and extra epochs have little left to fix.

### 5. More pixels gave the largest gain (slide 12; H17, H19)

| Model | 224 | 288 | 320 | 352 | 384 | 416 |
|---|---|---|---|---|---|---|
| 5% (`imp_trivialaug`, trained at 224) | 74.28 | **75.99** | 75.28 | 74.06 | | |
| 5% (`imp_trivialaug_288`, trained at 288) | | 77.94 | **78.75** | 78.73 | | |
| 5% (`imp_trivialaug_fixres288`, 224 + 2 epochs at 288) | | 77.28 | **77.73** | | | |
| Full FT (`full_ft_llrd_288`, trained at 288) | | 79.69 | 80.14 | **80.45** | 79.91 | 79.06 |

- **Testing larger:** random training crops make objects look larger than in a centred test crop, so testing at a larger size matches the object scale the model learned: +1.71 (5%), +1.61 (full FT) at 288 px with no retraining.
- **Training larger:** +2.02 (full FT) and +1.95 (5%) over only testing at 288 px. Fine-grained classes differ in small details that 224 px loses. Both models trained at 288 px peak at 320 to 352 px, and 320 px is the smallest size within 0.5 points of the best for both.
- **Cheaper route (run 44):** fine-tuning the finished 224 px model for 2 epochs at 288 px (FixRes) gives 77.28 (77.73 at 320 px): +1.33 for 0.34 h, about two thirds of the gain of training at 288 px from scratch (77.94; 78.75 at 320 px). Run 42 costs about 11% more compute in total (1.70 against 1.53 h) for +0.66 at 288 px and +1.02 at 320 px, above the noise bar, so it stays the final recipe.
- **Caveat:** 352 px was chosen on the same hold-out it was scored on, so 80.45 is slightly optimistic.
- **Chosen test sizes:** both final models are trained at 288 px and tested at 320 px, the smallest size within 0.5 points of the best: full FT 80.14 (352 px: 80.45, +0.31), 5% 78.75 (352 px: 78.73). 352 px needs about 21% more patches (compute per image) for no real gain; batch-1 latency on the A100 is the same at both sizes (7.1 to 7.3 ms full FT, 11.7 to 11.8 ms 5%, run 46), so the saving shows up in throughput and on smaller GPUs, not in batch-1 latency.

### 6. Tracks: full fine-tuning wins, against the brief's expectation (slide 14; H16, H20)

| Tuned | 5% | Full FT |
|---|---|---|
| 10 epochs, 224 px test | 74.28 | 76.06 |
| 10 epochs, 288 px test | 75.99 | 77.67 |
| Whole budget, 288 px test | 75.99 | 77.89 |
| Trained at 288 px, 288 px test | 77.94 | 79.69 |
| Trained at 288 px, 320 px test (chosen size) | 78.75 | 80.14 |
| Trained at 288 px, best test size | 78.75 (320 px) | 80.45 (352 px) |
| Params stored per task, incl. head | 12M | 93.5M |
| Submitted model, hold-out score at 320 px | 78.75 (run 42) | 80.14 (run 39) |
| Latency, batch 1, 320 px (submitted size; median of 5 x 100 passes) | 11.8 ms | 7.3 ms |
| Training, 48 GB-equivalent | 1.19 h | 1.67 h (288 px) |

- **The brief expected** PEFT to win with small data. Possible reasons it did not, here: full fine-tuning was regularised just as carefully (label smoothing, slow head, layer-wise decay), so it did not overfit as feared; 450k images is not small overall; and the backbone was already close to the task.
- **Fair comparison:** the gap of about 2 points holds at equal epochs, at equal budget and at equal training size: trained and tested at 288 px it is 1.75 (79.69 against 77.94, H20), and 1.39 at the chosen test size of 320 px (80.14 against 78.75). Training at 288 px lifts both tracks by about 2 points, so resolution and method are separate effects.

## Model selection process (slide 11)

### How we chose

- **One yardstick:** a fixed hold-out of 2 images per class (20,000 images, seed 0), the same for every run. Val and test labels are withheld, and the leaderboard allows 5 submissions a day, so it was only a sanity check, not a selection tool.
- **One change per run:** each run differs from its baseline in one setting, so a difference can be attributed.
- **Noise bar 0.5 points:** a smaller difference is a tie. **Tie-break:** take the cheapest option within 0.5 points of the best (design, test size). A regulariser that costs nothing at inference could be kept at the nominal best.
- **Fixed length:** 10 epochs, so runs compare at equal compute; budget-filling runs were tested separately.

### The funnel, 5% track

| Stage | Question | Picked | Top-1 |
|---|---|---|---|
| 1. Design | Which adapter at equal 5%? | Combined (tie; nominal best, became the base) | 71.05 |
| 2. Loss and lr | What fixes overconfidence? | Label smoothing 0.1, lr 1e-4 | 73.02 |
| 3. Head | How should the head learn? | Head lr x0.5, class-mean start from 10 images | 74.14 |
| 4. Regulariser | Does anything add more? | TrivialAugment + random erasing (tie, free at inference) | 74.28 |
| 5. Test size | Which size to score at? | 288 px | 75.99 |
| 6. Design re-check | Does the tuned recipe separate the designs? | AdaptFormer alone (tie: -0.37 at 288 px; about 17% faster) | 75.62 at 288 px |
| 6b. Training size | Does training at 288 px help the 5% design? | Yes: +1.95 (combined design, run 42); AdaptFormer alone gains less (77.07, run 49) | 77.94 at 288 px |
| 6c. Test size again | Which size for a model trained at 288 px? | 320 px (352 is level) | 78.75 |
| 7. Submitted model | Which model goes to the test set? | `imp_trivialaug_288` itself (run 42), at 320 px: its hold-out score is measured | 78.75 |

### The funnel, full fine-tuning track

| Stage | Picked | Top-1 |
|---|---|---|
| Baseline | `full_ft_baseline` | 72.53 |
| + label smoothing | `full_ft_ls` | 73.98 |
| + slow head | `full_ft_ls_headlow` | 75.75 |
| + layer-wise decay 0.75 (10 epochs) | `full_ft_llrd_10ep` | 76.06 |
| Train at 288 px | `full_ft_llrd_288` (79.69 at 288 px) | 80.45 at 352 px |
| Test size | 320 px: ties 352 (-0.31) with about 17% fewer patches | 80.14 |
| Submitted model | `full_ft_llrd_288` itself (run 39), at 320 px: the same training size, epochs, test size and hold-out as the 5% model | 80.14 |

### What we rejected and why

- **Ensemble of two full fine-tuning models:** +0.39 for twice the inference cost.
- **Using the whole budget:** no gain over 10 epochs on either track.
- **Stronger head starts and layer-wise decay on adapters:** both lowered accuracy.
- **AdaptFormer alone as the submitted 5% model:** it ties the combined design at 224 px training and is about 20% faster, but trained at 288 px it is 0.87 lower (run 49), outside the 0.5-point rule, so the combined design (run 42) was kept.
- **352 px test size for full FT:** +0.31 over 320 px, within noise, for about 21% more patches (same batch-1 latency on an A100).
- **Short fine-tune at 288 px instead of training at 288 px (run 44):** 0.66 lower at 288 px and 1.02 lower at 320 px, for about 11% less compute; useful only when a 224 px model already exists.
- **Head + class-mean ensemble at test time (run 45):** -0.15, a tie; an extra 768 x 10,000 product for nothing.

### Limits of the process

- The test size and some settings were chosen on the same hold-out they are reported on, so the best numbers are slightly optimistic.
- The submitted models are the hold-out models (runs 39 and 42), trained on 43 of the 45 images per class. Their test scores can be checked against measured hold-out scores, at the cost of 4.7% of the training data. The test set is the independent check on the 320 px test size, which was chosen on the hold-out.
- Settings never changed alone: weight decay, the adapters' scale 0.1, LoRA alpha, the r=12 / b=208 split, and AdaptFormer's bottleneck size at a fixed 5% (232 is simply the largest that fits).
