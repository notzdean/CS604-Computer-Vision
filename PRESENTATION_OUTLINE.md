# Presentation outline (7 Nov 2026, 15 min + 5 min Q&A)

Built against `project_requirements.md`: cover the investigation, the motivation, the design, the comparison with baselines and the ablations; focus on adaptation methods; show the process, including failed runs; visually rich slides. All scores are hold-out top-1 (2 images per class, 20,000 images), at 224 px unless a size is given. Differences under 0.5 points are noise. Leaderboard and test scores are not written here (public repo): add them to the slides only.

**Storyline:** we set out to find the best adaptation method under 5%. The method type mattered less than how the classifier head adapts and how many pixels the model sees, and, contrary to the brief's expectation, tuned full fine-tuning still beat the 5% design on this few-shot task. For the final models we applied one rule: among options within 0.5 points of the best, take the cheapest. That picked 320 px over 352 px for both tracks, and at a 224 px training size AdaptFormer alone over LoRA + AdaptFormer. Training at 288 px then added about 2 points to both tracks, and at 288 px the tie broke: the combined design led AdaptFormer alone by 0.87 (run 49), so we kept it. The best hold-out models are runs 39 and 42. The rules allow only models submitted in Phase 1 to go to the test set, so the two Phase 2 test submissions (5 Nov 2026) were `final_5pct_trivialaug` (run 30: the `imp_trivialaug` recipe on all 45 images per class, trained at 224 px, tested at 288 px) and `full_ft_llrd_288` (run 39, trained at 288 px, tested at 320 px); their test scores go on the slides, not in this repo.

## Run of show (15 minutes)

Written for the class: they did the same project, so the task takes 30 seconds and the time goes to findings. Each slide's title is its finding, as a sentence. Each slide has one main visual; the talk track is what to say, not what to put on the slide. The hypothesis verdict goes on the slide as a stamp (supported / rejected / split).

| # | Slide title (the finding) | Main visual | Speaker | Time | Clock |
|---|---|---|---|---|---|
| 1 | 10,000 classes, 45 photos each: what mattered | Sample image grid; the best hold-out scores and the two test scores | 1 | 0:30 | 0:30 |
| 2 | Same task, our rules: one change per run, 0.5 points is noise | Hold-out diagram (2 per class); iteration loop; seed check | 1 | 1:00 | 1:30 |
| 3 | We tested the brief's adaptation directions under one recipe | ViT block diagram: LoRA on qkv, AdaptFormer beside the FFN, BitFit on biases, head | 1 | 1:00 | 2:30 |
| 4 | At 5%, where the adapters go matters; which adapter does not | `h1_h4_design.png` + 12-block strip | 1 | 1:15 | 3:45 |
| 5 | Most of the adaptation is cheap | `params_vs_accuracy.png` | 1 | 1:00 | 4:45 |
| 6 | The real limit was memorisation, not capacity | Loss curves: train 0.07 vs hold-out 1.42; `h5_h8_overconfidence.png` | 2 | 1:15 | 6:00 |
| 7 | The head is the most sensitive part of the adaptation | Head-lr curve (2x to 0.25x); 70/30 bar | 2 | 1:00 | 7:00 |
| 8 | A better head start ends lower, and our first explanation was wrong | Head-start dumbbells; head-delay per-epoch curve | 2 | 1:15 | 8:15 |
| 9 | After that, nothing else moved the 5% recipe | `h12_h14_recipe.png` (all inside the noise band) | 2 | 0:45 | 9:00 |
| 10 | More pixels gave the largest gain, on both tracks | Test-size curve 224 to 416; train-at-288 bars | 3 | 1:15 | 10:15 |
| 11 | At the final resolution, the design tie broke | Three-row dumbbell: AdaptFormer alone vs combined at 224 / 288 test / 288 train | 3 | 0:45 | 11:00 |
| 12 | Our two models, and how we picked them | Two framework diagrams; selection funnel; test scores | 3 | 1:30 | 12:30 |
| 13 | Full fine-tuning still won, against the brief's expectation | Paired bars: accuracy, stored params, latency, GPU-hours | 3 | 1:15 | 13:45 |
| 14 | What we learned, and what we would do differently | 4 lessons; scoreboard 8 / 8 / 3 | 3 | 1:00 | 14:45 |

15 seconds of slack. Rehearse with a timer: slides 6 to 8 are the ones that run long.

### Talk track

**1. Title (0:30).** "Same task as everyone here: ViT-B/16, 10,000 classes, 45 photos each, 2 hours on 48 GB. Our best models score 80.1% (full fine-tuning) and 78.8% (5%) on our hold-out; our two test submissions are on the last results slide. The interesting part is which changes produced that, and which ones didn't."

**2. How we worked (1:00).** "Val and test have no labels, so we held out 2 photos per class from training, 20,000 images, and scored every run on them. Each run changes one setting against a named baseline, and we treat anything under 0.5 points as a tie. We checked that bar at the end: retraining with another seed moved the score by only 0.09 and 0.25 points, so 0.5 is on the safe side." Visual: hold-out diagram, loop "observe, hypothesise, one-change run, verdict", about 50 runs.

**3. The methods (1:00).** "The brief lists six directions. We built LoRA on attention, AdaptFormer on the feed-forward layers, both together, BitFit (only the biases), and the linear probe (only the head). The head is 7.7M parameters and, by the professor's rule, not counted in the 5%. Everything trains with the same recipe, so differences come from the method." Say plainly that Adapter, prefix tuning and selected-layer tuning were not run.

**4. Design at 5% (1:15).** H1, H2, H4. "Spending the full 5% beats the brief's 0.34% LoRA by 1.04 points at equal settings. Our first comparison said 2 points, but the small run had also trained less: re-running it fairly halved the gap. LoRA alone, AdaptFormer alone and both together tie within 0.33. Putting the whole budget in the last 6 blocks loses 0.92, so the early blocks need adapting too. Conclusion at this point: the design was not the bottleneck."

**5. Most adaptation is cheap (1:00).** H23. "Head only: 62.6. Add only the biases, 0.12% of the backbone: 71.3. That is 74% of what our 5% adapters add, with 40 times fewer parameters and no extra latency. The full 5% buys the last 3 points." Point at the curve: steep, then flat.

**6. Memorisation (1:15).** H5 to H8. "By the end of training, the untuned 5% model's training loss was 0.07 while its hold-out loss stayed around 1.42: it had memorised its 45 photos. Two fixes for over-confidence each helped: label smoothing +1.1, a lower learning rate +1.3. Together they gave no more than either alone, which tells us they fix the same problem. With smoothing, lr 1e-4 reached 73.0. Our second idea, smoothing only towards each class's 20 look-alikes, scored 0.61 lower: better calibrated, less often right. Spreading the doubt evenly works better."

**7. The head (1:00).** H10, H11. "The head has more trained parameters than the adapters, so we gave it its own learning rate. Half the adapters' rate: +1.1. Twice: -1.3. That 2.4-point swing is the most sensitive setting we found. The head alone gives 70% of the gain over zero-shot; the adapters give the other 30%."

**8. Head start (1:15).** H9, H21. "We expected a better starting head to help. Three times it ended 1 to 4 points lower. Our explanation was that a head which fits early grabs the easy part of the loss and starves the adapters. So we tested it, our own idea: freeze the head for the first epoch, the reverse of LP-FT. It fell 12.7 points behind after epoch 1, caught up by epoch 6, and ended level: 74.33 against 74.28. So *when* the head learns does not matter, only *how fast*. Our explanation was wrong; the one left standing is that those starts are built from the training photos and start over-confident." This is the slide for "novel ideas, including failed ones".

**9. Converged (0:45).** H12, H13, H14. "Four regularisers on top: all within 0.3 points, all ties. Training for the whole budget: no gain on either track. Layer-wise lr decay, which helps full fine-tuning, cost the adapters 1.5. At 224 px the recipe had converged."

**10. Pixels (1:15).** H17, H19, H20. "So we changed what the model sees. Testing a 224 px model at 288 px: +1.7, no retraining, because random training crops make objects look larger than a centred test crop. Training at 288 px: another +2.0, on both tracks. Fine-grained classes differ in small details that 224 px loses. That was the largest single gain of the project."

**11. The tie breaks (0:45).** H3 at 288 px. "At 224 px, AdaptFormer alone tied the combined design and is 20% faster, so we nearly chose it. Re-checked at the final 288 px training size, it was 0.87 lower, outside the noise bar, so we kept both adapters. Lesson: re-check the deciding comparison at the final settings."

**12. Final models and selection (1:30).** "One rule: among options within 0.5 points of the best, take the cheapest. 5% model: LoRA rank 12 plus AdaptFormer 208 in all 12 blocks, 4.99%, label smoothing, head at half rate, trained at 288, tested at 320 px: 78.75% hold-out, 11.8 ms, 1.70 h in 48 GB-equivalent. Full fine-tuning: layer-wise decay, slow head, smoothing, same sizes: 80.14%, 7.3 ms, 1.67 h. 352 px scored 0.31 higher for full fine-tuning, but within the tie rule, so 320. What went to the test set: the rules allow only models submitted in Phase 1, so for the 5% track we submitted run 30, the 224 px version of this recipe on all 45 images per class, tested at 288 px; for full fine-tuning, run 39 itself at 320 px. Both test scores land close to their hold-out estimates." Then the two CodaBench test scores (slides only, not the repo).

**13. Track comparison (1:15).** H16. "The brief expected efficient adaptation to beat full fine-tuning on small data. Here it did not: full fine-tuning leads by 1.4 to 2.1 points at equal epochs, equal budget and equal resolution, and the seed check says the gap is real. What 5% buys: 12M parameters stored per task instead of 93.5M. What it costs: about 1.4 points and 60% more latency, because AdaptFormer cannot be merged into the weights. It does not save training time: the frozen backbone still runs forward and backward. Our reading: full fine-tuning was regularised just as carefully, and 450,000 images is small per class but not small overall. On the test set the gap looks larger, because our 5% submission was trained at 224 px and the full fine-tuning one at 288 px: that adds the 2-point resolution gain to the method gap."

**14. Lessons (1:00).** "Four things we would tell another team. One: adapter coverage and size mattered more than adapter type. Two: treat the head as part of the adaptation; its learning rate was our most sensitive setting. Three: a better initialisation can hurt, and the obvious explanation for why can be wrong: test it. Four: when the recipe stops moving, change the input. What we would do differently: check image size in week 1, not week 5, then tune at 224 px. Scoreboard: 8 hypotheses supported, 8 rejected, 3 split."

### Visuals still to make

Ready: `h1_h4_design.png`, `params_vs_accuracy.png`, `h5_h8_overconfidence.png`, `h12_h14_recipe.png`, `h9_h11_head.png` (backup). To draw: sample image grid (needs images from Colab), ViT block diagram, hold-out and loop diagram, loss curves (from `results/abl_combined/log.jsonl`), head-lr curve, head-delay per-epoch curve (runs 52 and 26), test-size curve, design dumbbell at three sizes, two framework diagrams, track comparison bars.

### Likely questions (Q&A, 5 minutes)

| Question | Answer |
|---|---|
| Why is your zero-shot 35.7, below the brief's 43.7? | Our class means use 10 images per class (the hold-out is excluded). From 43 images it is 43.30, close to the brief's. That stronger start then ended lower (H9) |
| Why not Adapter, prefix tuning or selected layers? | Time; we chose depth over breadth: one recipe, tuned, with the head and resolution studies. BitFit and the linear probe cover "train existing weights" and "head only". Name it as a limitation |
| Is 0.5 points really noise? | Measured: a second seed moved runs by 0.09 and 0.25 (runs 47, 48). Two seeds per recipe bound the spread; they do not measure it precisely |
| Why 320 px and not 352? | 352 was 0.31 higher for full FT and level for the 5% model; within our 0.5 tie rule we take the cheaper size (about 17% fewer patches) |
| Didn't you choose settings on the same hold-out you report? | Yes, so the best scores are slightly optimistic. The test scores are the unbiased check; next time we would split the hold-out into a selection half and a reporting half |
| Why can't you merge the adapters for speed? | LoRA alone can be merged into qkv. AdaptFormer is a parallel branch with a ReLU, so it stays a separate layer at inference |
| Why does layer-wise decay help full FT but hurt adapters? | In full FT it protects the pre-trained early layers. With a frozen backbone there is nothing to protect, so it only slows the early adapters |
| Does the hold-out ranking match the test ranking? | Answer with the two CodaBench scores |

## Hypotheses (numbers as in `EXPERIMENT_LOG.md`)

Show the hypothesis as the title or subtitle of each iteration slide, and the verdict as a stamp (supported / rejected / split). The wording is reconstructed from the config notes, not pre-registered: say so if asked. Of the 24 in the log, 22 tested hypotheses are shown in 19 rows (three pairs merged), H11 is shown as a finding and H18 goes to backup; none is open. Scoreboard for slide 14: 8 supported, 8 rejected, 3 split (H2 + H3: a tie at 224 px, the combined design ahead when trained at 288 px; H13 + H15: helps one track, hurts the other; H20: lifts the 5% design, but the gap stays).

| # | Hypothesis | Verdict | Slide |
|---|---|---|---|
| H1 | More adapter parameters beat the brief's small LoRA | Supported. First +2 to +2.3, confounded (the small LoRA also trained 7 epochs at a higher lr); re-tested at equal epochs and lr (run 50): rank 116 is +1.04 over rank 8 (69.72). About half the first gap was size, half training | 4, 5 |
| H2 + H3 | Adapting attention and FFN beats either alone at equal 5% | Split: a tie untuned (all within 0.33) and on the tuned recipe at 224 px (AdaptFormer alone -0.50; -0.37 tested at 288 px), but trained at 288 px the combined design leads by 0.87 (run 49). It pays only at the final resolution | 4, 11 |
| H4 | Late blocks carry the adaptation, so spend the 5% on the last 6 | Rejected: -0.92 | 4 |
| H5 | Label smoothing helps, because the model is over-confident on 45 images | Supported: +1.12 (5%), +1.45 (full FT) | 6 |
| H6 + H8 | A lower lr helps, a higher one hurts (one sweep: 5e-4 to 5e-5) | Supported down to about 1e-4: 69.28 at 5e-4, 71.05 at 3e-4, 72.37 at 2e-4, 73.02 at 1e-4 (with label smoothing); 5e-5 level | 6 |
| H7 | Label smoothing and a lower lr add up | Rejected: no more than either alone, so they fix the same problem | 6 |
| H22 | Smoothing towards similar classes beats uniform smoothing (novel) | Rejected: 73.67 against 74.28 (-0.61, run 53), above the noise bar; top-5 also lower (90.12 against 90.47). The hold-out loss is lower (1.222 against 1.311), so the predictions are better calibrated but less often right | 6 |
| H9 | A better head start gives a better final model | Rejected three times: -1.07, -1.03, -4.00, though zero-shot rises | 8 |
| H10 | The head should learn more slowly than the adapters | Supported: +1.12 at 0.5x; the most sensitive setting | 7 |
| H21 | Our explanation of H9: an early-fitting head starves the adapters, so freezing the head for the first epoch helps (novel; the reverse of LP-FT) | Rejected: 74.33 against 74.28 (+0.05, run 52). 12.7 points behind after epoch 1, level by epoch 6, same final loss: the timing of the head does not matter | 8 |
| H12 | A stronger regulariser adds accuracy on top of the slow head | Rejected: four ties | 9 |
| H14 | Using the whole budget improves on 10 epochs | Rejected: -0.10 (5%), +0.20 (full FT) | 9 |
| H13 + H15 | Layer-wise lr decay helps both tracks | Split: helps full FT modestly (+0.51), hurts the 5% design (-1.50). A technique does not carry over between tracks | 9, 13 |
| H16 | Tuned, the 5% design matches full fine-tuning | Rejected: gap of 1.7 to 2.1 at equal epochs and equal budget | 13 |
| H17 | Testing above the training size helps (FixRes) | Supported: +1.61 to +1.71, no retraining | 10 |
| H19 | Training at a higher resolution beats only testing at it | Supported: +2.02, the largest single gain | 10 |
| H20 | Training at 288 px lifts the 5% design too, and so changes the track gap | Split: +1.95 for the 5% design (77.94 at 288 px), as much as full FT, so the gap stays at 1.75 | 10, 13 |
| H23 | The biases alone (BitFit, 0.12%) recover most of the adapters' gain over the linear probe | Supported: 71.30 (run 51), 74% of the gain with 2.4% of the adapters' parameters and no extra latency; 2.98 below the 5% design | 5 |
| H24 | Another seed moves the result by less than the 0.5-point noise bar | Supported: -0.09 (5%) and -0.25 (full FT) with seed 1 (runs 47, 48); the gap is 1.59 to 1.75, so it is real, and the noise bar is conservative | 2 |

Not shown as hypotheses:

- **H11 (finding, slide 7):** training the head alone gives about 70% of the gain over zero-shot, the adapters the other 30% (donut). It is a measurement that could not have been rejected, so no verdict stamp.
- **H18 (backup):** averaging two full fine-tuning models gives +0.39 for twice the inference cost. Not an adaptation method; it stays in the "rejected" bullet of slide 12.

## Design considerations (slides 3 and 12, and one caption per design slide)

| Consideration | What we weighed | How it shaped the design |
|---|---|---|
| Parameter budget | <5% of the backbone, head not counted | Spend the whole 5%: LoRA r=12 + AdaptFormer b=208 = 4.998% (4.992% at 288 px, where the backbone's position embeddings grow). AdaptFormer alone, b=232, ties it at 224 px training but is 0.87 lower at 288 px training (run 49) |
| Compute budget | 96 GB-h; A100 about 1,100 img/s in bf16 | Fixed 10-epoch runs keep comparisons fair; heavy augmentation makes the data loader the limit |
| Few-shot overfitting | 45 images per class, 10k classes | Choices aimed at overconfidence: label smoothing, lower lr, slower head |
| Inference latency | Added modules cost time on every forward pass | LoRA can be merged into the weights; AdaptFormer cannot. The combined model cannot merge its LoRA either, because AdaptFormer stays: 11.8 ms, against 9.5 ms for AdaptFormer alone and about 7 ms for full FT (batch 1; the same at 288 to 352 px, run 46). AdaptFormer alone would be the faster final model (9.8 against 12.6 ms logged at 288 px), but at 288 px training it is 0.87 lower, outside the 0.5-point tie rule |
| Storage per task | The point of PEFT | 12M against 93.5M params stored per task, head included |
| Keep the pre-trained features at step 0 | Avoid damaging them before training starts | Zero-initialised `lora_B` and `up`; head from class means, calibrated |

Per slide:

- **3 Methods:** LoRA changes what tokens attend to; AdaptFormer changes what the features are. We expected them to be complementary, so we tested each alone and combined at equal budget. The linear probe is the "new head only" contrast, and BitFit the "tune existing weights, add nothing" contrast: it trains only the backbone's biases, so it adds no modules and no latency.
- **4 Equal budget:** later blocks are more task-specific, so putting twice the width in the last 6 blocks seemed plausible. Rejected: early blocks need adapting too.
- **5 Params vs accuracy:** the curve is steep for the first 0.1% and flat after: BitFit's 0.12% recovers 74% of the adapters' gain over the linear probe. The question "why spend 5%?" has a measured answer: the extra 4.9% buys about 3 points, at the cost of 4 ms latency. The track rules reward accuracy, so we spend the full 5%; for a latency- or storage-bound deployment BitFit is the better trade.
- **6 Diagnosis:** the brief hints at loss design; overconfident memorisation pointed to label smoothing.
- **7-8 Head:** the head has 7.7M params against the adapters' 4.3M, so it gets its own lr. A stronger head start raises zero-shot accuracy but leaves less for the adapters to learn, so we chose the gentle start.
- **9 Recipe:** layer-wise lr decay protects pre-trained layers in full fine-tuning; with a frozen backbone it only slows the early adapters. Kept only regularisers that cost nothing at inference.
- **10 Pixels:** training at 288 px costs about 1.65x per epoch, affordable because 10 epochs already fit. Position embeddings are interpolated, so no new parameters.
- **12 Full fine-tuning:** protect the pre-trained features with layer-wise decay 0.75 and a slow head, rather than freezing anything.
- **12 Selection:** a tie goes to the cheaper option, and a measured result beats an assumed one. 320 px ties 352 px on both tracks (-0.31 full FT, +0.02 5%) with about 17% fewer patches (400 against 484): less compute per image, though the same batch-1 latency on an A100 (run 46). AdaptFormer alone tied the combined design at a 224 px training size and is about 20% faster, but trained at 288 px it is 0.87 lower (77.07 against 77.94, run 49), so the tie rule no longer picks it.
- **13 Comparison:** PEFT saves storage; it does not save training time (the frozen backbone still runs forward and backward) and costs latency unless merged.

## Results explained

Each finding: what we saw, our explanation, how sure we are.

### 1. Adapter type does not matter at 5%; coverage and size do (slide 4; H1, H2 + H3, H4)

| Design (5%, 10 epochs, lr 3e-4) | Top-1 |
|---|---|
| LoRA only | 70.76 |
| AdaptFormer only | 71.09 |
| LoRA + AdaptFormer | 71.05 |
| AdaptFormer, last 6 blocks only | 70.17 |
| LoRA r=8 (0.34%), same settings (run 50) | 69.72 |
| LoRA r=8 (0.34%, 7 epochs, lr 5e-4; first run) | 68.78 |

- **Explanation:** with 5% of the parameters, any of these designs has enough capacity to fit 45 images per class; the limit is generalisation, not capacity. Concentrating on the last blocks loses about a point, which suggests the early layers also need to shift for fine-grained detail.
- **Confidence:** the tie is solid (all within 0.33). Re-tested on the tuned recipe, AdaptFormer alone was 0.50 lower at 224 px (73.78 against 74.28) and 0.37 lower at 288 px (75.62 against 75.99): still a tie, and AdaptFormer alone is about 17% faster (9.7 against 11.7 ms at 288 px). **But trained at 288 px, the final resolution, it is 0.87 lower (77.07 against 77.94, run 49)**: with more pixels, adapting attention as well starts to pay. A second seed moves a run by at most 0.25 (H24, runs 47 and 48), so 0.87 is a real difference.
- **Size, without the confound:** the first small-LoRA run also trained fewer epochs at a higher lr. Re-run at equal settings (run 50), rank 8 scores 69.72: rank 116 is +1.04, so size helps, but only about half of the first +2 gap was size. Rank 8 is ahead for the first 3 epochs and flattens from epoch 7; rank 116 keeps rising, so the extra capacity pays late.

### 1b. Most of the adaptation is cheap (slide 5; H23)

Same recipe (`imp_trivialaug`: label smoothing 0.1, head lr 5e-5, TrivialAugment + random erasing, 10 epochs, 224 px):

| Trained in the backbone | Run | Trainable % | Top-1 | Latency (batch 1, 224 px) |
|---|---|---|---|---|
| nothing (head only) | `abl_linear_probe` | 0 | 62.62 | 8.5 ms (same plain model; timing noise) |
| biases only (BitFit) | `imp_bitfit` | 0.12 | 71.30 | 7.4 ms |
| LoRA + AdaptFormer | `imp_trivialaug` | 4.998 | 74.28 | 11.8 ms |
| everything | `full_ft_llrd_10ep` | 100 | 76.06 | 7.1 ms |

- **Seen:** the biases alone recover 8.68 of the 11.66 points the adapters add over the linear probe (74%), with 2.4% of their parameters (102,912) and no extra latency. Converged by epoch 10 (71.04 to 71.30 over the last 4 epochs).
- **Explanation:** most of the adaptation is a small shift of the pre-trained features, which a bias can express; this matches run 45, where class means of the adapted features already reach 69.82%. The last 3 points need the adapters' 40 times more parameters.
- **Confidence:** BitFit used bias lr 5e-4 (the 5% runs used 1e-4) and was not tuned, so 71.30 is a lower estimate for BitFit. The 2.98-point gap to the 5% design is far above the noise bar.

### 2. The limit was memorisation (slide 6; H5, H6 + H8, H7)

- **Seen:** at the end of training, train loss 0.07 against hold-out loss 1.42.
- **Label smoothing 0.1:** 72.17 (+1.12). **Lower lr 2e-4:** 72.37 (+1.32). **Higher lr 5e-4:** 69.28. **Both together:** 72.35, no more than either.
- **Explanation:** both changes stop the model from becoming over-confident on its 45 images, so they fix the same problem and do not add up. With label smoothing, lowering the lr further to 1e-4 reached 73.02; 5e-5 was level (73.52, at the noise bar).

### 3. The head is the key adaptation lever (slides 7 and 8; H9, H10, finding H11)

| Head lr (adapters at 1e-4) | Top-1 | Hold-out loss |
|---|---|---|
| 2x | 71.75 | 1.570 |
| 1x | 73.02 | 1.420 |
| 0.5x | **74.14** | 1.325 |
| 0.25x | 74.02 | 1.335 |

- **Better head starts all ended lower:** class means from 43 images (-1.07, and -1.03 on the tuned recipe); a head already trained by a linear probe (-4.00).
- **First explanation, tested and rejected (H21, run 52):** a head that fits the training images early takes the easy part of the loss and starves the adapters. Freezing the head for the first epoch put the run 12.7 points behind after epoch 1, level by epoch 6, and it ended at 74.33 against 74.28 with the same hold-out loss. When the head learns does not matter.
- **Explanation left standing (untested):** the stronger starts are built from the training images themselves, so they start closer to the training set and more over-confident; the slow head rate (H10) is what keeps that in check, which fits H10 being the most sensitive setting. In the worst case, train loss ended lower (1.90 against 2.04) while hold-out loss ended higher (1.54 against 1.31).
- **Head vs adapters:** training the head alone takes 35.7 to 62.62, about 70% of the gain to 73.02; the adapters give the other 30%.
- **Class means of the adapted features (run 45, no training):** 69.82% at 288 px with no trained head, against 43.30% on the frozen features (224 px, so the gap of about 25 points is approximate). Most of the gain is already in the features; the trained head adds about 6 points on top. Combining the head with the class-mean classifier is a tie (-0.15, cross-fitted): the head no longer overfits in a way a parameter-free classifier can fix.
- **Zero-shot against the brief's 43.7:** our 35.7 uses class means from 10 images per class (the hold-out images are excluded); from 43 images it is 43.30, close to the brief's figure.

### 4. The recipe converged (slide 9; H12, H14; H13 also on slide 9)

- Four regularisers on top of the 74.14 model all tie: stochastic depth 74.09, EMA 74.19, Mixup/CutMix (15 epochs) 73.97, TrivialAugment + random erasing 74.28.
- The whole budget instead of 10 epochs: -0.10 on the 5% track, +0.20 on full fine-tuning.
- Layer-wise lr decay on the 5% design: 72.78 (-1.50). It helps full fine-tuning (+0.51), which shows that a technique does not carry over between tracks automatically.
- **Explanation:** once the head is slow and the loss is smoothed, the model no longer overfits much, so extra regularisation and extra epochs have little left to fix.

### 5. More pixels gave the largest gain (slide 10; H17, H19)

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

### 6. Tracks: full fine-tuning wins, against the brief's expectation (slide 13; H16, H20)

| Tuned | 5% | Full FT |
|---|---|---|
| 10 epochs, 224 px test | 74.28 | 76.06 |
| 10 epochs, 288 px test | 75.99 | 77.67 |
| Whole budget, 288 px test | 75.99 | 77.89 |
| Trained at 288 px, 288 px test | 77.94 | 79.69 |
| Trained at 288 px, 320 px test (chosen size) | 78.75 | 80.14 |
| Trained at 288 px, best test size | 78.75 (320 px) | 80.45 (352 px) |
| Params stored per task, incl. head | 12M | 93.5M |
| Best hold-out model, score at 320 px | 78.75 (run 42) | 80.14 (run 39) |
| Latency, batch 1, 320 px (chosen size; median of 5 x 100 passes) | 11.8 ms | 7.3 ms |
| Training, 48 GB-equivalent | 1.19 h | 1.67 h (288 px) |

- **The brief expected** PEFT to win with small data. Possible reasons it did not, here: full fine-tuning was regularised just as carefully (label smoothing, slow head, layer-wise decay), so it did not overfit as feared; 450k images is not small overall; and the backbone was already close to the task.
- **Fair comparison:** the gap of about 2 points holds at equal epochs, at equal budget and at equal training size: trained and tested at 288 px it is 1.75 (79.69 against 77.94, H20), and 1.39 at the chosen test size of 320 px (80.14 against 78.75). Training at 288 px lifts both tracks by about 2 points, so resolution and method are separate effects.

## Model selection process (slide 12)

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
| 7. Final model | Which is the best 5% model? | `imp_trivialaug_288` (run 42), at 320 px | 78.75 |
| 8. Test submission | Which model may go to the test set? | Only models submitted in Phase 1 qualify, and run 42 was trained after it closed: `final_5pct_trivialaug` (run 30, the 224 px recipe on all 45 images per class), at 288 px | 75.99 (hold-out estimate, run 26 at 288 px) |

### The funnel, full fine-tuning track

| Stage | Picked | Top-1 |
|---|---|---|
| Baseline | `full_ft_baseline` | 72.53 |
| + label smoothing | `full_ft_ls` | 73.98 |
| + slow head | `full_ft_ls_headlow` | 75.75 |
| + layer-wise decay 0.75 (10 epochs) | `full_ft_llrd_10ep` | 76.06 |
| Train at 288 px | `full_ft_llrd_288` (79.69 at 288 px) | 80.45 at 352 px |
| Test size | 320 px: ties 352 (-0.31) with about 17% fewer patches | 80.14 |
| Submitted model | `full_ft_llrd_288` itself (run 39), at 320 px: its hold-out score is measured | 80.14 |

### What we rejected and why

- **Ensemble of two full fine-tuning models:** +0.39 for twice the inference cost.
- **Using the whole budget:** no gain over 10 epochs on either track.
- **Stronger head starts and layer-wise decay on adapters:** both lowered accuracy.
- **AdaptFormer alone as the final 5% model:** it ties the combined design at 224 px training and is about 20% faster, but trained at 288 px it is 0.87 lower (run 49), outside the 0.5-point rule, so the combined design (run 42) was kept.
- **352 px test size for full FT:** +0.31 over 320 px, within noise, for about 21% more patches (same batch-1 latency on an A100).
- **Short fine-tune at 288 px instead of training at 288 px (run 44):** 0.66 lower at 288 px and 1.02 lower at 320 px, for about 11% less compute; useful only when a 224 px model already exists.
- **Head + class-mean ensemble at test time (run 45):** -0.15, a tie; an extra 768 x 10,000 product for nothing.

### Limits of the process

- The test size and some settings were chosen on the same hold-out they are reported on, so the best numbers are slightly optimistic.
- The test submissions are not a matched pair: run 39 (full fine-tuning, trained at 288 px on 43 images per class) and run 30 (5%, trained at 224 px on all 45). The fair track comparison is on the hold-out (runs 39 and 42); the test scores check that the hold-out estimates hold.
- Settings never changed alone: weight decay, the adapters' scale 0.1, LoRA alpha, the r=12 / b=208 split, and AdaptFormer's bottleneck size at a fixed 5% (232 is simply the largest that fits).

### What we would do differently (slide 14, one line; report discussion)

- **Screen the large settings first, cheaply.** Resolution was the largest lever (+1.7 from testing at 288 px, H17; +2.0 more from training at 288 px, H19 and H20), but we found it late: the first test-size check was run 22 (22 Oct), the first 288 px training run 39 (30 Oct). Scoring the first baseline at 224 / 288 / 320 px costs minutes and no training (`common.evaluate --sizes`), and would have shown the direction in week 1.
- **Still tune at 224 px.** Training at 288 px takes about 1.7 times as long per epoch, and most findings carried over: label smoothing, the slower head, the lr range and the regulariser ties were found at 224 px, and training at 288 px lifted both tracks by about 2 points, leaving the gap almost unchanged.
- **Re-check the deciding comparisons at the final settings.** The one finding that did not carry over is the design tie: AdaptFormer alone tied the combined design at 224 px but was 0.87 lower at 288 px (run 49). So the comparisons that pick the final model (adapter design, test size) should be repeated at the final resolution, not every ablation.
- **Keep part of the hold-out for the final report only.** The test size and other settings were chosen on the same hold-out they are reported on. Splitting it into a selection half and a reporting half (1 image per class each) removes that bias at no training cost.
- **In one line for the slide:** screen first the settings that change what the model sees and how long it trains, then run the detailed ablations; we did it the other way round.
