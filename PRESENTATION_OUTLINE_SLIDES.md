# Presentation outline: the Slides artifact version (19 slides)

This outline follows the slide-deck artifact "10,000 Classes, 45 Photos — Slides" (built from the published page of the same name; link in the team chat, not here, because the repo is public). The 14-slide plan in `PRESENTATION_OUTLINE.md` is the alternative; its "Likely questions" table applies to both.

All scores are hold-out top-1 (2 photos per class held out, 20,000 images). Under 0.5 points is a tie. Test (CodaBench) scores are not on these slides and not in this file.

**Storyline:** the adapter design barely mattered; how the model was trained did, above all a slow, gentle head, and then more pixels. Full fine-tuning stayed 1.4 to 2 points ahead.

## Run of show (about 14 minutes, 1 minute of slack)

| # | Slide | On the slide | Section | Speaker | Time | Clock |
|---|---|---|---|---|---|---|
| 1 | 10,000 classes, 45 photos each | Title; tiles 35.7% (before training), 78.8% (5%), 80.1% (full FT) | Setup | 1 | 0:30 | 0:30 |
| 2 | Frozen ViT-B/16, adapters in every block | Model diagram: photo → patches → 12 frozen blocks → head; inside a block, LoRA r=12 beside attention, AdaptFormer 768→208→768 beside the FFN | Setup | 1 | 0:45 | 1:15 |
| 3 | Every experiment had to fit these rules | Six cards: one model, little data, two tracks, budget, how the 5% is counted, what is measured | Setup | 1 | 0:30 | 1:45 |
| 4 | 96 GB-hours per run, and the brief’s reference results | Budget bars per GPU; reference table 43.7 / 70.8 / 68.3 | Setup | 1 | 0:20 | 2:05 |
| 5 | Each step is one idea we kept | Climb table, 5% vs full FT, 68.78 → 78.75 and 72.53 → 80.14 | Climb and design | 1 | 0:45 | 2:50 |
| 6 | Where the adapters go matters; which adapter does not | Dot chart: rank 8 (68.78 first run, 69.72 equal settings), LoRA 70.76, AdaptFormer 71.09, both 71.05, last 6 blocks 70.17 | Climb and design | 1 | 1:00 | 3:50 |
| 7 | Most of the adaptation is cheap | Dot chart: head only 62.62, BitFit 71.30, 5% 74.28, full FT 76.06; cards 74%, 40×, 7.4 ms | Climb and design | 2 | 0:50 | 4:40 |
| 8 | The model was too sure of itself | Dot chart: lr 5e-4 to 5e-5, with and without smoothing; neighbour smoothing (H22) −0.61 | How it learns | 2 | 1:00 | 5:40 |
| 9 | The head should learn slowly | Dot chart: head ×2 −1.27, ×0.5 +1.12, ×0.25 +1.00; gain bar 35.7 → +26.9 head → +10.4 adapters | How it learns | 2 | 0:50 | 6:30 |
| 10 | A better head start ends lower, and our first explanation was wrong | Dot chart: −1.07, −1.03, −4.00; head frozen for epoch 1 (H21) +0.05 | How it learns | 2 | 1:00 | 7:30 |
| 11 | After that, nothing moved the 5% recipe | Dot chart: four regularisers within ±0.17; layer-wise decay −1.50; whole budget −0.10 | How it learns | 2 | 0:40 | 8:10 |
| 12 | More pixels lifted both tracks equally | Paired dots, 5% vs full FT at four sizes, gap 1.78 / 1.68 / 1.75 / 1.39; cards +1.7, +2.0, 1.4–1.8 | Pixels and results | 2 | 1:00 | 9:10 |
| 13 | The tie breaks at higher resolution | AdaptFormer alone vs both: −0.50, −0.37, −0.87; the cheapest-within-0.5 rule | Pixels and results | 2 | 0:40 | 9:50 |
| 14 | Our best models | Tiles: full FT 80.1% (run 39, 7.3 ms, 1.67 h), 5% 78.8% (run 42, 11.8 ms, 1.70 h), difference 1.4 pts; box: what went to the test set (runs 39 and 30) | Pixels and results | 3 | 1:00 | 10:50 |
| 15 | Was it luck? A second seed says no | Dot chart: 77.94 / 77.85 and 79.69 / 79.44 | Pixels and results | 3 | 0:30 | 11:20 |
| 16 | What didn’t work, and what it taught us | Table of 10 failed ideas; our two (H21, H22) shaded | Lessons | 3 | 0:50 | 12:10 |
| 17 | Five practical problems, and our fixes | T4 too slow, budget check, two processes (run 40), folder reuse, run 43 dropped | Lessons | 3 | 0:30 | 12:40 |
| 18 | If we did it again | Four cards: screen image size in week 1, tune at 224 px, re-check deciding comparisons, split the hold-out | Lessons | 3 | 0:40 | 13:20 |
| 19 | In one line | “The adapter design barely mattered; how we trained did, above all a slow, gentle head.” Scoreboard 8 / 8 / 3; Questions? | Lessons | 3 | 0:30 | 13:50 |

If time runs short, cut in this order: slide 4 (say the budget in one sentence on slide 3), slide 17, slide 5 (the climb is repeated by the chapters).

## Talk track

**1. Cover (0:30).** Same task as everyone here. Every score is hold-out top-1. Before training: 35.7%. Our best 5% model 78.8%, full fine-tuning 80.1%. The talk is how we got there, including what failed.

**2. The model (0:45).** The backbone is frozen in the 5% track. Every block gets LoRA beside the attention weights and an AdaptFormer side path beside the feed-forward layer: 4.29M weights, 4.99% of the backbone. The head is new, always trained, and not counted in the 5%.

**3. The rules (0:30).** One checkpoint, 45 photos per class, two tracks, the budget, the head-excluded 5%. Because val and test have no labels, we held out 2 photos per class and changed one setting per run.

**4. Budget and references (0:20).** 96 GB-hours: 2.4 h on our A100 40 GB. The brief’s numbers are a guide, not a target.

**5. The climb (0:45).** The whole story in one table: two big moves, the slower head and larger images. The gap between the tracks never closes.

**6. Design (1:00).** H1, H2, H4. Bigger adapters help by 1.0 point at equal settings, half the first comparison’s 2 points (the small run had also trained less). The three designs covering every block tie within 0.33; the last 6 blocks alone lose 0.92. The design was not the bottleneck.

**7. BitFit (0:50).** H23. Biases only, 0.12% of the backbone, reach 71.3%: 74% of what the adapters add over the head alone, with 40 times fewer weights and no extra latency. The full 5% buys the last 3 points.

**8. Over-confidence (1:00).** H5 to H8, H22. Training loss 0.07, hold-out loss 1.42: memorisation. Smoothing +1.1 and a lower lr +1.3, but not together: they fix the same thing. Our idea, smoothing only towards 20 look-alike classes: −0.61, better calibrated but less often right.

**9. Head speed (0:50).** H10, H11. The head has more weights than the adapters, so it got its own rate: half the adapters’ rate +1.1, twice −1.3, our most sensitive setting. The head alone gives about 70% of the gain over zero-shot.

**10. Head start (1:00).** H9, H21. Three stronger starts ended 1 to 4 points lower. We guessed an early-fitting head starves the adapters and tested it, the reverse of LP-FT: freezing the head for epoch 1 put the run 12.7 points behind, level by epoch 6, tie at the end. When the head learns does not matter, only how fast. Left standing, untested: these starts come from the training photos and start over-confident.

**11. Converged (0:40).** H12 to H14. Four regularisers and the whole budget: ties. Layer-wise decay, which helps full fine-tuning, cost the adapters 1.5. At 224 px the recipe had converged, so we changed the input.

**12. Pixels (1:00).** H16 to H20. Testing at 288 px: +1.7, no retraining. Training at 288 px: +2.0 on both tracks, the largest gain. The gap stays 1.4 to 1.8 at every size: resolution and method are separate effects.

**13. Twist (0:40).** H3 at 288 px. AdaptFormer alone tied at 224 px and is 20% faster, so we nearly chose it; trained at 288 px it was 0.87 lower, outside the tie zone. Trust a measurement at the final settings.

**14. Best models (1:00).** Both trained at 288 px, tested at 320 px. Full fine-tuning 80.1% at 7.3 ms; the 5% design 78.8% at 11.8 ms (AdaptFormer cannot be merged into the weights). Test set: only models submitted in the validation phase were allowed, so the submissions were run 39 and run 30 (the 224 px version of the 5% recipe). Say the two CodaBench scores aloud if the team wants them in the talk.

**15. Seeds (0:30).** H24. A second seed moved the scores by 0.09 and 0.25: the 0.5 noise bar is safe and the track gap is real.

**16. Failures (0:50).** Point at the two shaded rows: our own ideas, tested and rejected. The rest either did not transfer or had nothing left to fix.

**17. Practical problems (0:30).** One sentence each; the run-40 duplicate process is the one worth naming.

**18. If we did it again (0:40).** Screen image size in week 1; tune at 224 px; re-check the deciding comparisons at the final settings; keep part of the hold-out for reporting.

**19. Close (0:30).** Read the line. Then the four lessons, if time allows: coverage over adapter type; the head is part of the adaptation; a better start can hurt, so test the explanation; when the recipe stops moving, change the input. Questions.

## Compared with `PRESENTATION_OUTLINE.md`

- **Same findings and numbers**; this version has more setup (rules, budget, climb) and a fuller close (failures, practical problems, what we would change).
- **Not in this deck:** loss curves, a method-comparison slide covering BitFit and the linear probe side by side with LoRA and AdaptFormer, a dedicated track-comparison slide (storage 12.0M vs 93.5M, latency, no training-time saving, the brief’s expectation that PEFT wins on small data), and test scores. Cover the track comparison verbally on slide 14, or add it as a slide.
- **Speaker split:** person 1 slides 1–6, person 2 slides 7–13, person 3 slides 14–19.
