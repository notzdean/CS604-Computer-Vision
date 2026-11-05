# How the hypotheses unfold

A plain-language walk through the hypotheses (H1 to H24) in `EXPERIMENT_LOG.md`. All numbers are hold-out top-1 (2 images per class held out of train); differences under 0.5 points are noise. Leaderboard scores are not in this repo.

**The setting.** We have a model that has already seen millions of pictures, so it knows what things look like in general. Our job is to teach it 10,000 new, very similar categories, with only 45 example photos each. We have two ways to do it:

- **Full fine-tuning:** let the whole model change.
- **The 5% track:** freeze the model and add small "adapter" parts that learn, under 5% of the model's size.

Before training anything, we give the model a starting answer sheet: for each class, the average of its photos' features. That alone gets **35.7%**.

---

### Chapter 1: Where should the small parts go? (H1–H4)

First we tried the brief's small LoRA (0.34%) and got 68.8%.

- **H1:** using the full 5% allowance did better, about 71%. *Bigger adapters help.* There is a catch: the small run also trained for fewer epochs. Re-run at equal settings (run 50), rank 8 scores 69.7%: bigger adapters still win, by 1.0 point, about half the first gap.
- **H2:** we tried adapters on the attention part, on the feed-forward part, and on both. All three tied. *Where they go doesn't matter much.*
- **H4:** putting everything in the last 6 layers was worse. *Spread the adapters across the whole model.*

**Lesson:** the design was not the bottleneck, so we turned to how we train.

### Chapter 2: The model is too sure of itself (H5–H8)

With only 45 photos per class, the model memorises them and becomes over-confident.

- **H5:** label smoothing tells it "never be 100% sure". That gave **+1.1**, and helped full fine-tuning too.
- **H6, H8:** learning more slowly (a lower learning rate) also helped, down to about 1e-4.
- **H7:** the two together were no better than either alone. *Both fix the same problem: over-confidence.*

We reached **73.0%**.

### Chapter 3: The surprising role of the classifier head (H9–H11)

The head is the final layer that turns features into one of the 10,000 answers.

- **H9:** we expected a better starting head to give a better final model. It got worse, three times (up to −4 points). Something odd was happening.
- **H10:** making the head learn at half the adapters' speed gave **+1.1**. This was the most sensitive setting we found. We reached **74.1%**.
- **H11:** training only the head gives about 70% of the total gain. The adapters give the other 30%.

**Our first explanation, tested and rejected:** a head that learns early grabs the easy wins and leaves the adapters little to learn from. Run 52 (H21) froze the head for the first epoch: it fell 12.7 points behind, caught up by epoch 6 and ended level (74.33 against 74.28). So *when* the head learns does not matter; *how fast* it learns does. The explanation left standing, untested: the "better" starts are built from the training images, so they start over-confident.

### Chapter 4: Have we squeezed it dry? (H12–H14)

- **H12:** we tried four extra regularisers: stochastic depth, EMA, Mixup and TrivialAugment. All four tied, within 0.3 points of each other.
- **H13:** layer-wise learning-rate decay, a trick that helps full fine-tuning, *hurt* the 5% design (−1.5).
- **H14:** training longer, over the whole budget, gave nothing.

**Lesson:** the 5% recipe has converged at about **74.3%**. More tricks won't move it.

### Chapter 5: The other track catches up and passes (H15–H16)

- **H15:** full fine-tuning with layer-wise decay improved modestly and reached **76.3%**.
- **H16:** we hoped the tuned 5% design would match it. It didn't. The gap is about **2 points**, and it holds at equal training time and at equal budget.

**Lesson:** updating all the weights does better here than adding small parts. This is the honest answer to the project question.

### Chapter 6: More pixels (H17–H20)

Fine-grained classes differ in small details, so image size might matter.

- **H17:** testing on larger images (288 px instead of 224), with no retraining, gave **+1.7** for free.
- **H18:** averaging two models gave only +0.4, for twice the cost. Not worth it.
- **H19:** *training* at 288 px gave another **+2.0** for full fine-tuning (**79.7%**). This was the biggest single gain of the project.
- **H20:** training the 5% design at 288 px helped it just as much (+1.95, **77.9%**). The gap stayed about the same.

**Lesson:** resolution and method are separate effects. More pixels lift both tracks equally.

**A twist from Chapter 1 (run 49):** at 288 px, the AdaptFormer-only design scored 77.07%, 0.87 behind the combined design. At 224 px the designs tied, but with more pixels, adapting attention as well starts to pay. So we keep the combined design, even though AdaptFormer alone is about 20% faster.

Two side findings:
- **A cheaper shortcut (run 44):** two extra epochs at 288 px on an already-trained model gets most of the gain, for a fifth of the GPU time.
- **Class averages alone (run 45):** after adapting, the class-average answer sheet alone scores 69.8%. The adapters mostly build *better features*; the head just reads them out.

### The ending: our best models, and what we submitted

Each track's best run, tested at 320 px:
- full fine-tuning (run 39): **80.1%**
- the 5% design (run 42): **78.8%**

Both are hold-out scores, not leaderboard scores. This puts the 5% model 1.4 points behind while training under 5% of the weights.

**What went to the test set:** only models submitted in Phase 1 were allowed, and run 42 came later. So we submitted run 39 for full fine-tuning (at 320 px) and run 30 for the 5% track: the same recipe trained at 224 px on all 45 photos per class, tested at 288 px. The test scores are on the slides. Because the 5% submission was trained at 224 px, its test gap to full fine-tuning is wider than the 1.4 points above.

---

### Epilogue: late questions (runs 50–53)

| Question | Hypothesis | Run | Answer |
|---|---|---|---|
| Is "more parameters help" still true once training time is equal? | H1 re-test | 50 | Yes: +1.04 for rank 116 over rank 8 |
| **Novel:** does holding the head back first help the adapters learn? | H21 | 52 | No: +0.05, a tie |
| **Novel:** does smoothing towards *look-alike* classes beat smoothing over all 10,000? | H22 | 53 | No: -0.61 (73.67 against 74.28); better calibrated, less often right |

**How few weights do you need? (run 51):** training only the bias terms (BitFit, 0.12% of the model) reached 71.3%. That is three quarters of the way from the head-only model (62.6%) to our 5% design (74.3%), with 40 times fewer trained weights and no slowdown. The last 3 points are what the full 5% buys.

**Was it luck? (runs 47, 48):** we repeated both final recipes with a different random seed. The scores moved by only 0.09 (5%) and 0.25 (full fine-tuning), and full fine-tuning stayed ahead by 1.6 points. So our 0.5-point noise bar is on the safe side, and the gap between the tracks is real.

### If we did it again

Resolution was the biggest lever, yet we found it near the end (testing at 288 px on 22 Oct, training at 288 px on 30 Oct). Next time:

1. **Week 1: check the big settings cheaply.** Score the first model at several test sizes (minutes, no training), and run one short pilot at 224 against 288 px.
2. **Tune at 224 px**, where runs are fast. Most findings carried over to 288 px.
3. **Re-check only the deciding comparisons at the final settings.** The design tie broke at 288 px (run 49), so that kind of check matters.
4. **Keep part of the hold-out untouched** for the final report, so the chosen settings are not scored on the same images that picked them.

**The whole story in one line:** the design didn't matter; how we trained did, above all a slow, gentle head. Then the recipe stopped improving, full fine-tuning stayed about 2 points ahead, and larger images helped both tracks equally.
