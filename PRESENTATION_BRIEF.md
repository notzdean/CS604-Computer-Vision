# Presentation brief

Slide-by-slide content for the 15-minute project presentation (7 Nov 2026, 15 min + 5 min Q&A). Each slide has a title,
2 to 4 bullet points, the figure to show and short speaker notes. Charts come from `results/results.csv`, `results/<run>/log.jsonl` and `ABLATION_STORY.md`; the figure named on each slide
is the chart to show, drawn natively in the slide tool from that data (reference PNGs: `python figures/make_figures.py .`,
not committed). The storyline is
`ABLATION_STORY.md`. All accuracies are hold-out top-1 (20,000 labelled training images kept out of training). Items
marked **[pending]** wait for a run or for the professor.

**Key message:** we asked *where* to put the adapters, found the real problem was *over-confidence*, and the fix was
training the *classifier head* slowly.

**The brief's requirements this covers:** investigation, motivation of the design, the design, comparison with baselines,
ablation studies; results with top-1, trainable parameters, inference latency, training GPU-hours and loss curves.

| Part | Slides | Time |
|---|---|---|
| Setup | 1 to 3 | 2.5 min |
| Our design | 4 | 1.5 min |
| Ablations | 5 to 10 | 6 min |
| Comparison with full fine-tuning | 11 to 12 | 2.5 min |
| Close | 13 to 14 | 2 min |

---

## 1. Adapting ViT-B/16 to 10,000 classes with 5% of the weights

- CS604 group project, team 01: Nicole, Lilian, Ryan
- One-line message: the classifier head, not the adapters, decided the result

**Figure:** none (title slide).
**Notes (30 s):** Introduce the team and the one question we answer: how close can we get to full fine-tuning while
training only 5% of a pre-trained ViT-B/16, and what actually matters for that?

## 2. The task and the rules

- 10,000 fine-grained classes, 45 labelled images per class (450,000 images); val and test labels are withheld
- Base model fixed: ViT-B/16 pre-trained on ImageNet-21k, fine-tuned on ImageNet-1k
- Budget: 2 h x 48 GB of GPU per run. Two tracks: full fine-tuning, and efficient adaptation under 5% trained
- Counting rule (confirmed by the professor): the 7.7M-weight classifier head is not counted; 5% is of the 85.8M backbone

**Figure:** none, or a small icon row (dataset / model / budget).
**Notes (60 s):** Stress the two numbers that shape everything: only 45 images per class, and 10,000 classes, so the
classifier head alone has 7.7M weights, more than our whole adapter budget. Explain that we score every run on a
hold-out of 2 labelled images per class, because the official val labels are hidden.

## 3. Baselines: from 0% to 100% trained

- Zero-shot (head built from class averages, nothing trained): 35.7%
- Linear probe (head only): 62.6%; LoRA rank 8 (0.34%): 68.8%, close to the brief's 68.3%
- Our 5% model: 74.3%; full fine-tuning: 76.3%

**Figure:** `figures/1_baseline_ladder.png`
**Notes (60 s):** Walk up the ladder. Training only the head already gives most of the gain, which foreshadows the
key finding. Our 5% model closes most of the gap to full fine-tuning.

## 4. Our design: adapters everywhere, frozen backbone

- LoRA (low-rank update) on each block's attention `qkv`, AdaptFormer (small bottleneck) beside each block's FFN
- Together 4.998% of the backbone; the backbone stays frozen
- The head starts from the class averages of the training features, then trains with label smoothing and a slow rate

**Figure:** diagram to draw: one ViT block with the frozen attention and FFN in grey, LoRA on `qkv` and the AdaptFormer
branch on the FFN in colour, repeated over 12 blocks, then the classifier head.
**Notes (90 s):** Explain why both: attention and FFN are the two halves of a block, so we spent the budget on both
(lecture 5: LoRA and AdaptFormer). Mention the head initialisation: each class starts as the average of its training
features, so the model is reasonable before any training.

## 5. Ablation 1: where should the 5% go?

- At the same 5%: LoRA only 70.8, AdaptFormer only 71.1, both 71.1: a tie
- Concentrating the 5% in the last 6 layers is worse: 70.2
- All are about 2 points above LoRA rank 8, so the *number* of adapter weights matters, not where they sit

**Figure:** `figures/2_where_the_5pct_goes.png`
**Notes (60 s):** This was our first question and the answer surprised us: the placement barely matters. So adding
adapters was not the bottleneck, and we went looking for the real one.

## 6. Diagnosis: the model memorises

- Training loss falls to 0.07, but the hold-out loss rises in epochs 2 to 5 and then only stalls
- With 45 images per class the model learns the training images by heart and becomes over-confident
- A higher learning rate (5e-4) makes it worse: hold-out loss climbs from 1.57 to 1.88

**Figure:** `figures/3_loss_curves.png`
**Notes (60 s):** This is the turning point of the talk. The gap between the two curves is the problem; every later
experiment is aimed at it. Mention the high-learning-rate run as the clearest example.

## 7. Ablation 2: calm the model down

- Label smoothing (never 100% sure): +1.1
- Lower learning rate: 3e-4 to 2e-4 to 1e-4, each better
- Both fix the same thing: together at 2e-4 they give no more than either alone
- A better head start (built from all 43 images per class) did worse: the head memorised even sooner

**Figure:** optional small table of the four runs from `ABLATION_STORY.md` chapter 3.
**Notes (60 s):** These are standard fixes for over-confidence and both work. The head-start result is the first clue
that the problem sits in the head.

## 8. Key finding: train the head slowly

- The head has 7.7M weights, more than all the adapters (4.3M), and it memorises first
- Giving the head its own, slower rate: x2 71.8, x1 73.0, **x0.5 74.1**, x0.25 74.0
- About 70% of the gain over zero-shot comes from training the head, 30% from the adapters

**Figure:** `figures/4_head_learning_rate.png`
**Notes (90 s):** The biggest single improvement in the project, worth about 3 points over our first design. The head
starts in a good place (class averages); a slow head keeps that start while the adapters reshape the features under it.
The linear-probe comparison gives the 70/30 split.

## 9. Ablation 3: more regularisation hits a ceiling

- Stochastic depth 74.1, weight averaging 74.2, Mixup / CutMix 74.0, stronger augmentation 74.3
- All tie with the slow-head run (74.1): the over-confidence is already handled
- Mixup cost 50% more training time for the same result

**Figure:** optional bar or dot chart of the four runs (values in `ABLATION_STORY.md` chapter 4).
**Notes (45 s):** Keep this short: four standard tricks, no gain. That tells us we reached the ceiling of this recipe at
224 px, which motivates the next slide.

## 10. How we test matters: larger test images

- Random training crops make objects look larger than the 224 px test crop
- Testing the same model at 288 px: 76.0% (+1.7), no retraining; 320 px +1.0, 352 px worse
- Mirror averaging adds +0.25, at twice the inference cost
- Approved by the professor: our final model is tested at 288 px

**Figure:** optional line of accuracy against test size (224, 256, 288, 320, 352), values in `EXPERIMENT_LOG.md` (run 22).
**Notes (45 s):** A free gain from fixing a train/test mismatch. Say that the professor approved it and that full
fine-tuning should be tested the same way for a fair comparison.

## 11. Results: our 5% model against full fine-tuning

- Accuracy: 74.3% against 76.3% (about 97% of full fine-tuning)
- Weights stored per task: 12.0M against 93.5M
- Latency: 11.8 ms against 6.9 ms (the adapters add layers); training 1.19 h against 1.90 h (48 GB-equivalent)

**Figure:** `figures/5_final_comparison.png`
**Notes (75 s):** Go through each metric the brief asks for. Be honest about the trade-off: smaller to store, cheaper
to train, but slower at inference and 2 points behind.

## 12. Is the comparison fair?

- Full fine-tuning trained 18.2 epochs (95% of the budget), our model 10 epochs (about 55%), so we checked both ways
- Equal length (10 epochs): 76.1% against 74.3%, gap 1.8 (`full_ft_llrd_10ep`)
- Equal budget (our model 16.9 epochs): 74.2% against 76.3%, gap 2.1 (`imp_trivialaug_fullbudget`)
- Both tested at 288 px: 76.0% against 77.7% (10 epochs), gap 1.7

**Figure:** the three-row check table (check, run, result).
**Notes (75 s):** Show that we checked our own claim. The gap stays about 2 points whether we equalise epochs or budget,
so training time is not the explanation; the method is. Full fine-tuning simply benefits from 450,000 images, as the
course's adaptation table predicts for large datasets.

## 13. Takeaways

- Train the head slowly: it is the largest and most over-confident trained part
- How many adapter weights you have matters more than where they go
- With 450,000 images, full fine-tuning still leads; 5% gets about 97% of it at a fraction of the storage

**Figure:** none.
**Notes (60 s):** One sentence per takeaway, then repeat the key message from slide 1.

## 14. Limitations and next steps

- One seed per run: differences under about 0.5 points are noise
- Hold-out is 2 images per class; the leaderboard is the independent check
- Next: training at the larger image size, longer training, other head designs (cosine head)

**Figure:** none.
**Notes (60 s):** End on what we would do with more time, then open for questions.
