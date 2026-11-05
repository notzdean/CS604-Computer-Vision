# How we took a 10,000-class classifier from 36% to 80%

**In one sentence:** we set out to find the best place for 5% of a model's parameters, and learned that accuracy depended far more on how gently the classifier learned and how many pixels the model saw.

All scores are top-1 accuracy on 20,000 labelled training images kept out of training (2 per class). Differences under about 0.5 points are treated as noise. Scores are at 224 px unless a size is given.

## The task

We adapt a pre-trained ViT-B/16 to 10,000 fine-grained classes, using 450,000 images (45 per class). Training is capped at 2 hours on a 48 GB GPU or the equivalent. There are two tracks: train the whole model, or freeze it and train under 5% of its parameters. The 7.7M-parameter classifier head does not count towards the 5%.

## Where the points came from

| Step | 5% track | Full fine-tuning |
|---|---|---|
| Starting point: head set from class averages, nothing trained | 35.7 | 35.7 |
| First trained model | 71.05 | 72.53 |
| Label smoothing, gentler learning rates | 74.14 | 76.06 |
| Testing on larger images | 75.99 (288 px) | 77.67 (288 px) |
| Training on larger images | 77.94 (288 px) | 79.69 (288 px) |
| Testing at 320 px: our best models | **78.75** | **80.14** |

From "Testing on larger images" on, the 5% track's scores come from a model with an added augmentation (see "What didn't help"), which itself made no measurable difference.

## 1. Where the 5% goes barely matters

Our first question was how to spend the 5%. We tried three designs at the same budget, for 10 epochs:

| Design | Accuracy |
|---|---|
| LoRA on attention | 70.76 |
| AdaptFormer on the feed-forward layers | 71.09 |
| Both combined | 71.05 |
| AdaptFormer in the last 6 blocks only | 70.17 |

The three designs that cover every block tie, and concentrating the adapters in the last blocks loses about a point. Size does help: a much smaller LoRA (0.34%) scored 68.78, although that run was also shorter (7 epochs).

So the design choice was not what was holding accuracy back.

## 2. The model was memorising

The training curves pointed to the real limit. By the end of training, the 5% model's training loss was 0.07 while its loss on held-out images was 1.42. It had nearly memorised its 45 images per class.

Two changes that target this helped straight away, each adding over a point:

- **Label smoothing 0.1:** 72.17 (+1.12)
- **A lower learning rate, 2e-4:** 72.37 (+1.32), while a higher one, 5e-4, fell to 69.28

Together they gave 72.35, no more than either alone, which suggests they treat the same problem. With label smoothing, lowering the rate to 1e-4 reached 73.02.

## 3. The head needs to learn slowly

The head has more trained parameters than all the adapters combined (7.7M against 4.3M), so we gave it its own learning rate, keeping the adapters at 1e-4:

| Head learning rate | Accuracy | Held-out loss |
|---|---|---|
| Double (2e-4) | 71.75 | 1.570 |
| Same (1e-4) | 73.02 | 1.420 |
| Half (5e-5) | **74.14** | 1.325 |
| Quarter (2.5e-5) | 74.02 | 1.335 |

This was the most sensitive setting we found on the 5% track, a 2.4-point swing.

A second result pointed the same way. A better starting head should help, but it hurt all three times we tried it:

| Head start | Accuracy | Against the same run without it |
|---|---|---|
| Class averages from 43 images instead of 10 | 69.98 | -1.07 |
| The same, on the tuned recipe | 71.99 | -1.03 |
| A head already trained on frozen features | 70.28 | -4.00 |

Our reading, which we did not test directly, is that a head already fitted to the training images leaves less for the adapters to learn from and overfits sooner. In the worst case, training loss ended lower (1.90 against 2.04) while held-out loss ended much higher (1.54 against 1.31).

The head still matters most. Training it alone, with the backbone frozen, takes accuracy from 35.7 to 62.62. That's about 70% of the gain to 73.02, and the adapters supply the other 30%.

## 4. What didn't help

Once the head was slow, the usual tools stopped paying off. Each was added to the 74.14 model:

| Addition | Accuracy |
|---|---|
| Stochastic depth 0.1 | 74.09 |
| Weight averaging (EMA) | 74.19 |
| Mixup and CutMix, 15 epochs | 73.97 |
| TrivialAugment and random erasing | 74.28 |

All four are ties. Neither was time the limit: using the whole budget changed the score by -0.10 on the 5% track (16.9 epochs) and +0.20 on full fine-tuning (18.2 epochs). Averaging two full fine-tuning models added only 0.39, at twice the inference cost.

By this point, the recipe had converged at 224 px.

## 5. The biggest gain was more pixels

Random training crops make objects look larger than they do in a centred test crop. Testing on larger images corrects for that, with no retraining:

| Model | 224 px | 288 px | Gain |
|---|---|---|---|
| Best 5% model | 74.28 | 75.99 | +1.71 |
| Full fine-tuning | 76.06 | 77.67 | +1.61 |

Training was the next step. Because 10 epochs were already enough, the rest of the budget could pay for more pixels. Fine-grained classes differ in small details, so we trained full fine-tuning at 288 px, changing nothing else:

| Test size | Accuracy |
|---|---|
| 288 px | 79.69 |
| 320 px | 80.14 |
| **352 px** | **80.45** |
| 384 px | 79.91 |
| 416 px | 79.06 |

Training at 288 px added 2.02 points over only testing at that size, and the best test size added about 0.8 more. It took 2.00 h on a 40 GB GPU, or 1.67 h of the 2 h budget.

## 6. What 5% costs, and what it saves

At the same training size, full fine-tuning leads by about 2 points. The gap holds at equal epochs and at equal budget, so it comes from the method rather than from training length.

| Tuned, 224 px training | 5% | Full fine-tuning |
|---|---|---|
| Accuracy, 10 epochs, 224 px test | 74.28 | 76.06 |
| Accuracy, 10 epochs, 288 px test | 75.99 | 77.67 |
| Accuracy, whole budget, 288 px test | 75.99 | 77.89 |
| Parameters stored per task, including the head | 12M | 93.5M |
| Latency at batch 1 | 11.8 ms | 7.1 ms |

- **What 5% saves:** storage. Each new task needs about an eighth of the parameters.
- **What 5% doesn't save:** training time. On the same 80 GB GPU and at 10 epochs, the untuned 5% design took 1.18 h and tuned full fine-tuning took 1.20 h.
- **What it costs:** about 2 points and about 65% more inference latency, because the adapters add layers.

Trained at 288 px as well, the 5% design gains as much as full fine-tuning (+1.95), so the gap stays: 1.75 points at 288 px (77.94 against 79.69) and 1.39 at the chosen 320 px (78.75 against 80.14). A second seed moves each track by 0.25 points or less, so the gap is real.

## What we would say we learned

1. On this task, the amount and spread of adapters mattered more than their type.
2. The classifier head is a design decision in its own right: it carries most of the gain and overfits first.
3. A better start is not always better: all three stronger head starts ended lower, though why is our interpretation.
4. When regularisers and extra epochs stop helping, look at the input. Resolution beat everything else at equal epochs.

## Still open

- **Neighbour label smoothing (run 53):** smoothing towards each class's 20 most similar classes instead of all 10,000 scored 73.67 against 74.28 (-0.61): better calibrated (lower hold-out loss), but less often right. Uniform smoothing stays.
- **Settled since this was first written:** the 5% design trained at 288 px (77.94, run 42); the best models are runs 39 and 42, tested at 320 px; the test submissions were run 39 and, because only models submitted in Phase 1 qualified, run 30 (the 224 px recipe on all 45 images per class, tested at 288 px); the 16.1 ms latency reading at 352 px was a disturbance (run 46: about 7 ms for full fine-tuning at 320 to 352 px).
