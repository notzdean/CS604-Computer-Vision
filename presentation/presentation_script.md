# Presentation script: 10,000 classes, 45 photos each

About 14 minutes in total. Speaker 1 takes slides 1 to 6, Speaker 2 slides 7 to 13, and Nicole slides 14 to 19.

All scores come from the photos we kept aside (2 per class), not from the test set. The exact numbers are on the slides, so the script rounds them.

| Slides | Speaker | Time |
|---|---|---|
| 1 to 6 | Speaker 1 | about 4:00 |
| 7 to 13 | Speaker 2 | about 6:00 |
| 14 to 19 | Nicole | about 4:00 |

---

## Slide 1: 10,000 classes, 45 photos each (0:30) · Speaker 1

Imagine teaching a model to recognise 10,000 kinds of objects, with only 45 photos of each. On top of that, in one of our two tracks, we could train less than 5 percent of the model.

Before any training, our model scored about 36 percent. By the end, our 5 percent model reached almost 79 percent, and full fine-tuning reached about 80.

In the next few minutes, we will show how we got there, including the ideas that did not work.

## Slide 2: Frozen ViT-B/16, adapters in every block (0:45) · Speaker 1

We started from a ViT-B/16, a model that had already learned from millions of images. It cuts each photo into small patches and passes them through 12 blocks.

In our 5 percent track, we kept all 12 blocks frozen and added two small trainable parts to each one. LoRA sits beside the attention layer, while AdaptFormer sits beside the feed-forward layer.

Together, they add about four million weights, which is just under 5 percent of the model. We also trained a new final layer for the 10,000 classes, but the rules do not count it in the 5 percent.

Our main question was whether using both adapters really beats using just one.

## Slide 3: Every experiment had to fit these rules (0:30) · Speaker 1

To keep our comparisons fair, every experiment followed the same rules: the same starting model, the same 45 photos per class, and the same compute budget.

Since the validation and test labels were hidden, we kept 2 photos per class aside, which gave us 20,000 photos to score every run. And whenever two results were less than half a point apart, we treated them as equal.

## Slide 4: 96 GB-hours per run, and the brief's reference results (0:20) · Speaker 1

The budget was 96 GPU-memory hours per run, which on our A100 means about two and a half hours. The brief also gave some reference results from a PhD student, which we used as a guide, not as a target.

## Slide 5: Each step is one idea we kept (0:45) · Speaker 1

This table tells our whole story in one place. Each row is one idea we kept, and together they lifted the 5 percent model from about 69 to almost 79 percent, and full fine-tuning from about 73 to 80.

What matters most is where these gains came from. Using the full 5 percent helped a little, but the bigger steps came from how we trained the model, and then from larger images. Since each step builds on the one before, the gains do not simply add up, which is why we also tested each idea on its own.

Throughout all of this, the gap between the two tracks never closed.

## Slide 6: Where the adapters go matters; which adapter does not (1:00) · Speaker 1

Our first question was about the design itself.

First, size: with the same training settings, the small LoRA model scored about one point below the designs that use the full 5 percent. Our very first comparison made this gap look twice as big, because the small model had also trained for less time.

Next, we compared three designs of the same size: LoRA alone, AdaptFormer alone, and both together. All three scored about 71 percent, within half a point of each other, so for this task, the choice of adapter did not matter.

What did matter was where we put them. When we placed the adapters only in the last six blocks, the score dropped by almost a point, so it helps to adapt every block.

We therefore kept the combined design for now, but planned to check later whether a simpler one could do just as well.

## Slide 7: Most of the adaptation is cheap (0:50) · Speaker 2

Next, we asked whether we even needed the full 5 percent.

Training only the final layer gave about 63 percent. BitFit, which trains only the small bias values and touches far less than one percent of the model, reached about 71. Our 5 percent adapters reached about 74, and full fine-tuning about 76.

In other words, the biases alone gave about three quarters of what our adapters added, with 40 times fewer weights, and they do not slow the model down at all. So most of the adaptation is cheap, and the full 5 percent only buys the last few points.

## Slide 8: The model was too sure of itself (1:00) · Speaker 2

Next, we noticed that the model was too sure of itself. It learned the training photos almost perfectly, but it made many more mistakes on the photos we kept aside.

To fix this, we tried two things, one at a time. Label smoothing, which stops the model from being completely certain about any answer, added about one point. A lower learning rate also added about one point, while a higher one cost almost two.

However, when we combined the two, we gained nothing extra, because they fix the same problem. Only when we lowered the rate further on top of smoothing did the score rise again, and that became our recipe.

Lastly, we tried our own idea: smoothing only towards 20 look-alike classes. The model's error went down, but it picked the correct class less often, so we did not keep it.

## Slide 9: The head should learn slowly (0:50) · Speaker 2

We then looked at the final layer, which we call the head. Since it has more weights than all our adapters together, we gave it its own learning rate.

When the head learned twice as fast as the adapters, the score dropped by more than a point. When it learned at half the speed, the score rose by about a point, and a quarter of the speed gave the same benefit. So we chose half, and this turned out to be our most sensitive setting.

The bar at the bottom shows where our score comes from. Training the head took us from about 36 to 63 percent, and the adapters then added another 10 points on top.

## Slide 10: A better head start ends lower, and our first explanation was wrong (1:00) · Speaker 2

This result surprised us. We expected a better starting point for the head to give a better final score, but it did the opposite. Building the head from 43 photos per class instead of 10 cost about a point, and starting from a separately trained head cost four points.

Our first guess was that a strong head learns too fast at the start and leaves too little for the adapters. To test this, we paused the head's training for the first round, so the adapters could learn first. The model fell far behind at first, caught up by the sixth round, and ended with the same score.

So our guess was wrong: when the head starts learning does not matter, only how fast it learns. We still do not know the real cause, but the lesson is clear: a good explanation is not proof until we test it.

## Slide 11: After that, nothing moved the 5% recipe (0:40) · Speaker 2

After that, nothing else moved the 5 percent model.

We tried four ways to stop the model from memorising the photos: stochastic depth, weight averaging, Mixup with CutMix, and stronger augmentation. None of them changed the score by more than a fifth of a point, and training for the whole budget did not help either.

One idea even hurt: layer-wise decay, a learning-rate trick that helps full fine-tuning, cost our adapters one and a half points.

At this point, our training recipe had stopped improving, so instead we changed what the model sees.

## Slide 12: More pixels lifted both tracks equally (1:00) · Speaker 2

Our next idea was to give the model larger images. Since ViT cuts each photo into patches of 16 pixels, a larger image gives more patches and keeps more fine detail: 196 patches at 224 pixels, but 324 at 288.

First, we simply tested our trained models on larger images, without any retraining, and both tracks gained almost two points for free. Then we trained on larger images, which added about two more points on both tracks, our largest single gain. Testing at 320 pixels then added a little more.

However, the gap between the two tracks stayed between one and a half and two points at every size. So larger images help both methods equally, but they do not close the gap, and they make training slower.

## Slide 13: The tie breaks at higher resolution (0:40) · Speaker 2

This brings us back to our design question. On small images, AdaptFormer alone scored the same as our combined design and ran about 20 percent faster. Since our rule is to pick the cheapest option among equal results, we nearly switched to AdaptFormer alone.

But when we trained both on larger images, AdaptFormer alone fell almost a full point behind, so we kept both adapters. The lesson is simple: a comparison only counts at the settings we finally use.

---

## Slide 14: Our best models (0:55) · Nicole

Here are our two best models, both trained on larger images.

Full fine-tuning reached about 80 percent, while our 5 percent model came in just under 79, so the gap is less than one and a half points. Both took under two hours to train.

However, the 5 percent model is slower when it predicts, at almost 12 milliseconds per image against about 7, because its adapters add extra steps in every block. So we must distinguish parameter efficiency from training efficiency and inference efficiency. Here, fewer weights did not make training faster, and it made predictions slower.

For the test set, the rules only allowed models from the validation phase, and our best 5 percent model came later. So we submitted our full fine-tuning model and an earlier 5 percent model, trained on smaller images.

## Slide 15: Was it luck? A second seed says no (0:20) · Nicole

To check that this was not luck, we trained both models again from a different random start. Both scores moved by less than a quarter of a point, and full fine-tuning stayed ahead both times.

## Slide 16: What didn't work, and what it taught us (0:45) · Nicole

However, not every idea worked, and this table shows ten ideas that did not work for us.

The two blue rows are our own ideas. First, we told the model that confusing look-alike classes was a smaller mistake; its error went down, but it picked the correct class less often. Second, we paused the training of the final layer at the start; the model fell behind, caught up later, and ended with the same score.

Other ideas did not carry over. One learning-rate trick helps full fine-tuning but hurt our adapters, and combining two models doubled the cost for almost no gain at all.

## Slide 17: Five practical problems, and our fixes (0:35) · Nicole

We also ran into five practical problems.

First, the T4 GPU was too slow, so we moved to an A100. Second, our time check stopped some runs too early, so we made it measure the speed for longer. Third, two copies of one run wrote to the same folder and cut it short, so we ran it again cleanly. Fourth, reusing a folder name loaded an old model instead of training a new one, so now every run gets its own folder. Finally, we dropped one planned model, because training on larger images had already beaten it.

## Slide 18: If we did it again (0:35) · Nicole

Looking back, we would do four things differently.

First, we would try different image sizes at the very start, because larger images gave us our biggest gain, but we only found that late. Then we would do most of the testing on small images, since those runs are faster and the results carried over.

We would also repeat key comparisons with the final settings, because two designs that looked equal on small images were not equal on larger ones. Finally, we would split the photos we kept aside: one half to choose settings, the other half to give the final score.

## Slide 19: In one line (0:30) · Nicole

So, if we had to sum up our project in one line, it would be this: the adapter design barely mattered, but how we trained did, above all a slow, gentle head, and after that, larger images.

Across the project, we tested 22 ideas. Eight were supported, eight were rejected, and three were split, with mixed results.

Thank you, and we are happy to take your questions.
