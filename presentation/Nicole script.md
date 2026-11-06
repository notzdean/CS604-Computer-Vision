# Nicole script: slides 14 to 19

About 4 minutes in total. All scores are hold-out top-1, not test scores.

## Slide 14: Our best models (1:00)

Here are our two best models. We trained both at 288 pixels and tested them at 320 pixels.

Full fine-tuning reached 80.1 percent. Our 5 percent model reached 78.8 percent. The gap is only 1.4 points.

Both models took about 1.7 hours of training in budget terms.

But the 5 percent model is slower at test time. It needs 11.8 milliseconds per image, and full fine-tuning needs 7.3. The adapters add extra steps in every block, and AdaptFormer cannot be merged into the weights.

So fewer trained weights does not mean faster training or faster inference.

The box shows what went to the test set. The rules only allowed models from the validation phase, and run 42 came later. So we sent run 39 for full fine-tuning and run 30 for the 5 percent track. Run 30 uses the same recipe, but we trained it at 224 pixels and tested it at 288.

## Slide 15: Was it luck? A second seed says no (0:30)

Was our result just luck? To check, we trained both models again with a new random seed.

The 5 percent model moved by 0.09 points. Full fine-tuning moved by 0.25 points. Both changes are inside our 0.5-point tie zone.

Full fine-tuning stayed ahead with both seeds. Two seeds are not proof, but they show that our tie zone is safe.

## Slide 16: What didn't work, and what it taught us (0:50)

This table shows ten ideas that did not help. We compared each one with its own baseline.

The two blue rows are our own ideas.

First, we smoothed the labels only towards 20 look-alike classes. The loss went down, but accuracy fell by 0.61 points.

Second, we froze the head for the first epoch. The run fell far behind, caught up by epoch six, and ended in a tie.

Some ideas did not transfer. Layer-wise decay helps full fine-tuning, but it cost the adapters 1.5 points. Training for the whole budget did not help either. Averaging two models gave only 0.39 points for twice the cost.

Each failure ruled out one wrong idea, so it still taught us something.

## Slide 17: Five practical problems, and our fixes (0:30)

We also hit five practical problems.

A T4 GPU was too slow, so we ran everything on an A100. The budget check stopped runs too early, so we gave its speed test more steps.

The one worth naming is run 40. A second process wrote to the same folder, and the run stopped at 8.3 of 10 epochs. We re-ran it cleanly as run 41.

We also gave every run its own folder. And we dropped run 43, because training at 288 pixels beat it.

## Slide 18: If we did it again (0:40)

If we did it again, we would change four things.

First, we would test image size in week one. It was our biggest lever, but we found it late.

Second, we would then tune at 224 pixels. Those runs are faster, and the findings carried over.

Third, we would re-check the deciding comparisons at the final settings. Our design tie broke at 288 pixels.

Fourth, we would split the hold-out. One photo per class would choose the settings, and the other would report the score.

## Slide 19: In one line (0:30)

So here is our project in one line. The adapter design barely mattered. How we trained mattered, above all a slow, gentle head.

We tested 22 hypotheses. Eight were supported, eight were rejected, and three were split.

Thank you. We are happy to take your questions.
