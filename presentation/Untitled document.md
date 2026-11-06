Script

## **Slide 1 — 10,000 classes, 45 photos each (0:45)**

Imagine training a classifier to recognise 10,000 fine-grained categories, with only 45 photographs per class.

Now add a constraint: for one track, we can train less than 5% of the backbone's parameters.

Our question was not simply how to maximise accuracy. It was **where should we spend that limited adaptation capacity, and which design choices actually improve performance?**

We started at 35.7% accuracy with a classifier initialised from class averages.

Our best parameter-efficient model reached 78.8%, compared with 80.1% for full fine-tuning.

These are internal hold-out results, not hidden-test scores.

Today, I'll show how we reached those numbers, including the experiments that challenged our assumptions and what we learned from them.

## **Slide 2 — Frozen ViT-B/16, adapters in every block (0:45)**

We used a pre-trained ViT-B/16, with 12 transformer blocks.

For the efficient track, we froze the backbone and introduced two trainable components in every block.

**LoRA**, with rank 12, adapts the attention projections through low-rank updates.

**AdaptFormer**, with a bottleneck of 208, provides a small trainable pathway alongside the feed-forward network.

The design hypothesis was that adapting attention and feature transformations might offer complementary benefits.

Together, they use 4.29 million parameters, or 4.99% of the backbone.

The new 10,000-class classification head is trained separately and excluded from that limit.

The key question was whether using both adaptation mechanisms was genuinely better than using one.

## **Slide 3 — Every experiment had to fit these rules (0:20)**

We kept the same pre-trained checkpoint, dataset and compute budget across experiments.

Because official validation labels were hidden, we reserved two images per class, giving us a 20,000-image hold-out.

We evaluated top-1 accuracy and treated differences within 0.5 percentage points as practical ties.

That threshold guided engineering decisions; it was not a statistical significance test.

## **Slide 4 — 96 GB-hours per run (0:15)**

Our compute allowance was 96 GPU GB-hours per run, equivalent to 2.4 hours on our A100 40 GB.

Rather than exhaust the budget on every configuration, we prioritised controlled comparisons that could help us decide what to test next.

## **Slide 5 — Each step is one idea we kept (1:10)**

This slide summarises our investigation.

The efficient track improved from 68.78% to 78.75%, while full fine-tuning improved from 72.53% to 80.14%.

But what matters is not just the improvement. It's **where the points came from**.

We investigated three groups of decisions.

First, architecture: should we use LoRA, AdaptFormer or both, and where should we place the trainable parameters?

Second, optimisation: how should we set the learning rates, regularisation and training schedule?

Third, resolution: how much additional information could we recover by training or evaluating at a larger image size?

We started by spending nearly the full 5% parameter allowance. That helped, but architecture alone did not explain most of our eventual gains.

Optimisation changes produced substantial improvements, and resolution provided the final major lift.

These results are a sequence of selected configurations, not independent additive contributions. That's why we also ran controlled ablations.

Let's look at what those revealed.

## **Slide 6 — Where the adapters go matters; which adapter does not (1:05)**

Our first hypothesis was that combining LoRA and AdaptFormer would outperform either method independently.

To investigate, we compared the designs under equivalent training settings and similar parameter budgets.

The results were surprisingly close.

LoRA alone achieved 70.76%, AdaptFormer alone achieved 71.09%, and the combined design achieved 71.05%.

All three fell within our 0.5-point tie zone.

So our initial hypothesis was not supported at 224 pixels. Adding both mechanisms did not provide a measurable practical advantage.

We also tested another hypothesis: would concentrating trainable parameters in the last six transformer blocks be more effective?

That configuration fell to 70.17%.

One possible interpretation is that earlier transformer blocks also benefit from adaptation, especially for fine-grained recognition.

But the experiment did not isolate which layers were responsible.

**Our decision:** retain distributed adaptation and provisionally keep the combined architecture, while checking whether a simpler design could match it under the final training configuration.

This is an important distinction: we did not conclude that combining adapters was inherently superior.

## **Slide 7 — Most of the adaptation is cheap (0:40)**

We then asked whether we even needed the full 5%.

A linear probe, training only the classifier head, achieved 62.62%.

BitFit, which trains just 0.12% of the backbone through its biases, reached 71.30%.

Our 5% adapters reached 74.28%, while full fine-tuning achieved 76.06%.

Interestingly, BitFit captured approximately 74% of the improvement that the adapters delivered over the head-only baseline, using around 40 times fewer trainable backbone parameters.

It also introduced no additional inference modules.

The learning was that much of the benefit came from relatively small changes to the pre-trained representations.

That led us to investigate whether optimisation was more important than adding parameters.

## **Slide 8 — The model was too sure of itself (1:15)**

With 10,000 classes and relatively few examples per class, we were concerned that the model might become overconfident.

We investigated label smoothing and learning rates, changing one setting at a time against specified reference runs.

Starting from 71.05%, introducing label smoothing improved accuracy by 1.12 points.

Lowering the learning rate from 3 × 10⁻⁴ to 2 × 10⁻⁴ improved it by 1.32 points.

Increasing the learning rate to 5 × 10⁻⁴ made performance worse by 1.77 points.

This suggested that less aggressive parameter updates were beneficial.

However, combining label smoothing with the lower learning rate produced essentially no additional gain.

That was our first important learning: **two individually successful techniques do not necessarily produce an additive improvement**.

Hyperparameters interact, so we cannot simply combine every successful experiment and expect their gains to accumulate.

We then tested our own idea: instead of smoothing labels uniformly, could we smooth towards 20 visually similar classes?

The intuition was that confusing similar classes should be penalised less severely.

Interestingly, hold-out loss improved from 1.311 to 1.222, but top-1 accuracy fell by 0.61 points.

So we rejected it for our primary objective.

This demonstrated that optimising loss and maximising top-1 accuracy are not always equivalent. A lower loss alone also does not prove better calibration.

## **Slide 9 — The head should learn slowly (1:00)**

Next, we investigated the learning rate of the classifier head.

Our hypothesis was that the randomly initialised 10,000-class head might learn too aggressively relative to the adapters, potentially destabilising the joint optimisation.

We held the other settings constant and varied only the head learning-rate multiplier.

Doubling the head rate reduced accuracy by 1.27 points.

Halving it improved accuracy by 1.12 points.

Reducing it to a quarter also helped, but offered no additional practical benefit over halving it.

We therefore selected a multiplier of 0.5.

The lower diagram also shows how much accuracy improvement came from training the classifier before adding backbone adaptation.

The first major gain was learning the classification boundary; adaptation then provided an additional improvement.

The design lesson was that different trainable components do not necessarily need the same learning rate.

We observed that a slower head worked better. But the precise explanation remained a hypothesis, which motivated our next experiment.

## **Slide 10 — A better head start ends lower, and our first explanation was wrong (1:00)**

This was one of our most interesting negative results.

We initially expected a better classifier-head initialisation to improve final accuracy.

Instead, using 43 rather than 10 photographs per class for initialisation reduced accuracy by approximately one point.

Initialising from a separately trained linear probe performed even worse, reducing accuracy by four points.

Our first explanation was that a stronger initial head might interfere with early adapter learning.

But that was only a hypothesis.

To investigate, we froze the classifier head for the first training epoch, allowing the adapters to adjust first.

After epoch one, accuracy was 12.7 points behind.

By epoch six, the model had caught up. By the end, the difference was only \+0.05 points—effectively a tie.

So delaying head training did not improve final performance.

We therefore rejected our proposed remedy, but we could not establish the actual mechanism behind the weaker final result from stronger head initialisation.

**The learning here was about research rigour:** a plausible explanation is not evidence. We need experiments that can challenge it, and we must be willing to revise our interpretation when the results disagree.

## **Slide 11 — After that, nothing moved the 5% recipe (0:40)**

We investigated several additional techniques.

Stochastic depth, exponential moving averages, Mixup, CutMix and stronger augmentation all produced results within our practical tie zone.

Training for the full budget—about 16.9 epochs—also failed to improve accuracy.

One interesting contrast was layer-wise learning-rate decay.

It improved full fine-tuning by 0.51 points, but reduced adapter performance by 1.50 points.

So a technique that benefits full fine-tuning does not necessarily transfer to parameter-efficient adaptation.

Our decision was to stop adding complexity without demonstrated benefit.

At this stage, the optimisation gains were diminishing.

We therefore shifted our attention from how the model learned to **how much visual information it received**.

## **Slide 12 — More pixels lifted both tracks equally (1:20)**

Our next hypothesis was that image resolution could be limiting fine-grained recognition.

A ViT-B/16 processes 16-by-16-pixel patches.

At 224 pixels, we have 196 patches. At 288 pixels, that increases to 324, and at 320 pixels, to 400\.

Higher resolution may preserve fine visual details that disappear when an image is downsampled.

We tested two different interventions.

First, we increased inference resolution without retraining.

For the efficient model, accuracy improved from 74.28% at 224 pixels to 75.99% at 288 pixels—a gain of 1.71 points.

This mattered because it required no additional training.

Second, we trained at 288 pixels.

Accuracy increased further to 77.94%.

Finally, evaluating that model at 320 pixels raised accuracy to 78.75%.

Full fine-tuning showed almost the same pattern, finishing at 80.14%.

These gains were larger than most of our architectural changes.

But higher resolution has a cost. More patches mean greater attention computation and potentially higher inference latency and memory usage.

So the decision is not simply to maximise resolution indefinitely. We must consider the accuracy–compute trade-off.

**Our main learning:** in fine-grained recognition, preserving visual information can matter more than introducing additional model complexity.

Notice also that full fine-tuning remained ahead by roughly 1.4 to 1.8 points throughout.

Resolution improved both tracks; it did not eliminate the gap between them.

## **Slide 13 — The tie breaks at higher resolution (0:50)**

Remember our earlier architectural comparison.

At 224 pixels, AdaptFormer alone and the combined design were practically tied.

AdaptFormer alone was also about 20% faster at inference—9.5 versus 11.8 milliseconds.

Our decision rule would therefore favour the simpler model.

But before finalising the architecture, we repeated the comparison at 288-pixel training resolution.

This time, AdaptFormer alone scored 0.87 points below the combined model, outside our tie zone.

So we retained LoRA plus AdaptFormer for our strongest efficient configuration.

This changed our earlier interpretation.

The combined architecture offered little practical benefit at 224 pixels, but a more meaningful advantage after high-resolution training.

The research lesson was that **architectural rankings are conditional on the training regime**.

A fair final comparison needs to be made under the settings we actually intend to use.

## **Slide 14 — Our best models (0:40)**

Our strongest internal hold-out results were 80.14% for full fine-tuning and 78.75% for the efficient model.

That is a gap of just 1.39 percentage points.

Both required about 1.7 budget-normalised training hours.

However, despite training fewer backbone parameters, our efficient model was slower at inference: 11.8 milliseconds versus 7.3 milliseconds.

The extra adapter pathways introduced computation.

So we must distinguish **parameter efficiency from training efficiency and inference efficiency**.

There is also an evaluation qualification.

Our best efficient model, run 42, was developed after the test-submission window. The model submitted to the hidden test was the earlier run 30\.

Therefore, our 78.75% result is an internal hold-out result, not a hidden-test score.

## **Slide 15 — Was it luck? A second seed says no (0:45)**

We wanted to check whether our results were unusually favourable because of random initialisation.

So we repeated both tracks with a second random seed, at 288-pixel training and evaluation resolution.

The efficient model changed from 77.94% to 77.85%—only 0.09 points.

Full fine-tuning changed from 79.69% to 79.44%, a difference of 0.25 points.

Both differences were within our practical tie zone, and full fine-tuning remained ahead.

This supports the consistency of our observed ranking across those two seeds.

However, two seeds are insufficient for a reliable estimate of variance or a statistical significance claim.

A stronger study would use more seeds, report confidence intervals and evaluate whether improvements are consistent across classes.

So this was a useful robustness check, but not definitive proof.

## **Slide 16 — What didn't work, and what it taught us (0:45)**

We tested 22 hypotheses, consolidated into 19 verdicts: eight supported, eight rejected and three with split results.

Several negative findings changed our design decisions.

Concentrating adaptation in the last six blocks was worse.

Custom neighbour-based label smoothing reduced top-1 accuracy.

Longer training did not improve results.

And averaging two full fine-tuned models improved accuracy by only 0.39 points—within our tie zone—while roughly doubling inference cost.

These failures were not wasted experiments.

They helped us eliminate unsupported assumptions and avoid unnecessary complexity.

The important distinction is between an experiment that fails to improve accuracy and an experiment that fails to teach us anything.

A well-controlled negative result can still be valuable evidence.

## **Slide 17 — Five practical problems, and our fixes (0:15)**

We also encountered runtime and experiment-management issues.

We standardised on the A100, corrected premature budget termination and isolated run directories to prevent checkpoint reuse or concurrent writes.

These fixes mattered because experiments are only comparable when they execute as intended.

## **Slide 18 — If we did it again (1:00)**

Looking back, we would improve our experimental strategy in four ways.

**First, investigate resolution earlier.**

It was one of our most effective levers, yet we discovered it relatively late.

A quick resolution sweep could have identified its importance before spending substantial compute on architectural variations.

**Second, use lower-resolution experiments for initial screening.**

At 224 pixels, experiments are faster, and many optimisation findings carried over to 288 pixels.

We could shortlist promising methods cheaply, then evaluate them at higher resolution.

**Third, repeat the decisive comparisons under final conditions.**

The adapter comparison is a good example: a practical tie at 224 pixels became a 0.87-point difference at 288 pixels.

This means we should not assume that an earlier architectural ranking remains valid after changing the training regime.

**Fourth, strengthen the evaluation protocol.**

We repeatedly used the same two-image-per-class hold-out to select settings.

That creates a risk of optimistic selection bias.

A stronger design would reserve an untouched evaluation subset, perform additional seed replications and examine class-level errors.

The overarching lesson is that experimental rigour should be designed into the workflow from the beginning, not added only after finding a good model.

## **Slide 19 — In one line (0:30)**

To conclude, we reached 78.8% internal hold-out accuracy by adapting less than 5% of a pre-trained ViT backbone, within 1.4 points of full fine-tuning.

But our most important finding was not that a particular adapter architecture won.

It was that optimisation choices and image resolution were major drivers of performance.

Architecture still mattered, particularly under the final high-resolution configuration.

**Our central learning is that a good model is not just about where we add trainable parameters. It's about how we train, how we evaluate, and whether our experiments genuinely support our conclusions.**

Thank you.

