# CS604 project requirements

Collected from the project brief (`CS604 project description.pdf`, 3 pages; kept out of git by `.gitignore`), the professor's later announcements and rulings. The CodaBench link is left out on purpose: it contains a secret key.

Sources: **[brief p.N]** = the PDF, page N; **[announcement]** = a later announcement or ruling by the professor, not in the PDF. Quotes in "double quotes" are the brief's own words. The brief says nothing about the input image size.

## 1. Task [brief p.1]

- Fine-grained multi-class image classification, 10,000 classes, by adapting a pre-trained model (SFT, LoRA, Adapter, AdaptFormer).
- Evaluation metrics: top-1 accuracy, trainable parameters, inference latency, training GPU-hours.

## 2. Base model [brief p.1]

- ViT-B/16, IN21K pre-trained then fine-tuned on IN1K.
  - timm: the brief writes "timm.models.vit_base_patch16_224 (pretrained=True)" and links https://huggingface.co/timm/vit_base_patch16_224.augreg2_in21k_ft_in1k, i.e. `vit_base_patch16_224.augreg2_in21k_ft_in1k`
  - Hugging Face: `google/vit-base-patch16-224`, https://huggingface.co/google/vit-base-patch16-224
- The brief's Runtime Constraints list names the base model as "vit-base-patch16-224-in21k".
- **Professor's guidance [announcement]:** follow the recommended backbone as closely as possible. Do not switch unless there is a very strong, well-justified reason; if you do, explain in the report and presentation why it was necessary and how it addresses the problem.
- **Ruling (12 Oct 2026) [announcement]:** a different base checkpoint is not allowed; stay on `vit_base_patch16_224.augreg2_in21k_ft_in1k`.

## 3. Dataset [brief p.1]

- Hugging Face dataset `doem1997/cs604-course-data` (download manually). Anonymous dataset.
- Train: 450,000 images, 45 per class. Labels in `train.txt` (`image_path class_id`).
- Validation: 50,000 images, 5 per class (labels withheld).
- Test: 100,000 images, released in Phase 2 (labels withheld). "Submit your predictions to CodaBench to get results."
- Images in `train/`, `val/`, `test/`.

## 4. Compute constraint [brief p.1 and p.2]

- Training GPU-hours: **2 hours x 48 GB VRAM**, or equivalent (e.g. 4 h x 24 GB, or 2 h on 2 x 24 GB RTX 3090/4090). Tolerance **±10%** ("Runtime Constraints (±10%)").
- Stated separately for both tracks (p.2). It limits training only; the brief sets no limit on inference.

## 5. Tasks [brief p.2]

1. **Full fine-tuning** within the compute constraint.
2. **Efficient model adaptation:** freeze the pre-trained model and train only a small set of parameters (e.g. <5%) while improving performance. The brief notes it is expected to reduce catastrophic forgetting and to outperform full fine-tuning, especially with small datasets and few-shot situations. Possible directions:
   1. Selected layer fine-tuning
   2. LoRA: adds low-rank matrices to frozen weights
   3. Adapter: inserts small bottleneck layers
   4. AdaptFormer: inserts small bottleneck layers into the FFN
   5. Prefix tuning: adds a trainable continuous prefix to the input
   6. BitFit: trains only the bias parameters

- **Parameter counting ruling (11 Oct 2026) [announcement]:** the classifier head (768 x 10,000) is not counted; trainable % = adapter params / backbone params. Report the head's count separately and state how it is counted.

## 6. Evaluation metric [brief p.2]

The brief's heading is "Evaluation metric". It defines the metrics; what the report must contain is in section 12.

- "Top-1 Classification Average Accuracy (auto-calculated by CodaBench)": the leaderboard computes it from the submitted predictions. The brief does not require the report's accuracy numbers to come from CodaBench; section 12 asks only for "Top-1 average accuracy".
- "Trainable Parameters (e.g., LoRA is 0.34% of foundation model), please specify in your report."
- "Inference Latency (please specify in your report)."
- "Training time and VRAMs (please provide training log in your report)." The training log is asked for here, for training time and VRAM.

## 7. Reference results (from a PhD student) [brief p.2]

Neither a lower nor an upper bound; 90%+ is not expected.

| Method | Top-1 (%) | Trainable params (%) |
|---|---|---|
| Zero-shot with initialised classifier head | 43.7 | 0 |
| Full fine-tuning | 70.8 | 100 |
| LoRA | 68.3 | 0.34 |

## 8. Suggested things to look into [brief p.2; last point p.3]

- Optimisation: bf16 precision, gradient accumulation.
- Initialisation: how to initialise the classifier head and the PEFT adapters.
- Loss design: which loss to use.
- You may start from a decent open-source baseline and improve on it.

## 9. Focus and grading emphasis [announcement; not in the brief]

- **Focus on adaptation methods.** This is the main room for exploration and contribution: investigate, compare and refine different adaptation strategies, with real experimentation, analysis and deeper thinking.
- **The leaderboard is not the only criterion.** It is a reference; significant weight goes to the presentation and report, especially whether they show:
  - how the team analysed the problem;
  - how you interpreted your results;
  - how you iterated through experiments;
  - how you identified and tested new solution ideas;
  - what novelty and new thinking emerged, even if the final result was not the best.
- Document the reasoning, the experiments **including failed ones**, and the lessons learned. A clear, thoughtful process matters more than a higher leaderboard score.

## 10. Submissions [brief p.3]

- File names: `val_team_xx.txt` and `test_team_xx.txt`, one `image_path class_id` per line (e.g. `test/image_000025.jpg 7919`).
- Zipping the file for upload is not in the brief; it is what the CodaBench upload takes (`common/submit.py` writes the zip).
- **Phase 1, validation:** leaderboard open 26 Sep 2026 00:01 to 30 Oct 2026 23:59, max 5 submissions per day.
- **Phase 2, test:** test data released 31 Oct 2026 00:01; due **6 Nov 2026 23:59**; **2 submissions in total**.

## 11. Presentation [brief p.3]

- **7 Nov 2026, 12:00** (usual class hours). 15 min presentation + 5 min Q&A per group; order set nearer the date.
- Slides with figures, diagrams and charts for visualisation.
- **Professor's addition [announcement]:** prepare visually rich slides, using images, diagrams and experimental results to clearly explain the problem, methods and findings.
- Cover: your investigation, your method's motivation, design, comparison with baselines, and ablation studies.

## 12. Final report [brief p.3]

- Due **18 Nov 2026, 23:59**, submitted on eLearn (download it afterwards and check).
- Format: CVPR conference paper LaTeX template, https://github.com/cvpr-org/author-kit/releases/tag/CVPR2026-v1
- Marked on:
  - motivation / justification of the design (**important**);
  - what the design is (a framework diagram of the architecture is recommended);
  - ablation study on the different parts of the design;
  - results: top-1 average accuracy, trainable parameter count, inference latency, training GPU-hours (implementation details), loss curves during training.
