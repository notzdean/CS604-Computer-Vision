# CS604 project: adapting ViT-B/16 to a 10,000-class dataset

Team project: full fine-tuning and parameter-efficient adaptation (LoRA, AdaptFormer, BitFit, ...) of a pre-trained
ViT-B/16, under a 2 h x 48 GB budget per run. Everything runs on Google Colab GPUs from VS Code (or the Colab web UI).

The project is finished (18 Nov 2026). The final results are summarised under [Results](#results) below. Every run, the
24 hypotheses it tested and what each result means are in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

## Results

Hold-out top-1 on 20,000 labelled training images kept out of training (2 per class). Differences under about 0.5 points
are treated as ties. Trainable % counts the backbone only. The 10,000-class head (7.69M) is trained in every run but not counted.

| Model | Trained | Train / test px | Hold-out top-1 |
| --- | --- | --- | --- |
| Class-mean head, untrained | 0% | 224 / 224 | 35.70 |
| Linear probe (head only) | 0% | 224 / 224 | 62.62 |
| BitFit | 0.12% | 224 / 224 | 71.30 |
| LoRA + AdaptFormer, tuned (`imp_trivialaug`) | 4.998% | 224 / 224 | 74.28 |
| Full fine-tuning, 10 epochs (`full_ft_llrd_10ep`) | 100% | 224 / 224 | 76.06 |
| **5% final: LoRA + AdaptFormer (`imp_trivialaug_288`, run 42)** | 4.992% | 288 / 320 | **78.75** |
| **Full fine-tuning final (`full_ft_llrd_288`, run 39)** | 100% | 288 / 320 | **80.14** |

What we found:

- **Adapter design mattered less than training.** At an equal 5% budget, LoRA, AdaptFormer and both together tied at
  224 px. At 288 px the combination led AdaptFormer alone by 0.87.
- **The classifier head mattered most within the 5% track.** It learns best at half the adapters' learning rate (+1.12).
  Stronger head starts (43-image class means, a trained linear probe) ended lower.
- **Spare budget is better spent on pixels than on epochs.** Training at 288 px added about 2 points to both tracks. Using
  the whole budget at 224 px added nothing.
- **Full fine-tuning keeps a lead of 1.4 to 2.1 points** at equal epochs, budget or resolution. A second seed (runs 47, 48)
  moved each track by 0.25 points or less. The 5% design stores about 12M parameters per task against 93.5M, but is
  slower at batch 1 (11.8 against 7.3 ms) and does not train faster.

Paired tests and the calibration check behind these claims are in `analysis/`. Leaderboard and test scores are not
written in this public repo.

## Run one experiment (about 1.5 hours on a Colab A100)

The steps below reproduce any run from its config. All planned runs are done, so there is no open queue any more.

1. **Pick a run.** Choose a config from `configs/` (every run in `EXPERIMENT_LOG.md` names its config). If you start a new
   experiment, tell the team so nobody runs it twice.
2. **Open the notebook.** Pull the repo. In VS Code install the **Google Colab** extension, open
   `notebooks/02_train_and_submit.ipynb`, then Select Kernel -> Colab -> New Colab Server -> **A100**.
   A T4 is about 40 times slower and cannot finish a run.
3. **Set the first cell.**
   - `CONFIG` = the config name from the queue, for example `"tune_label_smoothing"`
   - `RUN = ""`
   - `WHO` = your name
   - `EXTRA = ""` unless the queue gives a value
   - `TEAM` only names the submission file; ask the team for the number
4. **Run All.** Approve the Google Drive access when asked. The notebook downloads the dataset onto the Colab machine
   (10 to 20 minutes, about 44 GB, nothing lands on your PC), trains for about 1.2 hours, and ends with a line that starts
   with `DONE`.
5. **Send the results back.** They are in your Google Drive under `My Drive/cs604_runs/<CONFIG>/`. Copy `log.jsonl` and
   `summary.json` into `results/<CONFIG>/` in the repo and push those two files, or send them to the team.
6. **Terminate the Colab session** when you are done. It uses credits for as long as it is connected.

Things to know before you start:

- **One server at a time.** Colab refused a second GPU server on our plan ("Unable to assign server. You have too many").
  Do not click "Remove Server" while a run is going: it can remove the server that is training.
- **Changing `CONFIG`? Keep `RUN` empty.** `RUN` names the output folder. If it still holds the name of a finished run,
  the new run loads that old model instead of training.
- **A second run in the same session** needs no new download: change `CONFIG` and Run All again.
- **A disconnect** loses the dataset on the Colab machine, but not the run. Run All again and training resumes from the
  last checkpoint on Drive.
- **Forgot the submission cell, or need a zip for someone else's run?** Use `notebooks/03_make_submission.ipynb`.
  It needs the run's `config.yaml` and `final.pt` in your own Drive and trains nothing.
- **Drive space:** about 0.2 GB per adapter run, about 1.5 GB for full fine-tuning.
- **Do not upload to CodaBench** unless you are the person handling submissions (5 per day in Phase 1, 2 in total in
  Phase 2). Leaderboard scores are not written in this public repo.

## Layout

```text
common/      SHARED. Change only by pull request that all 3 of us agree on.
  data.py      dataset reading, fixed hold-out split, transforms
  build.py     pre-trained ViT-B/16 + new head + method
  head_init.py nearest-class-mean classifier head (the "zero-shot" baseline)
  train.py     training loop (bf16/fp16, resume, budget, logging, results row)
  metrics.py   trainable params, latency, GPU info, results.csv writer
  submit.py    predict val/test, write + validate + zip the CodaBench file
methods/     One file per adaptation method: apply(model, args) -> model. Register in methods/__init__.py.
configs/     One yaml per run. abl_* = equal-budget ablation, tune_* = one hyper-parameter changed,
             imp_* = the tuned recipe plus one change, full_ft_* = full fine-tuning track,
             final_* = trained on all 45 images per class (no hold-out score).
notebooks/   00 data setup/check, 01 throughput test, 02 train + submit (the one you normally need),
             03 make the CodaBench zip from a finished run (no training), 04 hypothesis charts,
             05 test-set submission (Phase 2).
analysis/    ece.py (calibration, H22) and paired_tests.py (paired bootstrap + McNemar for the deciding
             comparisons). Both re-score saved checkpoints on Colab and train nothing.
results/     results.csv (one row per run), <run>/log.jsonl + summary.json for each finished run,
             and the chart scripts plot_hypotheses.py and plot_params_accuracy.py.
EXPERIMENT_LOG.md   results, hypotheses H1 to H24, interpretation, run history. CLAUDE.md = notes for coding agents.
```

Without the notebook, from the repo root on a GPU machine:
`python -m common.train configs/lora_baseline.yaml out_dir=runs/lora who=me`.

## Rules (so that our numbers are comparable)

- Same data split, preprocessing and metrics for everyone: use `common/`, do not fork it. Need a change? Open a PR and tell the team.
- New method = new file in `methods/` + a config in `configs/`. Work on your own branch (`yourname/method`).
- One change per run, so each run is a clean ablation. Every finished run gets its two files in `results/<run>/`, a row in
  `results/results.csv` (copied from `summary.json`) and a section in `EXPERIMENT_LOG.md`.
- Trainable % counts the adapter parameters against the backbone. The classifier head is not counted (confirmed by the
  professor). The 5% limit applies to the efficient-adaptation track only.
- Never commit data, checkpoints, zips, or the course PDF (it contains the CodaBench secret link). `.gitignore` covers these.
- Notebooks are committed **without outputs** and with placeholder `WHO` / `TEAM`: `pip install nbstripout && nbstripout --install`
  once per clone, and do not commit your own values.

## Status (18 Nov 2026)

Finished. 45 training runs over 24 hypotheses, all recorded in `results/` and `EXPERIMENT_LOG.md`. Both Phase 2 test
submissions were used (5 Nov): the 5% recipe trained on all 45 images per class (`final_5pct_trivialaug`, run 30) and the
full fine-tuning model `full_ft_llrd_288` (run 39). The presentation was given on 7 Nov.
