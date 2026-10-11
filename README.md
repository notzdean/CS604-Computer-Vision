# CS604 project: adapting ViT-B/16 to a 10,000-class dataset

Team project: full fine-tuning and parameter-efficient adaptation (LoRA, AdaptFormer, BitFit, ...) of a pre-trained
ViT-B/16, under a 2 h x 48 GB budget per run. Everything runs on Google Colab GPUs from VS Code (or the Colab web UI).

The results so far, what they mean and the list of runs still to do are in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

## Run one experiment (about 1.5 hours on a Colab A100)

1. **Pick a run.** Open the run queue in `EXPERIMENT_LOG.md`, choose one marked "open", and tell the team so nobody runs it twice.
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
configs/     One yaml per run. abl_* = equal-budget ablation, tune_* = abl_combined with one setting changed.
notebooks/   00 data setup/check, 01 throughput test, 02 train + submit (the one you normally need).
results/     results.csv (one row per run) and <run>/log.jsonl + summary.json for each finished run.
EXPERIMENT_LOG.md   results, interpretation, run queue. CLAUDE.md = notes for coding agents.
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

## Status (11 Oct 2026)

The pipeline works end to end on an A100, including the CodaBench file. On our hold-out (2 labelled training images per
class): LoRA rank 8 reaches 68.78% with 0.34% of the backbone trained, and LoRA + AdaptFormer reaches 71.29% with 4.998%.
Full fine-tuning (the required baseline) is running; the ablation and tuning runs are queued. Hyper-parameters are lightly tuned at best.
