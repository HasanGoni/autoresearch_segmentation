# autoresearch — binary segmentation

This is an experiment to have the LLM do its own research on binary image segmentation.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `seg-mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `segmentation_prepare.py` — fixed constants, data download, dataset class, dataloader, evaluation. Do not modify.
   - `segmentation_train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `DATA_DIR` in `segmentation_prepare.py` is set to a valid path containing `images/` and `masks/` subfolders. Run `uv run segmentation_prepare.py` to verify the dataset.
5. **Initialize segmentation_results.tsv**: Create `segmentation_results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run segmentation_train.py`.

**What you CAN do:**
- Modify `segmentation_train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, augmentation strategy, loss function, etc.

**What you CANNOT do:**
- Modify `segmentation_prepare.py`. It is read-only. It contains the fixed evaluation, data loading, dataset, and training constants (time budget, image size, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_iou` function in `segmentation_prepare.py` is the ground truth metric.

**The goal is simple: get the highest val_iou.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size, the loss function. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_iou gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_iou:          0.850000
training_seconds: 300.1
total_seconds:    320.5
peak_vram_mb:     8192.0
num_steps:        5000
num_params_M:     31.0
num_epochs:       15
depth:            4
base_channels:    64
```

You can extract the key metric from the log file:

```
grep "^val_iou:" run.log
```

## Logging results

When an experiment is done, log it to `segmentation_results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_iou	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_iou achieved (e.g. 0.850000) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 8.0 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_iou	memory_gb	status	description
a1b2c3d	0.850000	8.0	keep	baseline
b2c3d4e	0.872000	8.2	keep	increase base channels to 96
c3d4e5f	0.845000	8.0	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model depth (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/seg-mar5`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `segmentation_train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run segmentation_train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_iou:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the segmentation_results.tsv file, leave it untracked by git)
8. If val_iou improved (higher), you "advance" the branch, keeping the git commit
9. If val_iou is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate.

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — try different architectures (DeepLab-style ASPP, attention gates, feature pyramid), different loss functions (Dice loss, focal loss, combined losses), different augmentations, different optimizers, different learning rate schedules, try combining previous near-misses, try more radical changes. The loop runs until the human interrupts you, period.
