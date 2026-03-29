# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Autoresearch is an autonomous research platform where AI agents iteratively modify training scripts to improve model performance. The repo has two tracks:
- **LLM pretraining** (`prepare.py`, `train.py`, `program.md`) — metric: `val_bpb` (lower is better)
- **Binary segmentation** (`segmentation_prepare.py`, `segmentation_train.py`, `segmentation_program.md`) — metric: `val_iou` (higher is better)

## Commands

```bash
# Setup
uv sync                                              # Install all dependencies

# LLM side
uv run prepare.py                                    # Download LLM data (one-time)
uv run train.py > run.log 2>&1                       # Run LLM experiment
grep "^val_bpb:\|^peak_vram_mb:" run.log             # Extract LLM metrics

# Segmentation side
uv run segmentation_prepare.py --data-dir /path/to/data  # Verify dataset (one-time)
uv run segmentation_train.py > run.log 2>&1               # Run segmentation experiment
grep "^val_iou:\|^peak_vram_mb:" run.log                  # Extract segmentation metrics
```

## Segmentation Architecture

Three core files mirroring the LLM structure:

| File | Editable | Purpose |
|------|----------|---------|
| `segmentation_prepare.py` | **No** | Fixed evaluation harness: local dataset loading (`images/` + `masks/` with matching names), `BinarySegmentationDataset`, train/val dataloaders, and `evaluate_iou()` |
| `segmentation_train.py` | **Yes (by agent)** | U-Net model, AdamW optimizer, training loop with BCE loss |
| `segmentation_program.md` | **Yes (by human)** | Agent instructions governing the segmentation experiment loop |

### segmentation_prepare.py (fixed)

- **Dataset**: Local directory with `images/` and `masks/` subfolders (matching filenames). Set `DATA_DIR` in the file or pass `--data-dir`. Masks are binarized: nonzero → 1 (foreground), zero → 0 (background).
- **Constants**: `IMAGE_SIZE=256`, `NUM_CLASSES=1`, `TIME_BUDGET=300`, `VAL_SPLIT=0.15`
- **Augmentations** (train only): random H/V flip, brightness/contrast jitter, ImageNet normalization
- **Evaluation**: `evaluate_iou(model, val_loader, device)` — binary IoU at 0.5 threshold on sigmoid output
- **Dataloaders**: `make_train_loader(batch_size)`, `make_val_loader(batch_size)` — expects model to output `[B, 1, H, W]` logits

### segmentation_train.py (mutable)

- **Model**: U-Net with configurable `BASE_CHANNELS` (64) and `DEPTH` (4 encoder/decoder levels)
- **Components**: ConvBlock (Conv→BN→ReLU x2), MaxPool encoder, ConvTranspose decoder with skip connections
- **Optimizer**: AdamW with progress-based LR warmup/warmdown (same wall-clock scheduling as LLM side)
- **Loss**: `BCEWithLogitsLoss` for binary segmentation
- **Mixed precision**: bf16 autocast + GradScaler
- **Key hyperparams**: `BASE_CHANNELS`, `DEPTH`, `LEARNING_RATE`, `WEIGHT_DECAY`, `BATCH_SIZE`, `GRAD_ACCUM_STEPS`
- **Output**: `val_iou`, `training_seconds`, `total_seconds`, `peak_vram_mb`, `num_steps`, `num_params_M`, `num_epochs`, `depth`, `base_channels`

### Data flow (Segmentation)

Local dataset (`images/` + `masks/`) → binarize masks (nonzero=1) → augmentation → dataloader `[B,3,256,256]` images + `[B,256,256]` masks → U-Net → `[B,1,256,256]` logits → BCE loss → `evaluate_iou()` (sigmoid > 0.5 → binary IoU)

## LLM Architecture (existing, for reference)

| File | Editable | Purpose |
|------|----------|---------|
| `prepare.py` | **No** | Fixed evaluation harness: data download, BPE tokenizer (rustbpe), dataloader, and `evaluate_bpb()` |
| `train.py` | **Yes (by agent)** | GPT model, MuonAdamW optimizer, training loop |
| `program.md` | **Yes (by human)** | Agent instructions governing the experiment loop |

HuggingFace `climbmix-400b-shuffle` → parquet shards → BPE tokenizer (vocab 8192) → dataloader (context 2048) → GPT model → CE loss → `evaluate_bpb()`

## Agent Experiment Workflow

Same structure for both LLM and segmentation:
1. Modify the train script → commit → run 5-min training → extract metrics
2. If metric improved (`val_bpb` ↓ for LLM, `val_iou` ↑ for segmentation) → keep; else → `git reset`
3. Log to results TSV (`results.tsv` for LLM, `segmentation_results.tsv` for segmentation)
4. Repeat indefinitely

## Key Constraints

- **Never modify the prepare script** — it's the fixed evaluation contract
- **No external packages** beyond what's in `pyproject.toml`
- **Fixed 5-minute budget** per experiment
- **VRAM is a soft constraint** — OOM crashes waste an iteration
