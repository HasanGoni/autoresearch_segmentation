"""
Autoresearch binary segmentation training script. Single-GPU, single-file.
Adapted from the LLM training loop for image segmentation (Oxford-IIIT Pet).
Usage: uv run segmentation_train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import gc
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation_prepare import (
    IMAGE_SIZE, TIME_BUDGET,
    make_train_loader, make_val_loader, evaluate_iou,
)

# ---------------------------------------------------------------------------
# U-Net Model
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Two conv layers with BatchNorm and ReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class EncoderBlock(nn.Module):
    """Downsample with MaxPool then ConvBlock."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class DecoderBlock(nn.Module):
    """Upsample with ConvTranspose then concat skip and ConvBlock."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch from odd dimensions
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    """Simple U-Net for binary segmentation.

    Args:
        in_channels: number of input channels (3 for RGB)
        base_channels: number of channels in the first encoder level
        depth: number of encoder/decoder levels (excluding bottleneck)
        num_classes: output channels (1 for binary segmentation)
    """
    def __init__(self, in_channels=3, base_channels=64, depth=4, num_classes=1):
        super().__init__()
        self.depth = depth

        # Initial conv block
        self.init_conv = ConvBlock(in_channels, base_channels)

        # Encoder
        self.encoders = nn.ModuleList()
        ch = base_channels
        for i in range(depth):
            out_ch = ch * 2
            self.encoders.append(EncoderBlock(ch, out_ch))
            ch = out_ch

        # Decoder
        self.decoders = nn.ModuleList()
        for i in range(depth):
            skip_ch = ch // 2
            out_ch = ch // 2
            self.decoders.append(DecoderBlock(ch, skip_ch, out_ch))
            ch = out_ch

        # Final 1x1 conv
        self.final_conv = nn.Conv2d(ch, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder path
        skips = []
        x = self.init_conv(x)
        skips.append(x)
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)

        # Remove bottleneck from skips (it's the current x)
        skips.pop()

        # Decoder path
        for decoder in self.decoders:
            skip = skips.pop()
            x = decoder(x, skip)

        x = self.final_conv(x)
        return x

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
BASE_CHANNELS = 64      # channels in the first encoder level (doubles each level)
DEPTH = 4               # number of encoder/decoder levels

# Optimization
LEARNING_RATE = 1e-3    # AdamW learning rate
WEIGHT_DECAY = 1e-4     # AdamW weight decay
ADAM_BETAS = (0.9, 0.999)
WARMUP_RATIO = 0.05     # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.3    # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.01   # final LR as fraction of initial

# Training
BATCH_SIZE = 16         # per-device batch size (reduce if OOM)
GRAD_ACCUM_STEPS = 1    # gradient accumulation steps
NUM_WORKERS = 4         # dataloader workers

# ---------------------------------------------------------------------------
# Setup: model, optimizer, dataloaders
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

model = UNet(
    in_channels=3,
    base_channels=BASE_CHANNELS,
    depth=DEPTH,
    num_classes=1,
).to(device)

num_params = model.num_params()
print(f"Model: UNet (base_channels={BASE_CHANNELS}, depth={DEPTH})")
print(f"Parameters: {num_params:,} ({num_params / 1e6:.1f}M)")

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    betas=ADAM_BETAS,
)

scaler = torch.amp.GradScaler()

model = torch.compile(model, dynamic=False)

train_loader = make_train_loader(BATCH_SIZE, IMAGE_SIZE, NUM_WORKERS)
val_loader = make_val_loader(BATCH_SIZE, IMAGE_SIZE, NUM_WORKERS)

print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"Batch size: {BATCH_SIZE} x {GRAD_ACCUM_STEPS} accum = {BATCH_SIZE * GRAD_ACCUM_STEPS} effective")
print(f"Time budget: {TIME_BUDGET}s")

# Schedules (all based on progress = training_time / TIME_BUDGET)

def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0
epoch = 0

# BCE with logits loss for binary segmentation
loss_fn = nn.BCEWithLogitsLoss()

train_iter = iter(train_loader)

def get_batch():
    """Get next batch, cycling through epochs."""
    global train_iter, epoch
    try:
        images, masks = next(train_iter)
    except StopIteration:
        epoch += 1
        train_iter = iter(train_loader)
        images, masks = next(train_iter)
    return images.to(device, non_blocking=True), masks.to(device, non_blocking=True)

# Prefetch first batch
images, masks = get_batch()

while True:
    torch.cuda.synchronize()
    t0 = time.time()

    for micro_step in range(GRAD_ACCUM_STEPS):
        with autocast_ctx:
            logits = model(images)  # [B, 1, H, W]
            loss = loss_fn(logits.squeeze(1), masks) / GRAD_ACCUM_STEPS

        train_loss_val = loss.detach() * GRAD_ACCUM_STEPS
        scaler.scale(loss).backward()
        images, masks = get_batch()

    # Progress and LR schedule
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    for group in optimizer.param_groups:
        group["lr"] = LEARNING_RATE * lrm

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    train_loss_f = train_loss_val.item()

    # Fast fail: abort if loss is exploding or NaN
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        exit(1)

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    if step > 5:
        total_training_time += dt

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
    pct_done = 100 * progress
    remaining = max(0, TIME_BUDGET - total_training_time)

    print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lr: {LEARNING_RATE * lrm:.2e} | dt: {dt * 1000:.0f}ms | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

    # GC management
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    # Time's up — but only stop after warmup steps so we don't count compilation
    if step > 5 and total_training_time >= TIME_BUDGET:
        break

print()  # newline after \r training log

# Final eval
model.eval()
val_iou = evaluate_iou(model, val_loader, device)

# Final summary
t_end = time.time()
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

print("---")
print(f"val_iou:          {val_iou:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"num_epochs:       {epoch}")
print(f"depth:            {DEPTH}")
print(f"base_channels:    {BASE_CHANNELS}")
