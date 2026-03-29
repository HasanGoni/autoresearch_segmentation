"""
Data preparation and evaluation for binary segmentation experiments.
Expects a local data directory with images/ and masks/ subfolders
where image and mask files share the same name.

Usage:
    python segmentation_prepare.py --data-dir /path/to/dataset
    python segmentation_prepare.py --data-dir /path/to/dataset --verify

Set DATA_DIR below or pass --data-dir on the command line.
"""

import os
import sys
import argparse

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

IMAGE_SIZE = 256         # input resolution (H x W)
NUM_CLASSES = 1          # binary segmentation (foreground vs background)
TIME_BUDGET = 300        # training time budget in seconds (5 minutes)
VAL_SPLIT = 0.15         # fraction of data used for validation

# ---------------------------------------------------------------------------
# Configuration — set DATA_DIR to your local dataset path
# ---------------------------------------------------------------------------

# Directory containing images/ and masks/ subfolders.
# Override with --data-dir CLI arg or by editing this path directly.
DATA_DIR = ""  # e.g. "/home/user/datasets/my_segmentation_data"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MASK_EXTENSIONS = {".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"}

# ---------------------------------------------------------------------------
# Data discovery
# ---------------------------------------------------------------------------

def _resolve_data_dir():
    """Return the active data directory, checking it exists."""
    data_dir = DATA_DIR
    assert data_dir and os.path.isdir(data_dir), (
        f"DATA_DIR is not set or does not exist: '{data_dir}'. "
        f"Set DATA_DIR in segmentation_prepare.py or pass --data-dir."
    )
    return data_dir


def _find_matching_file(directory, stem, extensions):
    """Find a file in directory matching stem + any of the extensions."""
    for ext in extensions:
        candidate = os.path.join(directory, stem + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def _build_file_list(data_dir=None):
    """Return list of (image_path, mask_path) pairs with matching filenames.

    Expects:
        data_dir/
            images/   (image files)
            masks/    (mask files, same stem as corresponding image)
    """
    if data_dir is None:
        data_dir = _resolve_data_dir()
    images_dir = os.path.join(data_dir, "images")
    masks_dir = os.path.join(data_dir, "masks")

    assert os.path.isdir(images_dir), f"Images directory not found: {images_dir}"
    assert os.path.isdir(masks_dir), f"Masks directory not found: {masks_dir}"

    pairs = []
    image_files = sorted(os.listdir(images_dir))
    for img_file in image_files:
        stem, ext = os.path.splitext(img_file)
        if ext.lower() not in IMAGE_EXTENSIONS:
            continue
        mask_path = _find_matching_file(masks_dir, stem, MASK_EXTENSIONS)
        if mask_path is not None:
            pairs.append((os.path.join(images_dir, img_file), mask_path))

    return pairs


def get_train_val_split(data_dir=None):
    """
    Return deterministic (train_pairs, val_pairs) split.
    Uses a fixed seed so results are reproducible.
    """
    pairs = _build_file_list(data_dir)
    assert len(pairs) > 0, (
        "No image/mask pairs found. Check that your data directory has "
        "images/ and masks/ subfolders with matching filenames."
    )
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(pairs))
    val_size = max(1, int(len(pairs) * VAL_SPLIT))
    val_indices = set(indices[:val_size].tolist())
    train_pairs = [pairs[i] for i in range(len(pairs)) if i not in val_indices]
    val_pairs = [pairs[i] for i in range(len(pairs)) if i in val_indices]
    return train_pairs, val_pairs


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BinarySegmentationDataset(Dataset):
    """Binary segmentation dataset.

    Masks are loaded as grayscale and binarized:
        pixel > 0  →  1 (foreground)
        pixel == 0 →  0 (background)

    If your masks use a different convention (e.g. 255=foreground),
    this still works because any nonzero value maps to 1.
    """
    def __init__(self, pairs, image_size, augment=False):
        self.pairs = pairs
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        # Load image
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        img = np.array(img, dtype=np.float32) / 255.0  # [H, W, 3]

        # Load mask as grayscale and binarize (nonzero = foreground)
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        mask = np.array(mask, dtype=np.float32)
        binary_mask = (mask > 0).astype(np.float32)  # [H, W]

        # Augmentations (train only)
        if self.augment:
            # Random horizontal flip
            if np.random.random() > 0.5:
                img = img[:, ::-1, :].copy()
                binary_mask = binary_mask[:, ::-1].copy()
            # Random vertical flip
            if np.random.random() > 0.5:
                img = img[::-1, :, :].copy()
                binary_mask = binary_mask[::-1, :].copy()
            # Color jitter (brightness, contrast)
            brightness = 1.0 + np.random.uniform(-0.2, 0.2)
            contrast = 1.0 + np.random.uniform(-0.2, 0.2)
            img = img * brightness
            mean = img.mean()
            img = (img - mean) * contrast + mean
            img = np.clip(img, 0.0, 1.0)

        # Normalize: ImageNet stats
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std

        # Convert to tensors: img [3, H, W], mask [H, W]
        img = torch.from_numpy(img.transpose(2, 0, 1))
        binary_mask = torch.from_numpy(binary_mask)

        return img, binary_mask


# ---------------------------------------------------------------------------
# Runtime utilities (imported by segmentation_train.py)
# ---------------------------------------------------------------------------

def make_train_loader(batch_size, image_size=IMAGE_SIZE, num_workers=4):
    """Create training dataloader with augmentation."""
    train_pairs, _ = get_train_val_split()
    dataset = BinarySegmentationDataset(train_pairs, image_size, augment=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    return loader


def make_val_loader(batch_size, image_size=IMAGE_SIZE, num_workers=4):
    """Create validation dataloader (no augmentation, deterministic)."""
    _, val_pairs = get_train_val_split()
    dataset = BinarySegmentationDataset(val_pairs, image_size, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False,
    )
    return loader


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_iou(model, val_loader, device):
    """
    Binary IoU (Intersection over Union) for foreground class.
    Threshold: 0.5 on sigmoid output.
    Returns IoU as a float (higher is better, range 0-1).
    """
    model.eval()
    total_intersection = 0
    total_union = 0

    for images, masks in val_loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(images)  # [B, 1, H, W]

        preds = (logits.squeeze(1).float().sigmoid() > 0.5).long()  # [B, H, W]
        targets = masks.long()  # [B, H, W]

        intersection = ((preds == 1) & (targets == 1)).sum().item()
        union = ((preds == 1) | (targets == 1)).sum().item()

        total_intersection += intersection
        total_union += union

    iou = total_intersection / max(total_union, 1)
    model.train()
    return iou


# ---------------------------------------------------------------------------
# Main — verify dataset
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify data for binary segmentation experiments")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to dataset directory containing images/ and masks/ subfolders")
    args = parser.parse_args()

    # Allow CLI override of DATA_DIR
    if args.data_dir:
        DATA_DIR = args.data_dir

    if not DATA_DIR:
        print("ERROR: No data directory specified.")
        print("  Either set DATA_DIR in segmentation_prepare.py")
        print("  or pass --data-dir /path/to/your/dataset")
        sys.exit(1)

    print(f"Data directory: {DATA_DIR}")
    print(f"  images/ : {os.path.join(DATA_DIR, 'images')}")
    print(f"  masks/  : {os.path.join(DATA_DIR, 'masks')}")
    print()

    # Verify
    train_pairs, val_pairs = get_train_val_split()
    print(f"Total pairs found: {len(train_pairs) + len(val_pairs)}")
    print(f"Train samples:     {len(train_pairs)}")
    print(f"Val samples:       {len(val_pairs)}")
    print()

    # Show a few examples
    print("Sample pairs:")
    for img_path, mask_path in train_pairs[:3]:
        print(f"  {os.path.basename(img_path)}  <->  {os.path.basename(mask_path)}")
    print()
    print("Done! Ready to train.")
