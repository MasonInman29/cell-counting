"""
File: train_scnn.py
Authors: Mason Inman, Vincent Dumoulin, and Jiwon Choi
Adapted from original by Abdurahman Mohammed

Description: Train a cell-counting model (MMNet) on the IDCIA-style dataset,
             with ACP-aware loss or log-space loss, plus Jiwon-style stratified sampling and
             bucket-balanced batches.

Bias Mitigation:
    - Custom stratified train/val split preserving Low/Mid/High count buckets.
    - WeightedRandomSampler to balance batches by bucket during training.
    - ACP-aware loss to reduce bias towards low-count images.
    - Utilize 'final_augmented_dataset' to address limited data and class imbalance.
"""

import os
import argparse
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt

from dataset_handler import CellDataset
from model import MMNetCellCounter


# -------------------------------------------------------------------------
# Reproducibility helpers
# -------------------------------------------------------------------------

def set_random_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------------------------------------------------------
# Dataset utilities
# -------------------------------------------------------------------------

def detect_class_map(dataset_root: str):
    # Scan ground_truth CSV files and return a dict mapping label -> index.
    # If no label/type column is found, return None (single-class counting).
    root = Path(dataset_root)
    gt_dir = root / 'ground_truth'
    if not gt_dir.exists():
        return None

    labels = set()
    for p in sorted(gt_dir.glob('*.csv')):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        for col in ('type', 'label', 'class'):
            if col in df.columns:
                vals = df[col].dropna().unique().tolist()
                labels.update(vals)
                break

    if not labels:
        return None

    labels = sorted(list(labels))
    return {lab: i for i, lab in enumerate(labels)}


def _all_image_paths(root: Path):
    img_dir = root / 'img'
    exts = ('*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg')
    paths = []
    for ext in exts:
        paths.extend(img_dir.glob(ext))
    return sorted(paths)


# -------------------------------------------------------------------------
# count CSV + stratified split + bucket sampler
# -------------------------------------------------------------------------

# jiwons code - integrated by Mason
def load_or_build_cell_counts(dataset_root: str = "dataset") -> pd.DataFrame:
    """
    Build or load dataset_root/cell_counts.csv with:
      - id: image stem
      - path: full image path
      - count: total cell count
      - bucket: 'low' / 'mid' / 'high' based on count

    Uses all images under dataset_root/img and ground_truth/*.csv.
    If metadata.csv exists, attaches a split-like column if available.
    """
    root = Path(dataset_root)
    csv_path = root / "cell_counts.csv"

    if csv_path.exists():
        print(f"[INFO] Loading existing {csv_path}")
        return pd.read_csv(csv_path)

    print("[INFO] cell_counts.csv not found. Building it now...")
    img_paths = _all_image_paths(root)
    gt_dir = root / "ground_truth"
    meta_path = root / "metadata.csv"

    print(f"[INFO] Total images found under img/: {len(img_paths)}")

    rows = []
    for i, img_path in enumerate(img_paths, start=1):
        if i % 200 == 0:
            print(f"  {i}/{len(img_paths)} images processed...")

        stem = img_path.stem
        gt_path = gt_dir / f"{stem}.csv"
        if not gt_path.exists():
            continue

        try:
            gt = pd.read_csv(gt_path)
        except Exception:
            continue

        count = len(gt)

        # Buckets: 0–250 / 250–500 / 500+
        if count <= 250:
            bucket = "low"
        elif count <= 500:
            bucket = "mid"
        else:
            bucket = "high"

        rows.append({
            "id": stem,
            "path": str(img_path),
            "count": count,
            "bucket": bucket,
        })

    df = pd.DataFrame(rows)

    # Attach split-like info from metadata, if present
    if meta_path.exists() and not df.empty:
        try:
            meta = pd.read_csv(meta_path)
            meta.columns = [c.strip().lower() for c in meta.columns]
            if "id" in meta.columns:
                meta["id"] = meta["id"].astype(str)
                df["id"] = df["id"].astype(str)
                split_col = None
                for col in ("set", "split", "subset", "partition"):
                    if col in meta.columns:
                        split_col = col
                        break
                if split_col is not None:
                    df = df.merge(meta[["id", split_col]], on="id", how="left")
        except Exception as e:
            print(f"[WARN] Failed to merge metadata.csv: {e}")

    df.to_csv(csv_path, index=False)
    print("[INFO] Saved:", csv_path)
    print("\n[INFO] Bucket counts:\n", df["bucket"].value_counts())
    return df

# jiwons code - integrated by Mason
def stratified_train_val_split(df_counts: pd.DataFrame,
                               train_ratio: float = 0.8):
    """
    Stratified train/val split preserving Low/Mid/High bucket ratios.

    If a split-like column exists (set/split/subset/partition), we:
      - If any value == 'trainval', restrict pool to those rows.
      - Else, just use all rows (you can extend to exclude 'test' if needed).
    """
    if df_counts.empty:
        raise RuntimeError("cell_counts DataFrame is empty; cannot stratify.")

    # detect split-like column, if any
    split_col = None
    for col in ("set", "split", "subset", "partition"):
        if col in df_counts.columns:
            split_col = col
            break

    if split_col is not None:
        vals = df_counts[split_col].astype(str).str.lower()
        if "trainval" in vals.values:
            pool = df_counts[vals == "trainval"].copy()
        else:
            # Use all rows; you could optionally filter out explicit "test"
            pool = df_counts.copy()
    else:
        pool = df_counts.copy()

    train_rows = []
    val_rows = []

    for bucket in ["low", "mid", "high"]:
        sub = pool[pool["bucket"] == bucket]
        if len(sub) == 0:
            continue
        sub = sub.sample(frac=1.0, random_state=42)  # shuffle
        n_train = int(len(sub) * train_ratio)
        train_rows.append(sub.iloc[:n_train])
        val_rows.append(sub.iloc[n_train:])

    if not train_rows or not val_rows:
        # Fallback: non-stratified random split
        print("[WARN] Stratified split failed; falling back to random split.")
        pool = pool.sample(frac=1.0, random_state=42)
        n_train = int(len(pool) * train_ratio)
        train_df = pool.iloc[:n_train].reset_index(drop=True)
        val_df = pool.iloc[n_train:].reset_index(drop=True)
    else:
        train_df = pd.concat(train_rows, ignore_index=True)
        val_df = pd.concat(val_rows, ignore_index=True)

    print("[INFO] Stratified split:")
    print("  train buckets:\n", train_df["bucket"].value_counts())
    print("  val   buckets:\n", val_df["bucket"].value_counts())

    return train_df, val_df


# jiwons code - integrated by Mason
def get_data_loaders(dataset_root: str = 'dataset',
                     batch_size: int = 8,
                     val_fraction: float = 0.2,
                     seed: int = 42):
    """
    Creates training and validation DataLoaders for the dataset using:
      - cell_counts.csv
      - Stratified train/val split by bucket (low/mid/high)
      - Balanced batches via bucket-based WeightedRandomSampler

    Returns:
        train_loader, val_loader, class_map
    """
    set_random_seeds(seed)
    root = Path(dataset_root)

    # Load or build per-image counts and buckets
    df_counts = load_or_build_cell_counts(dataset_root)
    train_ratio = 1.0 - val_fraction
    train_df, val_df = stratified_train_val_split(df_counts, train_ratio=train_ratio)

    train_paths = [Path(p) for p in train_df["path"].tolist()]
    val_paths = [Path(p) for p in val_df["path"].tolist()]

    # Detect multi-type vs single-type counting
    class_map = detect_class_map(dataset_root)

    # Data transforms (same as before)
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = CellDataset(train_paths, transform=train_transform, class_map=class_map)
    val_dataset = CellDataset(val_paths, transform=val_transform, class_map=class_map)

    pin = torch.cuda.is_available()

    # jiwons code - integrated by Mason
    buckets = train_df["bucket"].values
    uniq, cnts = np.unique(buckets, return_counts=True)
    bucket_to_weight = {b: 1.0 / float(c) for b, c in zip(uniq, cnts)}

    sample_weights = np.array([bucket_to_weight[b] for b in buckets],
                              dtype=np.float32)
    sample_weights = torch.from_numpy(sample_weights)

    # Weighted sample to help class imbalance.
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=4,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=pin,
    )

    return train_loader, val_loader, class_map


# -------------------------------------------------------------------------
# Metrics (MAE, MSE, RMSE, MAPE, ACP)
# -------------------------------------------------------------------------

def compute_metrics(preds: np.ndarray, labels: np.ndarray, eps: float = 1e-8):
    # preds, labels: 1-D arrays of total counts per image (floats)
    # Returns dict: mae, mse, rmse, mape (percent), acp (percent within 5%).
    assert preds.shape == labels.shape
    err = preds - labels # direct L1 error

    abs_err = np.abs(err)

    mae = float(np.mean(abs_err))

    mse = float(np.mean(err ** 2))

    rmse = float(np.sqrt(mse))

    denom = np.maximum(np.abs(labels), eps)
    mape = float(np.mean((abs_err / denom))) * 100.0

    # ACP: accept if relative error <= 5%
    rel_ok = np.zeros_like(labels, dtype=bool) 
    mask_nonzero = np.abs(labels) > eps
    rel_ok[mask_nonzero] = (abs_err[mask_nonzero] / np.abs(labels[mask_nonzero])) <= 0.05 
    rel_ok[~mask_nonzero] = (abs_err[~mask_nonzero] <= 0.05)
    acp = float(rel_ok.mean() * 100.0) # e.g. ACP = the mean of "accurate" predictions, mapped to percent.

    # return in dict for convenience for plotting :D Yay
    return {'mae': mae, 'mse': mse, 'rmse': rmse, 'mape': mape, 'acp': acp}


def acp_aware_loss(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    ACP-aware loss on total counts per image:
    - Works for (B,) or (B, C) preds/labels.
    - Mixes absolute and relative error.
    - Upweights higher-count images within the batch.
    """
    # shape alignment
    if preds.ndim == 1 and labels.ndim == 2:
        preds = preds.unsqueeze(1)
    if preds.ndim == 2 and labels.ndim == 1:
        labels = labels.unsqueeze(1)

    # Clean any accidental NaNs/Infs in labels (e.g. from bad CSVs)
    labels = torch.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0)

    # totals per image (sum across classes if needed)
    p_tot = preds.sum(dim=1) if preds.ndim == 2 else preds.squeeze(1)
    l_tot = labels.sum(dim=1) if labels.ndim == 2 else labels.squeeze(1)

    # base & relative error
    base_mae = torch.abs(p_tot - l_tot)
    rel_err = base_mae / torch.clamp(l_tot.abs(), min=1.0)

    # upweight higher-count images
    max_l = torch.clamp(l_tot.max(), min=1.0)
    rel_scale = torch.clamp(l_tot / max_l, min=0.0)   # ~[0, 1]

    alpha = 2.0
    lambda_rel = 0.5

    weights = 1.0 + alpha * rel_scale

    # loss = mix of actual loss and relative loss (with constant multiplier hyperparams)
    loss = (weights * (base_mae + lambda_rel * rel_err)).mean()
    
    # Final safety check
    if not torch.isfinite(loss):
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=1e6)

    return loss

def relative_error_loss(
        preds: torch.Tensor,
        labels: torch.Tensor,
        min_denom: float = 1.0,
        power: float = 1.0,
    ) -> torch.Tensor:
        """
        Mean relative error loss on total counts per image.

        Loss per image:
        L_i = |p_i - y_i| / max(|y_i|, min_denom)

        Key idea of this loss function though is:
        relative_error = |pred - true| / |true|


        Args:
        preds:  (B,) or (B, C) predicted counts.
        labels: (B,) or (B, C) true counts.
        min_denom: lower bound for denominator to avoid exploding error
                    when y_i is near zero (e.g. 1.0 = 1 cell).

        Returns:
        Scalar tensor loss.
        """
        # Shape alignment
        if preds.ndim == 1 and labels.ndim == 2:
            preds = preds.unsqueeze(1)
        if preds.ndim == 2 and labels.ndim == 1:
            labels = labels.unsqueeze(1)

        # Clean labels (safety)
        labels = torch.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0)

        # Total count per image (sum across classes if needed)
        p_tot = preds.sum(dim=1) if preds.ndim == 2 else preds.squeeze(1)
        l_tot = labels.sum(dim=1) if labels.ndim == 2 else labels.squeeze(1)

        abs_err = torch.abs(p_tot - l_tot)

        # Clamp denominator to avoid division by zero and exploding error vals
        denominator = torch.clamp(l_tot.abs(), min=min_denom)
        rel_err = abs_err / denominator  # per-sample relative error

        loss = (rel_err).mean()

        if not torch.isfinite(loss):
            # ruh roh no bueno
            loss = torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=1e6)

        return loss

# -------------------------------------------------------------------------
# Training / evaluation
# -------------------------------------------------------------------------

def train_model(model,
                train_loader,
                val_loader,
                num_epochs: int = 100,
                learning_rate: float = 1e-3,
                checkpoint_dir: str = 'checkpoints',
                run_dir: Path = None,
                use_log: bool = False):
    """
    Train loop supporting:
      - Original ACP-aware loss on counts (use_log=False)
      - Log transform training: model predicts log(count+1),
        loss is L1 in log space (use_log=True), metrics are on counts via expm1.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # AMP toggle (kept off by default)
    use_amp = False
    scaler = GradScaler(enabled=(device.type == 'cuda' and use_amp))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, num_epochs))

    train_history = []  # list of dicts per epoch
    val_history = []
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running = 0.0

        # accumulate preds/labels for epoch metrics (sums across classes, in COUNT space)
        epoch_preds = []
        epoch_labels = []

        for imgs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == 'cuda' and use_amp)):
                out = model(imgs)
                preds = out[0] if isinstance(out, tuple) else out

                # Debug: check model outputs for NaNs/Inf
                if not torch.isfinite(preds).all():
                    print(">>> Non-finite model output detected!")
                    print("    sample values:", preds.detach().view(-1)[:10])
                    raise RuntimeError("Non-finite model output")

                # Align shapes for loss & metrics
                if preds.ndim == 1 and labels.ndim == 2:
                    preds_m = preds.unsqueeze(1)
                    labels_m = labels
                elif preds.ndim == 2 and labels.ndim == 1:
                    preds_m = preds
                    labels_m = labels.unsqueeze(1)
                else:
                    preds_m = preds
                    labels_m = labels

                if use_log:
                    # jiwons code - integrated by Mason: log transform training
                    # Model predicts log(count+1) per class; labels are log1p(count).
                    labels_log = torch.log1p(torch.clamp(labels_m, min=0.0))
                    loss = F.l1_loss(preds_m, labels_log)
                else:
                    # Original ACP-aware loss on counts
                    loss = acp_aware_loss(preds_m, labels_m)

                # Debug: check loss
                if not torch.isfinite(loss):
                    print(">>> Non-finite loss detected in training.")
                    print("    preds sample:", preds_m.detach().view(-1)[:10])
                    print("    labels sample:", labels_m.detach().view(-1)[:10])
                    raise RuntimeError("Non-finite loss in training")

            # Backward + optimizer
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            running += float(loss)

            # accumulate for metrics in COUNT space
            with torch.no_grad():
                if use_log:
                    preds_counts = torch.expm1(preds_m)
                else:
                    preds_counts = preds_m

                labels_counts = labels_m

                p = preds_counts.detach().cpu().float()
                l = labels_counts.detach().cpu().float()
                p_tot_np = p.sum(dim=1).numpy() if p.ndim == 2 else p.numpy()
                l_tot_np = l.sum(dim=1).numpy() if l.ndim == 2 else l.numpy()
                epoch_preds.append(p_tot_np)
                epoch_labels.append(l_tot_np)

        # compute train metrics for the epoch (COUNT space)
        if epoch_preds:
            train_preds = np.concatenate(epoch_preds, axis=0)
            train_labels = np.concatenate(epoch_labels, axis=0)
            train_metrics = compute_metrics(train_preds, train_labels)
        else:
            train_metrics = {k: float('nan') for k in ('mae', 'mse', 'rmse', 'mape', 'acp')}
        train_history.append(train_metrics)

        train_loss = running / max(1, len(train_loader))
        train_losses.append(train_loss)

        # validate
        val_metrics, val_loss = evaluate_model(model, val_loader, device, use_log=use_log)
        val_losses.append(val_loss)
        val_history.append(val_metrics)

        scheduler.step()

        print(
            f'Epoch {epoch+1:03d} | '
            f'train_loss={train_loss:.4f} val_loss={val_loss:.4f} | '
            f'train_MAE={train_metrics["mae"]:.4f} val_MAE={val_metrics["mae"]:.4f} | '
            f'train_ACP={train_metrics["acp"]:.2f}% val_ACP={val_metrics["acp"]:.2f}% '
            f'| use_log={use_log}'
        )

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Saved checkpoint: {checkpoint_path}')

        # Save train/val metrics per epoch to CSV in run_dir (if provided)
        if run_dir is not None:
            rows = []
            for epoch_idx, (tr, va) in enumerate(zip(train_history, val_history), start=1):
                tr_row = {'epoch': epoch_idx, 'split': 'train', **tr}
                va_row = {'epoch': epoch_idx, 'split': 'val',   **va}
                rows.append(tr_row)
                rows.append(va_row)
            metrics_df = pd.DataFrame(rows)
            metrics_path = run_dir / 'metrics.csv'
            metrics_df.to_csv(metrics_path, index=False)
            print(f'Saved metrics to {metrics_path}')

    return model, train_history, val_history, train_losses, val_losses


def evaluate_model(model, data_loader, device, use_log: bool = False):
    """
    Validation loop:
      - If use_log=False: uses ACP-aware loss on counts.
      - If use_log=True : uses MAE on counts derived via expm1(preds_log).
    Metrics are always computed in COUNT space.
    """
    model.eval()
    total = 0.0
    preds_list = []
    labels_list = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()

            out = model(inputs)
            preds = out[0] if isinstance(out, tuple) else out  # (B,C) or (B,)

            # Align shapes
            if preds.ndim == 1 and labels.ndim == 2:
                preds_m = preds.unsqueeze(1)
                labels_m = labels
            elif preds.ndim == 2 and labels.ndim == 1:
                preds_m = preds
                labels_m = labels.unsqueeze(1)
            else:
                preds_m = preds
                labels_m = labels

            if use_log:
                # jiwons code - integrated by Mason: evaluate MAE on counts via expm1
                preds_counts = torch.expm1(preds_m)
                labels_counts = labels_m
                p_tot = preds_counts.sum(dim=1)
                l_tot = labels_counts.sum(dim=1)
                loss = torch.mean(torch.abs(p_tot - l_tot))
            else:
                # ACP-aware loss on counts
                loss = acp_aware_loss(preds_m, labels_m)

            # NaN/Inf guard for validation
            if not torch.isfinite(loss):
                print(">>> Non-finite loss in validation.")
                print("    preds sample:", preds_m.view(-1)[:10])
                print("    labels sample:", labels_m.view(-1)[:10])
                raise RuntimeError("Non-finite loss in validation")

            total += float(loss)

            # accumulate totals per sample for metrics in COUNT space
            if use_log:
                preds_counts = torch.expm1(preds_m)
            else:
                preds_counts = preds_m
            labels_counts = labels_m

            p = preds_counts.cpu().float()
            l = labels_counts.cpu().float()
            p_tot = p.sum(dim=1).numpy() if p.ndim == 2 else p.numpy()
            l_tot = l.sum(dim=1).numpy() if l.ndim == 2 else l.numpy()
            preds_list.append(p_tot)
            labels_list.append(l_tot)

    val_loss = total / max(1, len(data_loader))

    if preds_list:
        all_preds = np.concatenate(preds_list, axis=0)
        all_labels = np.concatenate(labels_list, axis=0)
        metrics = compute_metrics(all_preds, all_labels)
    else:
        metrics = {k: float('nan') for k in ('mae', 'mse', 'rmse', 'mape', 'acp')}

    return metrics, val_loss


# -------------------------------------------------------------------------
# Plotting helpers
# -------------------------------------------------------------------------

def plot_metrics(train_hist, val_hist, out_dir: str = 'plots'):
    # Create and save one PNG per metric (train vs val) in ./plots/
    os.makedirs(out_dir, exist_ok=True)
    metrics = ['mae', 'mse', 'rmse', 'mape', 'acp']
    epochs = list(range(1, len(train_hist) + 1))
    for m in metrics:
        tr = [h[m] for h in train_hist]
        va = [h[m] for h in val_hist]
        plt.figure(figsize=(7, 4))
        plt.plot(epochs, tr, label='train')
        plt.plot(epochs, va, label='val')
        plt.xlabel('Epoch')
        plt.ylabel(m.upper())
        plt.title(m.upper())
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{m}.png'))
        plt.close()


def plot_losses(train_losses, val_losses, out_path: str = 'plots/losses.png'):
    # Plots the training and validation losses across epochs.
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(train_losses, label='Train loss')
    plt.plot(val_losses, label='Val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# -------------------------------------------------------------------------
# Main CLI
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train cell counter with ACP-aware or log-space loss.')
    parser.add_argument('--dataset-root', type=str, default='dataset',
                        help='Root directory containing img/ and ground_truth/ folders.')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--save-model', type=str, default='cell_counter.pth')
    parser.add_argument('--num-classes', type=int, default=None,
                        help='Override auto-detected number of classes.')                    
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out-root', type=str, default='runs',
                        help='Root directory where all run folders are created.')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Optional custom run name; if not set, use timestamp+model+seed.')
    # jiwons code - integrated by Mason
    parser.add_argument('--use-log', action='store_true',
                        help='Train on log(count+1) (L1 in log space) and evaluate metrics in count space via expm1.')
    args = parser.parse_args()

    # device handling (main train/eval functions still use torch.cuda.is_available())
    if args.device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    elif args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    set_random_seeds(args.seed)

    # ----------------- Unique run directory -----------------
    # Build a run_name if not provided: e.g. 20251119-213045_mmnet_seed42_log
    if args.run_name is None:
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        log_tag = '_log' if args.use_log else ''
        run_name = f'{timestamp}_seed{args.seed}{log_tag}'
    else:
        run_name = args.run_name

    out_root = Path(args.out_root)
    run_dir = out_root / run_name
    ckpt_dir = run_dir / 'checkpoints'
    plots_dir = run_dir / 'plots'

    # Make directories
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f'Run directory: {run_dir}')

    # prepare data loaders (Jiwon-style stratified + balanced batches)
    train_loader, val_loader, class_map = get_data_loaders(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        val_fraction=0.2,
        seed=args.seed,
    )

    # decide number of classes
    if args.num_classes is not None:
        num_classes = args.num_classes
    else:
        num_classes = len(class_map) if class_map is not None else 1

    print(f'Using MMNetCellCounter (num_classes={num_classes})')
    model = MMNetCellCounter(in_channels=3, num_classes=num_classes)

    # Train the model
    trained_model, train_history, val_history, train_losses, val_losses = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        checkpoint_dir=str(ckpt_dir),
        run_dir=run_dir,
        use_log=args.use_log,
    )

    # Plot the training and validation metrics
    plot_metrics(train_history, val_history, out_dir=str(plots_dir))
    plot_losses(train_losses, val_losses, out_path=str(plots_dir / 'losses.png'))

    # Save trained weights
    final_model_path = run_dir / args.save_model
    torch.save(trained_model.state_dict(), final_model_path)
    print(f'Saved trained model to {final_model_path}')

    # Final "test" on the validation split, just to print metrics nicely
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_metrics, test_loss = evaluate_model(trained_model.to(device), val_loader, device, use_log=args.use_log)
    print(f'Final Val Loss: {test_loss:.4f} | Val MAE: {val_metrics["mae"]:.4f} | Val ACP: {val_metrics["acp"]:.2f}%')

if __name__ == '__main__':
    main()
