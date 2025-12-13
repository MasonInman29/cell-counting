"""
main_unbalanced_v9.py

This script handles the *imbalance-focused* baseline and ablations.

Modes:
  --mode baseline  : CellCounter + weighted L1 + log transform
  --mode huber     : CellCounter + weighted Huber loss + log transform
  --mode hetero    : Heteroscedastic (CellCounter + variance head, log-space)
  --mode overlap   : Overlap tiling inference using a trained baseline/hetero model
  --mode relative  : CellCounter + relative error loss (no log transform)

Training modes (baseline / huber / hetero / relative) share:
  - stratified train/val split (by low/mid/high buckets)
  - WeightedRandomSampler (bucket inverse-frequency)
  - imbalance-aware bucket-weighted losses (baseline/huber/hetero)
  - log(count+1) training / expm1() at evaluation (baseline/huber/hetero);
    relative uses raw counts without log-transform.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset_handler import CellDataset
from model import CellCounter

from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

# Reuse metrics implementation from eval_by_staining
from eval_by_staining import compute_metrics as compute_metrics_np


# ---------------------------------------------------------------------
# 0. Global seed
# ---------------------------------------------------------------------


def set_global_seed(seed: int = 42):
    """Set seed for Python, NumPy, and PyTorch (CPU/CUDA)."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Fix random seed for reproducibility
set_global_seed(42)


# ---------------------------------------------------------------------
# 1. Build / load cell_counts.csv
# ---------------------------------------------------------------------


def load_or_build_cell_counts(dataset_root: str = "dataset") -> pd.DataFrame:
    """
    Scan dataset/img and dataset/ground_truth and build a table with:
      - id
      - path (image path)
      - count (number of cells)
      - bucket ("low" / "mid" / "high")
    Store it as dataset/cell_counts.csv.
    If the file already exists, just load it.
    """
    root = Path(dataset_root)
    csv_path = root / "cell_counts.csv"

    if csv_path.exists():
        print(f"[INFO] Loading existing {csv_path}")
        return pd.read_csv(csv_path)

    print("[INFO] cell_counts.csv not found. Building it now...")
    img_dir = root / "img"
    gt_dir = root / "ground_truth"
    meta_path = root / "metadata.csv"

    img_paths = sorted(list(img_dir.glob("*.tif")) + list(img_dir.glob("*.tiff")))
    print("Total images:", len(img_paths))

    rows = []
    for i, img_path in enumerate(img_paths, start=1):
        if i % 200 == 0:
            print(f"  {i}/{len(img_paths)} processed...")

        stem = img_path.stem
        gt_path = gt_dir / f"{stem}.csv"
        if not gt_path.exists():
            continue

        gt = pd.read_csv(gt_path)
        count = len(gt)

        # Buckets: 0–250 / 250–500 / 500+
        if count <= 250:
            bucket = "low"
        elif count <= 500:
            bucket = "mid"
        else:
            bucket = "high"

        rows.append(
            {
                "id": stem,
                "path": str(img_path),
                "count": count,
                "bucket": bucket,
            }
        )

    df = pd.DataFrame(rows)

    # Attach "set" column (trainval / test) from metadata.csv, if available
    if meta_path.exists():
        meta = pd.read_csv(meta_path)
        meta.columns = [c.strip().lower() for c in meta.columns]
        if "id" in meta.columns and "set" in meta.columns:
            meta["id"] = meta["id"].astype(str)
            df["id"] = df["id"].astype(str)
            df = df.merge(meta[["id", "set"]], on="id", how="left")

    df.to_csv(csv_path, index=False)
    print("[INFO] Saved:", csv_path)
    print("\n[INFO] Bucket counts:\n", df["bucket"].value_counts())
    if "set" in df.columns:
        print("\n[INFO] Set counts:\n", df["set"].value_counts())
    return df


# ---------------------------------------------------------------------
# 2. Stratified train/val split (keep Low/Mid/High ratio)
# ---------------------------------------------------------------------


def stratified_train_val_split(df_counts: pd.DataFrame, train_ratio: float = 0.8):
    """
    Perform stratified train/val split on Low/Mid/High buckets.
    If metadata.csv has set == "trainval", only split that subset.
    """
    if "set" in df_counts.columns:
        pool = df_counts[df_counts["set"] == "trainval"].copy()
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

    if len(train_rows) == 0:
        # Fallback if something went wrong
        train_df = pool.copy()
        val_df = pool.iloc[0:0].copy()
    else:
        train_df = pd.concat(train_rows, ignore_index=True)
        val_df = pd.concat(val_rows, ignore_index=True)

    print("[INFO] Stratified split:")
    if len(train_df) > 0:
        print("  train buckets:\n", train_df["bucket"].value_counts())
    else:
        print("  train buckets: EMPTY")
    if len(val_df) > 0:
        print("  val   buckets:\n", val_df["bucket"].value_counts())
    else:
        print("  val   buckets: EMPTY")

    return train_df, val_df


# ---------------------------------------------------------------------
# 3. DataLoaders (balanced sampler) + test split
# ---------------------------------------------------------------------


def get_data_loaders_unbalanced(batch_size: int = 8, dataset_root: str = "dataset"):
    """
    - Use cell_counts.csv
    - Stratified train/val split on set == 'trainval'
    - Separate test set: set == 'test'
    - Balanced batches via WeightedRandomSampler using bucket inverse frequency

    Returns:
        train_loader, val_loader, test_loader, bucket_to_weight,
        val_paths, val_transform
    """
    df_counts = load_or_build_cell_counts(dataset_root)

    # Train/val split on trainval subset
    train_df, val_df = stratified_train_val_split(df_counts, train_ratio=0.8)

    # Test split from metadata.set == 'test'
    if "set" in df_counts.columns:
        test_df = df_counts[df_counts["set"] == "test"].copy()
    else:
        test_df = df_counts.iloc[0:0].copy()

    train_paths = train_df["path"].tolist()
    val_paths = val_df["path"].tolist()
    test_paths = test_df["path"].tolist()

    # Transforms (same as original main)
    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_dataset = CellDataset(train_paths, transform=train_transform)
    val_dataset = CellDataset(val_paths, transform=val_transform)
    test_dataset = CellDataset(test_paths, transform=val_transform)

    use_cuda = torch.cuda.is_available()

    # Balanced sampler: bucket inverse frequency
    if len(train_df) > 0:
        buckets = train_df["bucket"].values
        uniq, cnts = np.unique(buckets, return_counts=True)
        bucket_to_weight = {b: 1.0 / float(c) for b, c in zip(uniq, cnts)}
        sample_weights = np.array(
            [bucket_to_weight[b] for b in buckets], dtype=np.float32
        )

        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=use_cuda,
        )
    else:
        bucket_to_weight = {}
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=use_cuda,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=use_cuda,
    )

    if len(test_dataset) > 0:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=use_cuda,
        )
    else:
        test_loader = None

    return (
        train_loader,
        val_loader,
        test_loader,
        bucket_to_weight,
        val_paths,
        val_transform,
    )


# ---------------------------------------------------------------------
# 4. Common utilities (bucket, evaluation, plotting)
# ---------------------------------------------------------------------


def bucket_from_count(count: float) -> str:
    """Map a raw cell count to a bucket string."""
    c = float(count)
    if c <= 250:
        return "low"
    elif c <= 500:
        return "mid"
    else:
        return "high"


def evaluate_model_unbalanced(
    model: nn.Module, data_loader: DataLoader, device, use_log: bool = True
) -> float:
    """
    Simple evaluation that returns only MAE.
    If model outputs log(y+1), convert back using expm1 with clamping to avoid overflow.
    """
    if data_loader is None:
        return float("nan")

    model.eval()
    mae = nn.L1Loss(reduction="mean")

    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            if isinstance(outputs, tuple):  # hetero model returns (mu_log, log_var)
                outputs = outputs[0]

            if use_log:
                # Clamp in log-space to avoid numerical overflow
                preds_log = outputs.clamp(min=0.0, max=15.0)
                preds = torch.expm1(preds_log)
            else:
                preds = outputs

            loss = mae(preds, labels)
            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(1, n_batches)


def compute_acp_10_20_from_arrays(
    preds: np.ndarray, labels: np.ndarray, eps: float = 1e-8
):
    """
    Compute ACP@10% and ACP@20% given prediction and label arrays.
    For non-zero labels, use relative error threshold (10% / 20%).
    For near-zero labels, fall back to absolute error threshold (0.10 / 0.20).
    """
    assert preds.shape == labels.shape

    preds = np.nan_to_num(preds, nan=0.0, posinf=1e6, neginf=-1e6)
    labels = np.nan_to_num(labels, nan=0.0, posinf=1e6, neginf=-1e6)

    abs_err = np.abs(preds - labels)
    mask_nonzero = np.abs(labels) > eps

    rel_ok10 = np.zeros_like(labels, dtype=bool)
    rel_ok20 = np.zeros_like(labels, dtype=bool)

    # Non-zero labels: use relative thresholds
    rel_ok10[mask_nonzero] = (
        abs_err[mask_nonzero] / np.abs(labels[mask_nonzero]) <= 0.10
    )
    rel_ok20[mask_nonzero] = (
        abs_err[mask_nonzero] / np.abs(labels[mask_nonzero]) <= 0.20
    )

    # Zero labels: use small absolute thresholds
    rel_ok10[~mask_nonzero] = abs_err[~mask_nonzero] <= 0.10
    rel_ok20[~mask_nonzero] = abs_err[~mask_nonzero] <= 0.20

    acp10 = float(rel_ok10.mean() * 100.0)
    acp20 = float(rel_ok20.mean() * 100.0)
    return acp10, acp20


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

    Key idea:
        relative_error = |pred - true| / |true|

    Returns:
        Scalar tensor loss.
    """
    # Shape alignment
    if preds.ndim == 1 and labels.ndim == 2:
        preds = preds.unsqueeze(1)
    if preds.ndim == 2 and labels.ndim == 1:
        labels = labels.unsqueeze(1)

    # Clean labels
    labels = torch.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0)

    # Total count per image (here it's already global, but keep it general)
    p_tot = preds.sum(dim=1) if preds.ndim == 2 else preds.squeeze(1)
    l_tot = labels.sum(dim=1) if labels.ndim == 2 else labels.squeeze(1)

    abs_err = torch.abs(p_tot - l_tot)

    # Clamp denominator to avoid division by zero / exploding error
    denominator = torch.clamp(l_tot.abs(), min=min_denom)
    rel_err = abs_err / denominator  # per-sample relative error

    if power != 1.0:
        rel_err = rel_err ** power

    loss = rel_err.mean()

    if not torch.isfinite(loss):
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=1e6)

    return loss


def evaluate_model_unbalanced_metrics(
    model: nn.Module, data_loader: DataLoader, device, use_log: bool = True
) -> dict:
    """
    Collect all predictions / labels and compute:
      - MAE, MSE, RMSE, MAPE, ACP@5% (from eval_by_staining.compute_metrics)
      - ACP@10%, ACP@20%

    Returns:
        {
          "mae": ...,
          "mse": ...,
          "rmse": ...,
          "mape": ...,
          "acp5": ...,
          "acp10": ...,
          "acp20": ...
        }
    """
    if data_loader is None:
        return {
            k: float("nan")
            for k in ("mae", "mse", "rmse", "mape", "acp5", "acp10", "acp20")
        }

    model.eval()
    preds_all = []
    labels_all = []

    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs = imgs.to(device)
            labels = labels.to(device).float()

            out = model(imgs)
            if isinstance(out, tuple):  # hetero model
                out = out[0]

            if use_log:
                preds_log = out.clamp(min=0.0, max=15.0)
                preds = torch.expm1(preds_log)
            else:
                preds = out

            p = preds.view(-1).cpu().float().numpy()
            l = labels.view(-1).cpu().float().numpy()
            preds_all.append(p)
            labels_all.append(l)

    if not preds_all:
        return {
            k: float("nan")
            for k in ("mae", "mse", "rmse", "mape", "acp5", "acp10", "acp20")
        }

    preds_all = np.concatenate(preds_all, axis=0)
    labels_all = np.concatenate(labels_all, axis=0)

    base = compute_metrics_np(preds_all, labels_all)  # mae/mse/rmse/mape/acp(5%)
    acp10, acp20 = compute_acp_10_20_from_arrays(preds_all, labels_all)

    return {
        "mae": base["mae"],
        "mse": base["mse"],
        "rmse": base["rmse"],
        "mape": base["mape"],
        "acp5": base["acp"],
        "acp10": acp10,
        "acp20": acp20,
    }


def evaluate_model_unbalanced_by_bucket(
    model: nn.Module,
    data_loader: DataLoader,
    device,
    use_log: bool = True,
) -> dict:
    """
    Compute metrics per bucket (low / mid / high / all).

    Returns:
        {
          "low":  {"n": ..., "mae": ..., "rmse": ..., "mape": ..., "acp5": ..., "acp10": ..., "acp20": ...},
          "mid":  {...},
          "high": {...},
          "all":  {...}
        }
    """
    if data_loader is None:
        return {}

    model.eval()
    preds_list = []
    labels_list = []
    bucket_list = []

    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs = imgs.to(device)
            labels = labels.to(device).float()

            out = model(imgs)
            if isinstance(out, tuple):  # hetero model (mu_log, log_var)
                out = out[0]

            if use_log:
                preds_log = out.clamp(min=0.0, max=15.0)
                preds = torch.expm1(preds_log)
            else:
                preds = out

            p = preds.view(-1).cpu().float().numpy()
            l = labels.view(-1).cpu().float().numpy()
            preds_list.append(p)
            labels_list.append(l)

            bucket_list.extend([bucket_from_count(c) for c in l])

    if not preds_list:
        return {}

    preds_all = np.concatenate(preds_list, axis=0)
    labels_all = np.concatenate(labels_list, axis=0)
    buckets_all = np.array(bucket_list)

    results = {}

    def _compute_for_mask(mask, name: str):
        if mask.sum() == 0:
            return
        base = compute_metrics_np(preds_all[mask], labels_all[mask])
        acp10, acp20 = compute_acp_10_20_from_arrays(
            preds_all[mask], labels_all[mask]
        )
        results[name] = {
            "n": int(mask.sum()),
            "mae": base["mae"],
            "mse": base["mse"],
            "rmse": base["rmse"],
            "mape": base["mape"],
            "acp5": base["acp"],
            "acp10": acp10,
            "acp20": acp20,
        }

    for b in ["low", "mid", "high"]:
        mask = buckets_all == b
        _compute_for_mask(mask, b)

    # all
    _compute_for_mask(np.ones_like(buckets_all, dtype=bool), "all")

    return results


def plot_losses(
    train_losses,
    val_losses,
    out_path: str = "loss_plot_unbalanced.png",
    title: str = "Training & Validation",
):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation MAE")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.savefig(out_path)
    plt.close()


# ---------------------------------------------------------------------
# 5. Baseline: Weighted L1 + log transform (+ Early stopping)
# ---------------------------------------------------------------------


def train_model_baseline(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    bucket_to_weight: dict,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    use_log: bool = True,
    patience: int = 5,
):
    """
    Baseline:
      - Stratified train/val split
      - Balanced sampler
      - Weighted L1 loss
      - Log transform
      - Early stopping based on validation MAE
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    base_criterion = nn.L1Loss(reduction="none")  # per-sample loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for imgs, labels in tqdm(
            train_loader, desc=f"[Baseline] Epoch {epoch+1}/{num_epochs}"
        ):
            imgs = imgs.to(device)
            labels = labels.to(device)  # (B,1)

            # Log transform
            if use_log:
                labels_log = torch.log1p(labels)
            else:
                labels_log = labels

            optimizer.zero_grad()
            outputs = model(imgs)  # (B,1)

            loss_per_sample = base_criterion(outputs, labels_log).view(-1)

            # Bucket-based sample weights
            labels_cpu = labels.view(-1).detach().cpu().numpy()
            bucket_list = [bucket_from_count(c) for c in labels_cpu]
            weights = torch.tensor(
                [bucket_to_weight.get(b, 1.0) for b in bucket_list],
                dtype=torch.float32,
                device=device,
            )

            loss = (loss_per_sample * weights).mean()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        print(f"[Baseline] Epoch {epoch+1}, Training Loss: {epoch_loss:.4f}")

        val_loss = evaluate_model_unbalanced(
            model, val_loader, device, use_log=use_log
        )
        val_losses.append(val_loss)
        print(f"[Baseline] Validation MAE: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val - 1e-3:
            best_val = val_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[Baseline] Early stopping triggered at epoch {epoch+1}")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_losses, val_losses, best_val


# ---------------------------------------------------------------------
# 6. Ablation 1: Huber loss (+ Early stopping)
# ---------------------------------------------------------------------


def train_model_huber(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    bucket_to_weight: dict,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    use_log: bool = True,
    delta: float = 1.0,
    patience: int = 5,
):
    """
    Same as baseline but using SmoothL1 (Huber) instead of L1.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    base_criterion = nn.SmoothL1Loss(beta=delta, reduction="none")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for imgs, labels in tqdm(
            train_loader, desc=f"[Huber] Epoch {epoch+1}/{num_epochs}"
        ):
            imgs = imgs.to(device)
            labels = labels.to(device)

            if use_log:
                labels_log = torch.log1p(labels)
            else:
                labels_log = labels

            optimizer.zero_grad()
            outputs = model(imgs)

            loss_per_sample = base_criterion(outputs, labels_log).view(-1)

            labels_cpu = labels.view(-1).detach().cpu().numpy()
            bucket_list = [bucket_from_count(c) for c in labels_cpu]
            weights = torch.tensor(
                [bucket_to_weight.get(b, 1.0) for b in bucket_list],
                dtype=torch.float32,
                device=device,
            )

            loss = (loss_per_sample * weights).mean()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        print(f"[Huber] Epoch {epoch+1}, Training Loss: {epoch_loss:.4f}")

        val_loss = evaluate_model_unbalanced(
            model, val_loader, device, use_log=use_log
        )
        val_losses.append(val_loss)
        print(f"[Huber] Validation MAE: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val - 1e-3:
            best_val = val_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[Huber] Early stopping triggered at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_losses, val_losses, best_val


# ---------------------------------------------------------------------
# 6b. Ablation: Relative error loss (+ Early stopping)
# ---------------------------------------------------------------------


def train_model_relative(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    bucket_to_weight: dict,  # kept for API consistency (not used here)
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    patience: int = 5,
):
    """
    Ablation:
      - Same data pipeline (stratified split, balanced sampler)
      - Train on raw counts (no log transform)
      - Objective: mean relative error loss on per-image total counts
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for imgs, labels in tqdm(
            train_loader, desc=f"[Relative] Epoch {epoch+1}/{num_epochs}"
        ):
            imgs = imgs.to(device)
            labels = labels.to(device)  # (B,1), raw counts

            optimizer.zero_grad()
            outputs = model(imgs)  # (B,1), raw count prediction

            # Relative error loss on total counts
            loss = relative_error_loss(outputs, labels, min_denom=1.0)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        print(f"[Relative] Epoch {epoch+1}, Training Loss: {epoch_loss:.4f}")

        # Validation: evaluate MAE on raw counts (use_log=False)
        val_loss = evaluate_model_unbalanced(
            model, val_loader, device, use_log=False
        )
        val_losses.append(val_loss)
        print(f"[Relative] Validation MAE: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val - 1e-3:
            best_val = val_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[Relative] Early stopping triggered at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_losses, val_losses, best_val


# ---------------------------------------------------------------------
# 7. Heteroscedastic regression (+ Early stopping)
# ---------------------------------------------------------------------


class HeteroCellCounter(nn.Module):
    """
    Wrapper around a base CellCounter that predicts both:
      - mu_log: mean of log(y+1)
      - log_var: log variance for heteroscedastic Gaussian NLL
    """

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base = base_model
        self.sigma_head = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1)
        )

    def forward(self, x):
        mu_log = self.base(x)  # (B,1)
        log_var = self.sigma_head(mu_log)  # (B,1)
        return mu_log, log_var


def train_model_hetero(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    bucket_to_weight: dict,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    use_log: bool = True,
    patience: int = 5,
):
    """
    Heteroscedastic Gaussian NLL:
      loss = 0.5 * exp(-log_var) * (mu - y)^2 + 0.5 * log_var
    (with bucket-based weighting)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for imgs, labels in tqdm(
            train_loader, desc=f"[Hetero] Epoch {epoch+1}/{num_epochs}"
        ):
            imgs = imgs.to(device)
            labels = labels.to(device)

            if use_log:
                labels_log = torch.log1p(labels)
            else:
                labels_log = labels

            optimizer.zero_grad()

            mu_log, log_var = model(imgs)  # (B,1), (B,1)

            diff2 = (mu_log - labels_log) ** 2
            nll = 0.5 * torch.exp(-log_var) * diff2 + 0.5 * log_var
            loss_per_sample = nll.view(-1)

            labels_cpu = labels.view(-1).detach().cpu().numpy()
            bucket_list = [bucket_from_count(c) for c in labels_cpu]
            weights = torch.tensor(
                [bucket_to_weight.get(b, 1.0) for b in bucket_list],
                dtype=torch.float32,
                device=device,
            )

            loss = (loss_per_sample * weights).mean()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        print(f"[Hetero] Epoch {epoch+1}, Training NLL: {epoch_loss:.4f}")

        val_loss = evaluate_model_unbalanced(
            model, val_loader, device, use_log=use_log
        )
        val_losses.append(val_loss)
        print(f"[Hetero] Validation MAE: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val - 1e-3:
            best_val = val_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[Hetero] Early stopping triggered at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_losses, val_losses, best_val


# ---------------------------------------------------------------------
# 8. Overlap tiling (inference)
# ---------------------------------------------------------------------


def predict_count_overlap_tiling(
    model: nn.Module,
    pil_img: Image.Image,
    transform,
    tile_size: int = 256,
    overlap: int = 64,
    device=None,
    use_log: bool = True,
) -> float:
    """
    Simple overlap tiling:
      - Split a large image into overlapping tiles of size tile_size x tile_size
      - Predict count for each tile
      - Return the average of tile predictions
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    w, h = pil_img.size
    step = tile_size - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than tile_size")

    preds = []

    with torch.no_grad():
        y = 0
        while y < h:
            x = 0
            y_end = min(y + tile_size, h)
            y_start = max(0, y_end - tile_size)

            while x < w:
                x_end = min(x + tile_size, w)
                x_start = max(0, x_end - tile_size)

                crop = pil_img.crop((x_start, y_start, x_end, y_end))
                img_t = transform(crop).unsqueeze(0).to(device)

                out = model(img_t)
                if isinstance(out, tuple):  # hetero model
                    out = out[0]

                if use_log:
                    preds_log = out.clamp(min=0.0, max=15.0)
                    count_pred = torch.expm1(preds_log)
                else:
                    count_pred = out

                preds.append(count_pred.item())

                x += step
            y += step

    return float(np.mean(preds))


# ---------------------------------------------------------------------
# 9. Experiment runners
# ---------------------------------------------------------------------


def _print_bucket_metrics(title: str, bucket_metrics: dict):
    if not bucket_metrics:
        print(f"{title} (no data)")
        return
    print(title)
    for b in ["low", "mid", "high", "all"]:
        if b not in bucket_metrics:
            continue
        m = bucket_metrics[b]
        print(
            f"  [{b.upper()}] N={m['n']} "
            f"MAE={m['mae']:.4f}  "
            f"RMSE={m['rmse']:.4f}  "
            f"ACP@5={m['acp5']:.2f}%  "
            f"ACP@10={m['acp10']:.2f}%  "
            f"ACP@20={m['acp20']:.2f}%"
        )


def run_baseline(num_epochs=100, patience=5):
    batch_size = 8
    learning_rate = 1e-3

    (
        train_loader,
        val_loader,
        test_loader,
        bucket_to_weight,
        _,
        _,
    ) = get_data_loaders_unbalanced(batch_size=batch_size, dataset_root="dataset")

    model = CellCounter()
    trained_model, train_losses, val_losses, best_val = train_model_baseline(
        model,
        train_loader,
        val_loader,
        bucket_to_weight=bucket_to_weight,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        use_log=True,
        patience=patience,
    )

    plot_losses(
        train_losses,
        val_losses,
        out_path="loss_plot_baseline.png",
        title="Baseline",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_val = evaluate_model_unbalanced(
        trained_model, val_loader, device, use_log=True
    )
    print(f"[BASELINE BEST VAL] MAE (tracked): {best_val:.4f}")
    print(f"[BASELINE FINAL VAL] MAE (after reload): {final_val:.4f}")

    # Full metrics on validation and test
    val_metrics = evaluate_model_unbalanced_metrics(
        trained_model, val_loader, device, use_log=True
    )
    print(
        "[BASELINE VAL METRICS] "
        f"MAE={val_metrics['mae']:.4f}  "
        f"RMSE={val_metrics['rmse']:.4f}  "
        f"MAPE={val_metrics['mape']:.2f}%  "
        f"ACP@5={val_metrics['acp5']:.2f}%  "
        f"ACP@10={val_metrics['acp10']:.2f}%  "
        f"ACP@20={val_metrics['acp20']:.2f}%"
    )

    # Per-bucket metrics on validation
    bucket_metrics_val = evaluate_model_unbalanced_by_bucket(
        trained_model, val_loader, device, use_log=True
    )
    _print_bucket_metrics("\n[BASELINE VAL METRICS BY BUCKET]", bucket_metrics_val)

    if test_loader is not None:
        test_metrics = evaluate_model_unbalanced_metrics(
            trained_model, test_loader, device, use_log=True
        )
        print(
            "\n[BASELINE TEST METRICS] "
            f"MAE={test_metrics['mae']:.4f}  "
            f"RMSE={test_metrics['rmse']:.4f}  "
            f"MAPE={test_metrics['mape']:.2f}%  "
            f"ACP@5={test_metrics['acp5']:.2f}%  "
            f"ACP@10={test_metrics['acp10']:.2f}%  "
            f"ACP@20={test_metrics['acp20']:.2f}%"
        )

        bucket_metrics_test = evaluate_model_unbalanced_by_bucket(
            trained_model, test_loader, device, use_log=True
        )
        _print_bucket_metrics("\n[BASELINE TEST METRICS BY BUCKET]", bucket_metrics_test)
    else:
        print("[BASELINE] No explicit test set (test_loader is None).")

    torch.save(trained_model.state_dict(), "cell_counter_unbalanced_baseline.pth")
    print("[INFO] Saved model: cell_counter_unbalanced_baseline.pth")


def run_huber(num_epochs=100, patience=5):
    batch_size = 8
    learning_rate = 1e-3

    (
        train_loader,
        val_loader,
        test_loader,
        bucket_to_weight,
        _,
        _,
    ) = get_data_loaders_unbalanced(batch_size=batch_size, dataset_root="dataset")

    model = CellCounter()
    trained_model, train_losses, val_losses, best_val = train_model_huber(
        model,
        train_loader,
        val_loader,
        bucket_to_weight=bucket_to_weight,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        use_log=True,
        patience=patience,
    )

    plot_losses(
        train_losses,
        val_losses,
        out_path="loss_plot_huber.png",
        title="Huber Loss",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_val = evaluate_model_unbalanced(
        trained_model, val_loader, device, use_log=True
    )
    print(f"[HUBER BEST VAL] MAE: {best_val:.4f}")
    print(f"[HUBER FINAL VAL] MAE: {final_val:.4f}")

    val_metrics = evaluate_model_unbalanced_metrics(
        trained_model, val_loader, device, use_log=True
    )
    print(
        "[HUBER VAL METRICS] "
        f"MAE={val_metrics['mae']:.4f}  "
        f"RMSE={val_metrics['rmse']:.4f}  "
        f"MAPE={val_metrics['mape']:.2f}%  "
        f"ACP@5={val_metrics['acp5']:.2f}%  "
        f"ACP@10={val_metrics['acp10']:.2f}%  "
        f"ACP@20={val_metrics['acp20']:.2f}%"
    )

    bucket_metrics_val = evaluate_model_unbalanced_by_bucket(
        trained_model, val_loader, device, use_log=True
    )
    _print_bucket_metrics("\n[HUBER VAL METRICS BY BUCKET]", bucket_metrics_val)

    if test_loader is not None:
        test_metrics = evaluate_model_unbalanced_metrics(
            trained_model, test_loader, device, use_log=True
        )
        print(
            "[HUBER TEST METRICS] "
            f"MAE={test_metrics['mae']:.4f}  "
            f"RMSE={test_metrics['rmse']:.4f}  "
            f"MAPE={test_metrics['mape']:.2f}%  "
            f"ACP@5={test_metrics['acp5']:.2f}%  "
            f"ACP@10={test_metrics['acp10']:.2f}%  "
            f"ACP@20={test_metrics['acp20']:.2f}%"
        )

        bucket_metrics_test = evaluate_model_unbalanced_by_bucket(
            trained_model, test_loader, device, use_log=True
        )
        _print_bucket_metrics("[HUBER TEST METRICS BY BUCKET]", bucket_metrics_test)
    else:
        print("[HUBER] No explicit test set.")

    torch.save(trained_model.state_dict(), "cell_counter_huber.pth")
    print("[INFO] Saved model: cell_counter_huber.pth")


def run_relative(num_epochs=100, patience=5):
    batch_size = 8
    learning_rate = 1e-3

    (
        train_loader,
        val_loader,
        test_loader,
        bucket_to_weight,
        _,
        _,
    ) = get_data_loaders_unbalanced(batch_size=batch_size, dataset_root="dataset")

    model = CellCounter()
    trained_model, train_losses, val_losses, best_val = train_model_relative(
        model,
        train_loader,
        val_loader,
        bucket_to_weight=bucket_to_weight,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        patience=patience,
    )

    plot_losses(
        train_losses,
        val_losses,
        out_path="loss_plot_relative.png",
        title="Relative Error Loss",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_val = evaluate_model_unbalanced(
        trained_model, val_loader, device, use_log=False
    )
    print(f"[RELATIVE BEST VAL] MAE: {best_val:.4f}")
    print(f"[RELATIVE FINAL VAL] MAE: {final_val:.4f}")

    # Full metrics (raw counts, no log)
    val_metrics = evaluate_model_unbalanced_metrics(
        trained_model, val_loader, device, use_log=False
    )
    print(
        "[RELATIVE VAL METRICS] "
        f"MAE={val_metrics['mae']:.4f}  "
        f"RMSE={val_metrics['rmse']:.4f}  "
        f"MAPE={val_metrics['mape']:.2f}%  "
        f"ACP@5={val_metrics['acp5']:.2f}%  "
        f"ACP@10={val_metrics['acp10']:.2f}%  "
        f"ACP@20={val_metrics['acp20']:.2f}%"
    )

    bucket_metrics_val = evaluate_model_unbalanced_by_bucket(
        trained_model, val_loader, device, use_log=False
    )
    _print_bucket_metrics("\n[RELATIVE VAL METRICS BY BUCKET]", bucket_metrics_val)

    if test_loader is not None:
        test_metrics = evaluate_model_unbalanced_metrics(
            trained_model, test_loader, device, use_log=False
        )
        print(
            "[RELATIVE TEST METRICS] "
            f"MAE={test_metrics['mae']:.4f}  "
            f"RMSE={test_metrics['rmse']:.4f}  "
            f"MAPE={test_metrics['mape']:.2f}%  "
            f"ACP@5={test_metrics['acp5']:.2f}%  "
            f"ACP@10={test_metrics['acp10']:.2f}%  "
            f"ACP@20={test_metrics['acp20']:.2f}%"
        )

        bucket_metrics_test = evaluate_model_unbalanced_by_bucket(
            trained_model, test_loader, device, use_log=False
        )
        _print_bucket_metrics("[RELATIVE TEST METRICS BY BUCKET]", bucket_metrics_test)
    else:
        print("[RELATIVE] No explicit test set.")

    torch.save(trained_model.state_dict(), "cell_counter_relative.pth")
    print("[INFO] Saved model: cell_counter_relative.pth")


def run_hetero(num_epochs=100, patience=5):
    batch_size = 8
    learning_rate = 1e-3

    (
        train_loader,
        val_loader,
        test_loader,
        bucket_to_weight,
        _,
        _,
    ) = get_data_loaders_unbalanced(batch_size=batch_size, dataset_root="dataset")

    base_model = CellCounter()
    model = HeteroCellCounter(base_model)

    trained_model, train_losses, val_losses, best_val = train_model_hetero(
        model,
        train_loader,
        val_loader,
        bucket_to_weight=bucket_to_weight,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        use_log=True,
        patience=patience,
    )

    plot_losses(
        train_losses,
        val_losses,
        out_path="loss_plot_hetero.png",
        title="Heteroscedastic",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_val = evaluate_model_unbalanced(
        trained_model, val_loader, device, use_log=True
    )
    print(f"[HETERO BEST VAL] MAE: {best_val:.4f}")
    print(f"[HETERO FINAL VAL] MAE: {final_val:.4f}")

    val_metrics = evaluate_model_unbalanced_metrics(
        trained_model, val_loader, device, use_log=True
    )
    print(
        "[HETERO VAL METRICS] "
        f"MAE={val_metrics['mae']:.4f}  "
        f"RMSE={val_metrics['rmse']:.4f}  "
        f"MAPE={val_metrics['mape']:.2f}%  "
        f"ACP@5={val_metrics['acp5']:.2f}%  "
        f"ACP@10={val_metrics['acp10']:.2f}%  "
        f"ACP@20={val_metrics['acp20']:.2f}%"
    )

    bucket_metrics_val = evaluate_model_unbalanced_by_bucket(
        trained_model, val_loader, device, use_log=True
    )
    _print_bucket_metrics("\n[HETERO VAL METRICS BY BUCKET]", bucket_metrics_val)

    if test_loader is not None:
        test_metrics = evaluate_model_unbalanced_metrics(
            trained_model, test_loader, device, use_log=True
        )
        print(
            "[HETERO TEST METRICS] "
            f"MAE={test_metrics['mae']:.4f}  "
            f"RMSE={test_metrics['rmse']:.4f}  "
            f"MAPE={test_metrics['mape']:.2f}%  "
            f"ACP@5={test_metrics['acp5']:.2f}%  "
            f"ACP@10={test_metrics['acp10']:.2f}%  "
            f"ACP@20={test_metrics['acp20']:.2f}%"
        )

        bucket_metrics_test = evaluate_model_unbalanced_by_bucket(
            trained_model, test_loader, device, use_log=True
        )
        _print_bucket_metrics("[HETERO TEST METRICS BY BUCKET]", bucket_metrics_test)
    else:
        print("[HETERO] No explicit test set.")

    torch.save(trained_model.state_dict(), "cell_counter_hetero.pth")
    print("[INFO] Saved model: cell_counter_hetero.pth")


def run_overlap(
    model_path="cell_counter_unbalanced_baseline.pth",
    dataset_root="dataset",
    max_images=None,
):
    """
    Load a trained model and evaluate overlap-tiling inference on the validation set.
    If model_path name contains 'hetero', a HeteroCellCounter is instantiated.
    Otherwise, a baseline CellCounter is used.
    """
    batch_size = 8
    (
        _,
        val_loader,
        _,
        _,
        val_paths,
        val_transform,
    ) = get_data_loaders_unbalanced(batch_size=batch_size, dataset_root=dataset_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Choose proper model class depending on the checkpoint
    model_path_obj = Path(model_path)
    if "hetero" in model_path_obj.stem.lower():
        print("[INFO] Using HeteroCellCounter for overlap tiling")
        base_model = CellCounter()
        model = HeteroCellCounter(base_model)
        use_log = True
    elif "relative" in model_path_obj.stem.lower():
        print("[INFO] Using CellCounter (relative-loss trained) for overlap tiling")
        model = CellCounter()
        use_log = False
    else:
        print("[INFO] Using baseline CellCounter for overlap tiling")
        model = CellCounter()
        use_log = True

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    gt_dir = Path(dataset_root) / "ground_truth"

    preds = []
    labels = []
    buckets = []

    paths_to_eval = val_paths if max_images is None else val_paths[: max_images]

    for p in tqdm(paths_to_eval, desc="[Overlap] Inference"):
        p = Path(p)
        img = Image.open(p).convert("RGB")
        pred = predict_count_overlap_tiling(
            model,
            img,
            transform=val_transform,
            tile_size=256,
            overlap=64,
            device=device,
            use_log=use_log,
        )

        gt_path = gt_dir / f"{p.stem}.csv"
        gt = pd.read_csv(gt_path)
        true_count = len(gt)

        preds.append(pred)
        labels.append(true_count)
        buckets.append(bucket_from_count(true_count))

    preds = np.array(preds, dtype=float)
    labels = np.array(labels, dtype=float)

    if len(preds) == 0:
        print("[OVERLAP] No validation images to evaluate.")
        return

    base_metrics = compute_metrics_np(preds, labels)
    acp10, acp20 = compute_acp_10_20_from_arrays(preds, labels)

    print(
        f"[OVERLAP] VAL (ALL) "
        f"MAE={base_metrics['mae']:.4f}  "
        f"RMSE={base_metrics['rmse']:.4f}  "
        f"MAPE={base_metrics['mape']:.2f}%  "
        f"ACP@5={base_metrics['acp']:.2f}%  "
        f"ACP@10={acp10:.2f}%  "
        f"ACP@20={acp20:.2f}%"
    )

    # Per-bucket metrics for overlap
    buckets = np.array(buckets)
    overlap_bucket_metrics = {}

    for b in ["low", "mid", "high"]:
        mask = buckets == b
        if mask.sum() == 0:
            continue
        base_b = compute_metrics_np(preds[mask], labels[mask])
        acp10_b, acp20_b = compute_acp_10_20_from_arrays(preds[mask], labels[mask])
        overlap_bucket_metrics[b] = {
            "n": int(mask.sum()),
            "mae": base_b["mae"],
            "rmse": base_b["rmse"],
            "mape": base_b["mape"],
            "acp5": base_b["acp"],
            "acp10": acp10_b,
            "acp20": acp20_b,
        }

    _print_bucket_metrics("\n[OVERLAP VAL METRICS BY BUCKET]", overlap_bucket_metrics)


# ---------------------------------------------------------------------
# 10. Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["baseline", "huber", "hetero", "overlap", "relative"],
        default="baseline",
        help="which experiment to run",
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="number of training epochs"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="early stopping patience (in epochs)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="cell_counter_unbalanced_baseline.pth",
        help="model path used for overlap mode",
    )
    args = parser.parse_args()

    # Seed is already fixed at import time, but you could re-set here if desired.
    # set_global_seed(42)

    if args.mode == "baseline":
        run_baseline(num_epochs=args.epochs, patience=args.patience)
    elif args.mode == "huber":
        run_huber(num_epochs=args.epochs, patience=args.patience)
    elif args.mode == "hetero":
        run_hetero(num_epochs=args.epochs, patience=args.patience)
    elif args.mode == "overlap":
        run_overlap(
            model_path=args.model_path, dataset_root="dataset", max_images=100
        )
    elif args.mode == "relative":
        run_relative(num_epochs=args.epochs, patience=args.patience)
