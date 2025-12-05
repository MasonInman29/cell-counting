"""
File: eval_by_staining.py
Author: Mason Inman (adapted)
Description:
    Evaluate a trained cell counting model on the TEST split, broken down by
    the "staining" column from metadata.csv.

    For each unique staining (e.g., Cy3, AF488, DAPI), this script computes:
      - MAE, MSE, RMSE, MAPE, ACP (5% acceptance)
    and also overall metrics across all test images.

Usage example:

    python eval_by_staining.py \
        --dataset-root dataset \
        --model-path runs/.../cell_counter.pth \
        --model-type mmnet \
        --use-log \
        --batch-size 8 \
        --out-csv staining_metrics_mmnet_log.csv
"""

import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset_handler import CellDataset
from model import CellCounter, SwitchCNN, MMNetCellCounter


# -------------------------------------------------------------------------
# Repro / helpers
# -------------------------------------------------------------------------

def set_random_seeds(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def detect_class_map(dataset_root: str):
    """
    Scan ground_truth CSV files and return a dict mapping label -> index.
    If no label/type/class column is found, return None (single-class counting).
    """
    root = Path(dataset_root)
    gt_dir = root / "ground_truth"
    if not gt_dir.exists():
        return None

    labels = set()
    for p in sorted(gt_dir.glob("*.csv")):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        for col in ("type", "label", "class"):
            if col in df.columns:
                vals = df[col].dropna().unique().tolist()
                labels.update(vals)
                break

    if not labels:
        return None

    labels = sorted(list(labels))
    return {lab: i for i, lab in enumerate(labels)}


def compute_metrics(preds: np.ndarray, labels: np.ndarray, eps: float = 1e-8):
    """
    preds, labels: 1-D arrays of total counts per image (floats)
    Returns dict: mae, mse, rmse, mape (percent), acp (percent within 5%).
    """
    assert preds.shape == labels.shape

    # Clean NaN/Inf first
    preds = np.nan_to_num(preds, nan=0.0, posinf=1e6, neginf=-1e6)
    labels = np.nan_to_num(labels, nan=0.0, posinf=1e6, neginf=-1e6)

    err = preds - labels
    # Clip errors to avoid overflow in err**2
    err = np.clip(err, -1e6, 1e6)
    abs_err = np.abs(err)

    mae = float(np.mean(abs_err))
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))

    denom = np.maximum(np.abs(labels), eps)
    mape = float(np.mean((abs_err / denom))) * 100.0

    # ACP: accept if relative error <= 5% (handle label==0 with small absolute tolerance)
    rel_ok = np.zeros_like(labels, dtype=bool)
    mask_nonzero = np.abs(labels) > eps
    rel_ok[mask_nonzero] = (abs_err[mask_nonzero] / np.abs(labels[mask_nonzero])) <= 0.05
    rel_ok[~mask_nonzero] = (abs_err[~mask_nonzero] <= 0.05)
    acp = float(rel_ok.mean() * 100.0)

    return {"mae": mae, "mse": mse, "rmse": rmse, "mape": mape, "acp": acp}


# -------------------------------------------------------------------------
# Metadata / path resolution
# -------------------------------------------------------------------------

def load_metadata(dataset_root: str) -> pd.DataFrame:
    root = Path(dataset_root)
    meta_path = root / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.csv not found under {root}")

    df = pd.read_csv(meta_path)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def resolve_image_path_from_id(dataset_root: str, img_id: str) -> str:
    """
    Given an image id (e.g., '2330'), try to resolve the corresponding image path
    under dataset_root/img with common extensions.
    """
    root = Path(dataset_root)
    img_dir = root / "img"
    stem = str(img_id)

    exts = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
    for ext in exts:
        candidate = img_dir / f"{stem}{ext}"
        if candidate.exists():
            return str(candidate)

    # If nothing matched, you can decide to raise or return None.
    raise FileNotFoundError(f"No image file found for id={img_id} in {img_dir} with extensions {exts}")


def get_test_df_by_staining(dataset_root: str,
                            staining_col: str = "staining",
                            set_col: str = "set",
                            test_values=None) -> pd.DataFrame:
    """
    Load metadata, select TEST rows based on 'set' column, and attach full paths
    using the 'id' column via dataset_root/img/{id}.(tif/tiff/png/jpg/jpeg).

    test_values: list of strings considered as "test" in set_col.
                 Default: ['test', 'testing'].
    """
    if test_values is None:
        test_values = ["test", "testing"]

    df = load_metadata(dataset_root)

    if "id" not in df.columns:
        raise ValueError("metadata.csv must have an 'id' column matching image filenames.")

    if staining_col.lower() not in df.columns:
        raise ValueError(f"Staining column '{staining_col}' not found in metadata.csv")

    if set_col.lower() not in df.columns:
        raise ValueError(
            f"Set/split column '{set_col}' not found in metadata.csv. "
            "You can change this via --set-col."
        )

    staining_col = staining_col.lower()
    set_col = set_col.lower()

    set_vals = df[set_col].astype(str).str.lower()
    mask_test = set_vals.isin([v.lower() for v in test_values])
    test_df = df[mask_test].copy()

    if test_df.empty:
        raise RuntimeError(
            f"No test rows found in metadata where {set_col} in {test_values}. "
            "Check your metadata or adjust --set-col/--test-values."
        )

    # Attach image paths using id
    paths = []
    for img_id in test_df["id"].astype(str).tolist():
        try:
            p = resolve_image_path_from_id(dataset_root, img_id)
            paths.append(p)
        except FileNotFoundError:
            paths.append(None)

    test_df["full_path"] = paths
    # keep only rows where the file actually exists
    test_df = test_df[test_df["full_path"].notna()].copy()

    if test_df.empty:
        raise RuntimeError("After resolving paths from id, no test image files were found on disk.")

    return test_df


# -------------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------------

def make_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def evaluate_loader(model: nn.Module,
                    loader: DataLoader,
                    device: torch.device,
                    use_log: bool) -> dict:
    """
    Evaluate model on a DataLoader and return metrics on total counts.
    Handles:
      - models that output log(count+1) (use_log=True)
      - models that output counts directly (use_log=False)
    """
    model.eval()
    preds_all = []
    labels_all = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()

            out = model(imgs)
            preds = out[0] if isinstance(out, tuple) else out

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

            # Convert to counts if needed
            if use_log:
                preds_log = preds_m.clamp(min=0.0, max=15.0)
                preds_counts = torch.expm1(preds_log)
            else:
                preds_counts = preds_m

            labels_counts = labels_m

            preds_counts = torch.nan_to_num(preds_counts, nan=0.0, posinf=1e6, neginf=-1e6)
            labels_counts = torch.nan_to_num(labels_counts, nan=0.0, posinf=1e6, neginf=-1e6)

            p = preds_counts.cpu().float()
            l = labels_counts.cpu().float()
            p_tot = p.sum(dim=1).numpy() if p.ndim == 2 else p.numpy()
            l_tot = l.sum(dim=1).numpy() if l.ndim == 2 else l.numpy()

            preds_all.append(p_tot)
            labels_all.append(l_tot)

    if not preds_all:
        return {k: float("nan") for k in ("mae", "mse", "rmse", "mape", "acp")}

    preds_all = np.concatenate(preds_all, axis=0)
    labels_all = np.concatenate(labels_all, axis=0)
    return compute_metrics(preds_all, labels_all)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate model by staining on test set.")
    parser.add_argument("--dataset-root", type=str, default="dataset",
                        help="Root directory containing img/, ground_truth/, metadata.csv.")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to trained model weights (.pth).")
    parser.add_argument("--model-type", type=str,
                        choices=["baseline", "switch", "mmnet"],
                        default="mmnet",
                        help="Which model architecture to instantiate.")
    parser.add_argument("--num-classes", type=int, default=None,
                        help="Override auto-detected number of classes.")
    parser.add_argument("--use-log", action="store_true",
                        help="Indicates the model outputs log(count+1).")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--staining-col", type=str, default="staining",
                        help="Column in metadata.csv indicating staining type.")
    parser.add_argument("--set-col", type=str, default="set",
                        help="Column in metadata.csv indicating split (e.g., trainval/test).")
    parser.add_argument("--test-values", type=str, default="test,testing",
                        help="Comma-separated values in set-col that define the test set.")
    parser.add_argument("--out-csv", type=str, default="staining_metrics.csv",
                        help="Where to save per-staining metrics CSV.")
    args = parser.parse_args()

    set_random_seeds(args.seed)

    dataset_root = args.dataset_root
    test_values = [v.strip() for v in args.test_values.split(",") if v.strip()]

    # Load test metadata + paths + staining
    test_df = get_test_df_by_staining(
        dataset_root=dataset_root,
        staining_col=args.staining_col,
        set_col=args.set_col,
        test_values=test_values,
    )

    # Decide num_classes
    if args.num_classes is not None:
        num_classes = args.num_classes
    else:
        class_map = detect_class_map(dataset_root)
        num_classes = len(class_map) if class_map is not None else 1

    # Instantiate model
    if args.model_type == "mmnet":
        print(f"[INFO] Using MMNetCellCounter (num_classes={num_classes})")
        model = MMNetCellCounter(in_channels=3, num_classes=num_classes)
    elif args.model_type == "switch":
        print(f"[INFO] Using SwitchCNN (num_classes={num_classes})")
        model = SwitchCNN(num_classes=num_classes)
    else:
        print(f"[INFO] Using baseline CellCounter (num_classes={num_classes})")
        model = CellCounter()

    # Load weights
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    state_dict = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    print(f"[INFO] Loaded model weights from {args.model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    transform = make_transform()
    pin = torch.cuda.is_available()

    stain_col = args.staining_col.lower()
    stainings = sorted(test_df[stain_col].dropna().unique().tolist())
    results = []

    print(f"[INFO] Found {len(stainings)} staining types in test set: {stainings}")

    # Per-staining evaluation
    for stain in stainings:
        sub_df = test_df[test_df[stain_col] == stain]
        paths = [Path(p) for p in sub_df["full_path"].tolist()]

        if not paths:
            continue

        print(f"\n[INFO] Evaluating staining='{stain}' with {len(paths)} images...")

        dataset = CellDataset(paths, transform=transform, class_map=None)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=pin,
        )

        metrics = evaluate_loader(model, loader, device, use_log=args.use_log)
        print(f"  MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  "
              f"MAPE={metrics['mape']:.2f}%  ACP={metrics['acp']:.2f}%")

        results.append({
            "staining": stain,
            "n_images": len(paths),
            **metrics,
        })

    # Overall metrics on entire test set
    print("\n[INFO] Evaluating overall test metrics (all stainings combined)...")
    all_paths = [Path(p) for p in test_df["full_path"].tolist()]
    dataset_all = CellDataset(all_paths, transform=transform, class_map=None)
    loader_all = DataLoader(
        dataset_all,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=pin,
    )
    overall_metrics = evaluate_loader(model, loader_all, device, use_log=args.use_log)
    print(f"[OVERALL] MAE={overall_metrics['mae']:.4f}  RMSE={overall_metrics['rmse']:.4f}  "
          f"MAPE={overall_metrics['mape']:.2f}%  ACP={overall_metrics['acp']:.2f}%")

    results.append({
        "staining": "ALL",
        "n_images": len(all_paths),
        **overall_metrics,
    })

    # Save to CSV
    out_path = Path(args.out_csv)
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"[INFO] Saved per-staining metrics to {out_path}")


if __name__ == "__main__":
    main()
