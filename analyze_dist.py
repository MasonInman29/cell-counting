# analyze_distribution.py
import torch
import numpy as np
from train_scnn import get_data_loaders

def summarize(loader, name):
    totals = []
    for imgs, labels in loader:
        # labels: (B,) or (B,C)
        labels = labels.view(labels.size(0), -1)
        total = labels.sum(dim=1).numpy()
        totals.append(total)
    totals = np.concatenate(totals, axis=0)

    print(f"{name}: n={len(totals)}")
    print(f"  mean={totals.mean():.2f} std={totals.std():.2f}")
    print(f"  min={totals.min():.2f} max={totals.max():.2f}")

    # crude bins: tweak to match your dataset
    bins = [0, 50, 100, 200, 400, 800, 1e9]
    hist, edges = np.histogram(totals, bins=bins)
    print("  bins:", edges)
    print("  hist:", hist)

if __name__ == "__main__":
    train_loader, val_loader, class_map = get_data_loaders(
        dataset_root="../dataset", batch_size=32, val_fraction=0.2, seed=42
    )
    summarize(train_loader, "train")
    summarize(val_loader, "val")


    print(f"Augmented Dataset:")
    train_loader, val_loader, class_map = get_data_loaders(
        dataset_root="./final_augmented_dataset", batch_size=32, val_fraction=0.2, seed=42
    )
    summarize(train_loader, "train")
    summarize(val_loader, "val")
