'''
File: main.py
Author: Abdurahman Mohammed
Edited by: Vincent Lindvall
Date: 2025-12-02
Description: A Python script that trains a cell counting model
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset_handler import CellDataset
from model import CellCounter, SwitchCNN
from pathlib import Path                                                           # <-- CHANGED
import pandas as pd                                                                # <-- CHANGED
from typing import List
import os                                                            # <-- CHANGED
import random

def _image_paths_from_metadata(root: str, include_sets: List[str]) -> list: # <-- CHANGED
    root = Path(root)
    meta = root / "metadata.csv"
    img_dir = root / "img"
    if meta.exists():
        df = pd.read_csv(meta)
        df.columns = [c.strip().lower() for c in df.columns]
        if "id" not in df.columns or "set" not in df.columns:
            raise ValueError("metadata.csv must have at least 'id' and 'set' columns.")
        df = df[df["set"].isin(include_sets)]
        #df = df[df["staining"].isin(["DAPI"])]
        ids = df["id"].astype(str).tolist()
        paths = []
        for i in ids:
            p_tif  = img_dir / f"{i}.tif"
            p_tiff = img_dir / f"{i}.tiff"
            if p_tif.exists():
                paths.append(str(p_tif))
            elif p_tiff.exists():
                paths.append(str(p_tiff))
        return paths
    else:
        # fallback: no metadata.csv, just glob everything      # <-- CHANGED
        return sorted(glob(str(img_dir / "*.tif"))) + sorted(glob(str(img_dir / "*.tiff")))

from torch.cuda.amp import GradScaler, autocast

def entropy_loss(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean entropy (maximize by adding with a negative sign or minimize negative entropy)."""
    return -(p * (p + eps).log()).sum(dim=1).mean()

class UsageTracker:
    """Tracks moving-average expert usage for balance regularization."""
    def __init__(self, K: int, momentum: float = 0.9):
        self.K = K
        self.m = momentum
        self.u = torch.full((K,), 1.0 / K)

    @torch.no_grad()
    def update(self, probs: torch.Tensor):
        # probs: (B, K) on any device
        pm = probs.detach().mean(dim=0).cpu()
        self.u = self.m * self.u + (1.0 - self.m) * pm

    def loss(self) -> torch.Tensor:
        target = torch.full((self.K,), 1.0 / self.K)
        return (self.u - target).abs().mean()
    
def detect_class_map(dataset_root: str | Path):
    """
    Scan ground_truth CSV files and return a dict mapping label -> index.
    If no label/type column is found, return None.
    """
    dataset_root = Path(dataset_root)
    gt_dir = dataset_root / "ground_truth"
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
                # add non-null values
                vals = df[col].dropna().unique().tolist()
                labels.update(vals)
                break

    if not labels:
        return None

    # make deterministic mapping
    labels = sorted(list(labels))
    return {lab: i for i, lab in enumerate(labels)}

# -----------------------------------------------------------------------------

def get_data_loaders(batch_size=8):
    '''
    Creates training and validation data loaders for the IDCIA dataset. <-- For Fall 2025 our dataset changed

    Args:
        batch_size (int): The batch size for the data loaders.

    Returns:
        train_loader (DataLoader): A DataLoader object for the training set.
        val_loader (DataLoader): A DataLoader object for the validation set.
    '''

    dataset_root = "dataset"                                                     # <-- CHANGED

    trainval_paths = _image_paths_from_metadata(dataset_root, include_sets=["trainval"])  # <-- CHANGED
    test_paths   = _image_paths_from_metadata(dataset_root, include_sets=["test"])        # <-- CHANGED

    # -- Add Augmented Images --
    # Specify what augmented image folders to include by commenting / un-commenting
    augmented_image_folders_to_use = [ 
        #"v_flip",
        #"h_flip",
        #"0_point_5_contrast",
        #"1_point_5_contrast",
        #"blurred",
        #"sharpened",
        #"h_and_v_flip",
        #"v_flip_and_1_point_5_contrast",
    ]
    
    for augmented_image_folder in augmented_image_folders_to_use:
        trainval_paths.extend(glob(f"augmented_dataset/{augmented_image_folder}/img/*.tiff"))

    # Randomize order of training/validation images
    random.seed(3)
    random.shuffle(trainval_paths)

    # Split the trainval images into two lists
    n = len(trainval_paths)
    split = max(1, int(0.05 * n))
    val_paths = trainval_paths[:split]
    train_paths = trainval_paths[split:]

    # Define transforms (unchanged)
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

    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Create datasets
    train_dataset = CellDataset(train_paths, transform=train_transform)          # <-- CHANGED (paths var)
    val_dataset   = CellDataset(val_paths,   transform=val_transform)            # <-- CHANGED
    test_dataset  = CellDataset(test_paths,   transform=test_transform)

    # Create dataloaders
    use_cuda = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=use_cuda)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=use_cuda)
    test_loader   = DataLoader(test_dataset,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=use_cuda)

    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.L1Loss()  # MAE for counts
    # --- two param groups: router a bit faster ---
    router_params = []
    base_params = []
    for n, p in model.named_parameters():
        (router_params if "router" in n else base_params).append(p)
    optimizer = optim.AdamW(
        [{"params": base_params, "lr": learning_rate},
         {"params": router_params, "lr": learning_rate * 2.0}],
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, num_epochs))
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # regularizers
    LAMBDA_ENT = 0.005     # much smaller -> don't force uniform
    LAMBDA_BAL = 0.0       # start at 0; we can turn it on later if needed
    LAMBDA_CE  = 0.2       # router pseudo-label supervision (key fix)
    WARMUP_EPOCHS = 6

    tracker = UsageTracker(K=3)
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        running = 0.0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda")):
                out = model(imgs)

                # Handle CellCounter vs SwitchCNN
                if isinstance(out, tuple) and len(out) >= 5:
                    # SwitchCNN path
                    preds, probs, tau, logits, cols = out   # preds:(B,C), probs:(B,3), cols:(B,3,C)
                    # shape align
                    if preds.ndim == 1 and labels.ndim == 2: preds = preds.unsqueeze(1)
                    if preds.ndim == 2 and labels.ndim == 1: labels = labels.unsqueeze(1)

                    # primary count loss
                    loss = criterion(preds, labels)

                    # --- pseudo-label CE (which column is best for this sample?) ---
                    # per-column L1 to labels → argmin target
                    # Reduce across class dim C to a scalar per column
                    col_mae = (cols - labels.unsqueeze(1)).abs().mean(dim=-1)   # (B,3)
                    targets = torch.argmin(col_mae, dim=1)                      # (B,)
                    ce = F.cross_entropy(logits, targets)
                    loss = loss + LAMBDA_CE * ce

                    # small entropy bonus to avoid early peaky behavior
                    ent = entropy_loss(probs)
                    loss = loss + LAMBDA_ENT * ent

                    # optional balance (keep off initially)
                    if LAMBDA_BAL > 0 and (epoch + 1) > WARMUP_EPOCHS:
                        loss = loss + LAMBDA_BAL * tracker.loss()

                else:
                    # CellCounter path (out is (B,) or (B,1))
                    preds = out
                    if preds.ndim == 1 and labels.ndim == 2: preds = preds.unsqueeze(1)
                    if preds.ndim == 2 and labels.ndim == 1: labels = labels.unsqueeze(1)
                    loss = criterion(preds, labels)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()

            running += float(loss)

            # update usage if we have probs
            if isinstance(out, tuple) and len(out) >= 5:
                tracker.update(probs.detach())

        train_loss = running / max(1, len(train_loader))
        train_losses.append(train_loss)

        # validate
        val_loss = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        scheduler.step()

        u_str = ", ".join([f"{x:.2f}" for x in tracker.u.tolist()])
        print(f"Epoch {epoch+1:03d} | train={train_loss:.4f} | val={val_loss:.4f} | usage={u_str}")

        # Save checkpoint every 5 epochs
        os.makedirs("checkpoints", exist_ok=True)
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"checkpoints/epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
            
        # Optional: anneal router temperature a bit each epoch (nudges away from 1/3)
        if hasattr(model, "router"):
            with torch.no_grad():
                tau_now = model.router.log_tau.exp().clamp(0.7, 5.0)  # floor @ 0.7
                model.router.log_tau.copy_(tau_now.log())

    return model, train_losses, val_losses



def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(data_loader)

def plot_losses(train_losses, val_losses):
    '''
    Plots the training and validation losses.
    '''
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.close()

def main():
    # Set hyperparameters
    batch_size = 8
    num_epochs = 10
    learning_rate = 1e-3

    # Get data loaders
    train_loader, val_loader, test_loader = get_data_loaders(batch_size)

    # Detect number of classes and create the appropriate model
    dataset_root = Path("dataset")
    class_map = detect_class_map(dataset_root)
    num_classes = 1 if class_map is None else len(class_map)

    if num_classes == 1:
        model = CellCounter()
    else:
        model = SwitchCNN(num_classes=num_classes)

    # Train the model
    trained_model, train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs, learning_rate)

    # Plot the training and validation losses
    plot_losses(train_losses, val_losses)

    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.L1Loss()
    test_loss = evaluate_model(trained_model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")

    # Save the model
    torch.save(trained_model.state_dict(), "cell_counter.pth")

if __name__ == "__main__":
    main()
