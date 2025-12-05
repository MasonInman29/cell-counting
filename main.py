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
from model import CellCounter
from pathlib import Path                                                           # <-- CHANGED
import pandas as pd                                                                # <-- CHANGED
from typing import List                                                            # <-- CHANGED
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
    '''
    Trains the cell counting model on the training set and evaluates it on the validation set.
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}, Training Loss: {epoch_loss:.4f}")

        # Evaluate the model on the validation set
        val_loss = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        print(f"Validation Loss: {val_loss:.4f}")

    return model, train_losses, val_losses

def evaluate_model(model, data_loader, criterion, device):
    '''
        Evaluates the model on a validation or test set.
    '''
    model.eval()
    total_loss = 0.0
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

    # Create the model
    model = CellCounter()

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
